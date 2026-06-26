# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct NIXL backend for disaggregated prefill/decode KV transfer.

Each rank registers its paged KV buffer once, exports NIXL peer metadata, and
the decode side pulls source block ranges directly into its local KV blocks.

Backend selection belongs in ``transfer_backends.base``. A future NCCL backend
can be registered there and selected by launcher config, e.g.
``MEGATRON_KV_TRANSFER_BACKEND=nccl``, without changing this NIXL backend.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from typing import Any, Dict, List, Optional

import torch

from megatron.core.inference.kv_reshard_plan import (
    KvTopology,
    TransferSegment,
    build_reshard_plan,
)

logger = logging.getLogger(__name__)

try:
    from nixl._api import nixl_agent  # type: ignore[import-not-found]

    _HAVE_NIXL = True
except ImportError:
    nixl_agent = None  # type: ignore[assignment]
    _HAVE_NIXL = False


# NIXL exposes polling, not a blocking wait. A long stall usually means peer or
# fabric failure, so cap the wait.
_POLL_INTERVAL_S = 0.0005  # 0.5 ms
_POLL_TIMEOUT_S = 30.0


def have_nixl() -> bool:
    return _HAVE_NIXL


class NixlTransferBackend:
    """Per-rank NIXL agent owning a registration over the paged KV buffer.

    Per-block transfers are descriptor ranges over that registration. Peer
    metadata is exchanged by the control plane and registered lazily on first
    pull.
    """

    name = "nixl"

    def __init__(
        self,
        agent_name: str,
        memory_buffer: torch.Tensor,
        expected_num_blocks: int,
        tp_size: Optional[int] = None,
        tp_rank: Optional[int] = None,
        num_kv_heads_global: Optional[int] = None,
        heads_per_partition: Optional[int] = None,
        head_dim: Optional[int] = None,
        tokens_per_block: Optional[int] = None,
        pp_rank: Optional[int] = None,
        layer_start: Optional[int] = None,
        layer_end: Optional[int] = None,
    ):
        if not _HAVE_NIXL:
            raise RuntimeError(
                "NixlTransferBackend requires the nixl Python package. Install the "
                "NIXL runtime and `pip install nixl` before launching "
                "disaggregated workers."
            )
        self.agent_name = agent_name
        self._memory_buffer = memory_buffer

        # TP topology enables heterogeneous-TP KV re-sharding. If incomplete,
        # only matched layouts are supported.
        self._tp_size = tp_size
        self._tp_rank = tp_rank
        self._num_kv_heads_global = num_kv_heads_global
        self._heads_per_partition = heads_per_partition
        self._head_dim = head_dim
        self._tokens_per_block = tokens_per_block
        self._reshard_capable = None not in (
            tp_size,
            tp_rank,
            num_kv_heads_global,
            heads_per_partition,
            head_dim,
            tokens_per_block,
        )
        # PP topology enables layer subsetting across heterogeneous PP layouts.
        self._pp_rank = pp_rank
        self._layer_start = layer_start
        self._layer_end = layer_end

        # Locate the blocks axis by allocator size instead of layout position.
        # Current layouts are MLA [L, B, T, D] and K/V split [2, L, B, T, H, d].
        shape = list(memory_buffer.shape)
        candidates = [i for i, dim in enumerate(shape) if dim == expected_num_blocks]
        if not candidates:
            raise RuntimeError(
                f"NixlTransferBackend: no axis in memory_buffer shape {shape} "
                f"matches expected_num_blocks={expected_num_blocks}. Layout "
                "is unrecognized; bug in caller or new Megatron tensor shape."
            )
        if len(candidates) > 1:
            raise RuntimeError(
                f"NixlTransferBackend: ambiguous blocks axis in shape {shape} "
                f"(expected_num_blocks={expected_num_blocks} matches multiple "
                f"axes {candidates}). Caller must pass a more distinctive value."
            )
        self._blocks_axis = candidates[0]
        self._num_blocks = expected_num_blocks
        self._buf_ptr = memory_buffer.data_ptr()
        self._buf_numel = memory_buffer.numel()
        self._element_size = memory_buffer.element_size()
        self._device_id = (
            memory_buffer.device.index if memory_buffer.is_cuda else 0
        )

        # Each (outer, block) pair is one contiguous slice. The outer stride
        # skips over the full block pool for that outer index.
        shape = list(memory_buffer.shape)
        elements_per_slice = 1
        for dim in shape[self._blocks_axis + 1 :]:
            elements_per_slice *= dim
        self._bytes_per_slice = self._element_size * elements_per_slice
        num_outer = 1
        for dim in shape[: self._blocks_axis]:
            num_outer *= dim
        self._num_outer = num_outer
        self._outer_stride_bytes = self._num_blocks * self._bytes_per_slice
        self._per_block_bytes = self._num_outer * self._bytes_per_slice

        # Snapshot used by kv_reshard_plan at transfer time.
        self._topology = KvTopology(
            tp_size=tp_size,
            tp_rank=tp_rank,
            num_kv_heads_global=num_kv_heads_global,
            heads_per_partition=heads_per_partition,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            pp_rank=pp_rank,
            layer_start=layer_start,
            layer_end=layer_end,
            num_outer=self._num_outer,
        )

        # Configure UCX before agent construction. Avoid TCP for VRAM addresses;
        # operators may override this by setting UCX_TLS before launch.
        os.environ.setdefault("UCX_TLS", "cuda_ipc,cuda_copy,cma,shm,self")
        # Explicit registration makes the UCX memtype cache unnecessary and
        # avoids stale VRAM/host classifications.
        os.environ.setdefault("UCX_MEMTYPE_CACHE", "n")

        self._agent = nixl_agent(agent_name)
        self._reg_handle = self._agent.register_memory(memory_buffer)

        # Base64 keeps NIXL metadata safe for msgpack/json control messages.
        self._agent_metadata = self._agent.get_agent_metadata()

        # Peer agent_name -> id returned by add_remote_agent.
        self._known_peers: Dict[str, Any] = {}

        logger.info(
            "NixlTransferBackend[%s] registered %d-block buffer "
            "(blocks_axis=%d, %d outer-slices/block × %d bytes/slice = "
            "%d bytes/block, device=%d, shape=%s)",
            agent_name,
            self._num_blocks,
            self._blocks_axis,
            self._num_outer,
            self._bytes_per_slice,
            self._per_block_bytes,
            self._device_id,
            shape,
        )

    def export_meta(self) -> Dict[str, Any]:
        """Return JSON/msgpack-safe metadata for shipping to a decode peer.

        Layout fields describe the scatter-gather address ranges needed to pull
        source blocks into decode-owned blocks.
        """
        meta = {
            "agent_name": self.agent_name,
            "agent_metadata_b64": base64.b64encode(self._agent_metadata).decode(
                "ascii"
            ),
            "base_addr": self._buf_ptr,
            "bytes_per_slice": self._bytes_per_slice,
            "num_outer": self._num_outer,
            "outer_stride_bytes": self._outer_stride_bytes,
            "num_blocks": self._num_blocks,
            "device_id": self._device_id,
            "blocks_axis": self._blocks_axis,
        }
        # TP topology is included only when heterogeneous-TP transfer is possible.
        if self._reshard_capable:
            meta.update(
                {
                    "tp_size": self._tp_size,
                    "tp_rank": self._tp_rank,
                    "num_kv_heads_global": self._num_kv_heads_global,
                    "heads_per_partition": self._heads_per_partition,
                    "head_dim": self._head_dim,
                    "tokens_per_block": self._tokens_per_block,
                }
            )
        # PP topology lets decode select the layer slices it owns.
        if self._layer_start is not None and self._layer_end is not None:
            meta.update(
                {
                    "pp_rank": self._pp_rank,
                    "layer_start": self._layer_start,
                    "layer_end": self._layer_end,
                }
            )
        return meta

    def _ensure_peer_registered(self, peer_meta: Dict[str, Any]) -> str:
        """Register the peer with NIXL on first use; return its agent id."""
        peer_name = peer_meta["agent_name"]
        existing = self._known_peers.get(peer_name)
        if existing is not None:
            return existing
        metadata_b64 = peer_meta.get("agent_metadata_b64")
        if not metadata_b64:
            raise ValueError(
                f"peer_meta for {peer_name!r} is missing agent_metadata_b64"
            )
        peer_id = self._agent.add_remote_agent(base64.b64decode(metadata_b64))
        # NIXL versions return either the peer name or an opaque id.
        resolved = peer_id if peer_id else peer_name
        self._known_peers[peer_name] = resolved
        logger.info(
            "NixlTransferBackend[%s] registered peer %s", self.agent_name, peer_name
        )
        return resolved

    def pull_blocks(
        self,
        peer_meta: Any,
        src_block_ids: List[int],
        dst_block_ids: List[int],
    ) -> None:
        """Synchronously pull blocks from one or more peer agents into local blocks.

        Args:
            peer_meta: an ``export_meta()`` dict from a single prefill peer, OR a
                list of dicts for heterogeneous TP. Heterogeneous PP uses
                ``{"pp_metas": [{"tp_metas": ..., "block_ids": [...]}, ...]}``.
            src_block_ids: Source block ids unless carried by ``pp_metas``.
            dst_block_ids: Local block ids to write into.
        """
        if not isinstance(peer_meta, dict) or "pp_metas" not in peer_meta:
            if len(src_block_ids) != len(dst_block_ids):
                raise ValueError(
                    f"src/dst block_id length mismatch: "
                    f"{len(src_block_ids)} vs {len(dst_block_ids)}"
                )
            if not src_block_ids:
                return

        segments = build_reshard_plan(peer_meta, src_block_ids, self._topology)
        for seg in segments:
            self._execute_segment(seg, dst_block_ids)

    def _await_xfer(self, xfer: Any, ctx: str) -> None:
        """Submit + spin-poll a NIXL transfer to completion."""
        self._agent.transfer(xfer)
        deadline = time.perf_counter() + _POLL_TIMEOUT_S
        while True:
            state = self._agent.check_xfer_state(xfer)
            if state == "DONE":
                return
            if state == "ERR":
                raise RuntimeError(f"NIXL transfer failed ({ctx})")
            if time.perf_counter() > deadline:
                raise TimeoutError(
                    f"NIXL transfer timed out after {_POLL_TIMEOUT_S}s "
                    f"({ctx}, last_state={state})"
                )
            time.sleep(_POLL_INTERVAL_S)

    def _execute_segment(self, seg: TransferSegment, dst_block_ids: List[int]) -> None:
        """Execute one :class:`TransferSegment` via NIXL.

        ``n_heads == 0`` copies full slices. Otherwise, copy per-token head
        fragments for heterogeneous TP.
        """
        pm = seg.peer_meta
        peer_base = pm["base_addr"]
        peer_device_id = pm.get("device_id", 0)
        peer_bps = pm["bytes_per_slice"]
        peer_os = pm["outer_stride_bytes"]
        peer_id = self._ensure_peer_registered(pm)

        src_block_ids = seg.src_block_ids
        src_o_start = seg.src_o_start
        dst_o_start = seg.dst_o_start
        n_outer = seg.n_outer
        bps = self._bytes_per_slice
        local_os = self._outer_stride_bytes

        src_tuples: List[Any] = []
        dst_tuples: List[Any] = []

        if seg.n_heads == 0:
            # Full-slice copy: one descriptor per (block, outer).
            for src_b, dst_b in zip(src_block_ids, dst_block_ids):
                for i in range(n_outer):
                    src_o = src_o_start + i
                    dst_o = dst_o_start + i
                    src_tuples.append(
                        (peer_base + src_o * peer_os + src_b * peer_bps, peer_bps, peer_device_id)
                    )
                    dst_tuples.append(
                        (self._buf_ptr + dst_o * local_os + dst_b * bps, bps, self._device_id)
                    )
            ctx = (
                f"matched peer={peer_id} outer[{src_o_start}:+{n_outer}] "
                f"blocks={len(src_block_ids)}"
            )
        else:
            # Head sub-range copy: one descriptor per token.
            topo = self._topology
            d_bytes = topo.head_dim * self._element_size  # type: ignore[operator]
            local_token_stride = topo.heads_per_partition * d_bytes  # type: ignore[operator]
            peer_token_stride = pm["heads_per_partition"] * d_bytes
            T = topo.tokens_per_block
            frag_bytes = seg.n_heads * d_bytes
            src_h_off = seg.src_h0 * d_bytes
            dst_h_off = seg.dst_h0 * d_bytes

            for src_b, dst_b in zip(src_block_ids, dst_block_ids):
                for i in range(n_outer):
                    src_o = src_o_start + i
                    dst_o = dst_o_start + i
                    src_slice = peer_base + src_o * peer_os + src_b * peer_bps + src_h_off
                    dst_slice = self._buf_ptr + dst_o * local_os + dst_b * bps + dst_h_off
                    for t in range(T):
                        src_tuples.append(
                            (src_slice + t * peer_token_stride, frag_bytes, peer_device_id)
                        )
                        dst_tuples.append(
                            (dst_slice + t * local_token_stride, frag_bytes, self._device_id)
                        )
            ctx = (
                f"reshard peer={peer_id} outer[{src_o_start}:+{n_outer}] "
                f"heads[{seg.src_h0}:+{seg.n_heads}] blocks={len(src_block_ids)}"
            )

        src_descs = self._agent.get_xfer_descs(src_tuples, mem_type="VRAM")
        dst_descs = self._agent.get_xfer_descs(dst_tuples, mem_type="VRAM")
        # READ pulls remote -> local. Signature is (op, local, remote, peer).
        xfer = self._agent.initialize_xfer("READ", dst_descs, src_descs, peer_id)
        self._await_xfer(xfer, ctx)

    def close(self) -> None:
        if self._agent is None:
            return
        try:
            self._agent.deregister_memory(self._reg_handle)
        except Exception:  # noqa: BLE001 - shutdown path
            logger.exception("NixlTransferBackend: deregister_memory failed")
        self._agent = None


def make_agent(
    role: str,
    rank: int,
    listen_addr: Optional[str],  # accepted but unused; NIXL does its own discovery
    memory_buffer: torch.Tensor,
    expected_num_blocks: int,
    tp_size: Optional[int] = None,
    tp_rank: Optional[int] = None,
    num_kv_heads_global: Optional[int] = None,
    heads_per_partition: Optional[int] = None,
    head_dim: Optional[int] = None,
    tokens_per_block: Optional[int] = None,
    pp_rank: Optional[int] = None,
    layer_start: Optional[int] = None,
    layer_end: Optional[int] = None,
) -> Optional[NixlTransferBackend]:
    """Construct an agent, or return None when KV transfer is disabled.

    ``listen_addr`` is kept for launcher compatibility; NIXL uses metadata-based
    peer discovery. TP/PP arguments enable heterogeneous parallelism transfer.
    """
    if not listen_addr:
        return None
    agent_name = f"{role}-rank{rank}"
    return NixlTransferBackend(
        agent_name,
        memory_buffer,
        expected_num_blocks,
        tp_size=tp_size,
        tp_rank=tp_rank,
        num_kv_heads_global=num_kv_heads_global,
        heads_per_partition=heads_per_partition,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        pp_rank=pp_rank,
        layer_start=layer_start,
        layer_end=layer_end,
    )


# Backward-compatible name for callers/tests that still import the old agent.
KvTransferAgent = NixlTransferBackend
