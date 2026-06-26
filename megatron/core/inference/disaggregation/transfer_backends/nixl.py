# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct NIXL backend for disaggregated prefill/decode KV transfer.

API reference (validated against ``nixl_cu12`` 1.2.0 and Dynamo's
``components/src/dynamo/common/multimodal/embedding_transfer.py``):

- ``agent = nixl_agent(name)`` — local NIXL agent.
- ``agent.register_memory(tensor)`` — register a torch tensor; returns a
  registration handle. NIXL infers ``VRAM`` from CUDA tensors automatically.
- ``agent.get_agent_metadata() -> bytes`` — serialized connection blob to ship
  to peers out-of-band. We carry it inside ``disaggregated_params["kv_meta"]``
  base64-encoded.
- ``agent.add_remote_agent(metadata_bytes) -> remote_agent_id`` — call once per
  peer before any transfer to it.
- ``agent.get_xfer_descs(list_of_(ptr, size, device_id), mem_type=...)`` —
  build a descriptor list for sub-ranges of the registered buffer.
- ``agent.initialize_xfer("READ"|"WRITE", local_descs, remote_descs, remote_agent_id)``
  → xfer handle. For pulls, "READ" with local_descs as destination and
  remote_descs as source.
- ``agent.transfer(xfer_handle)`` — submit.
- ``agent.check_xfer_state(xfer_handle) -> "DONE"|"ERR"|...`` — poll.
- ``agent.deregister_memory(handle)`` — cleanup.

Phase-3 wiring: each rank constructs a single agent, registers its paged
``DynamicInferenceContext.memory_buffer``, and exposes the agent metadata via
:meth:`KvTransferAgent.export_meta`. The prefill engine ships that dict in the
reply; the decode engine receives it, calls :meth:`add_remote_agent` once,
then :meth:`pull_blocks` for each handoff request.
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


# Polling cadence + safety cap for `check_xfer_state`. NIXL doesn't expose a
# blocking wait; we spin with a tight initial interval (transfer latency on
# NVLink is sub-millisecond) and back off slightly. 30-second cap is well
# beyond any plausible per-block transfer on healthy hardware — a longer
# stall means something is wrong (peer crashed, fabric drop, etc.).
_POLL_INTERVAL_S = 0.0005  # 0.5 ms
_POLL_TIMEOUT_S = 30.0


def have_nixl() -> bool:
    return _HAVE_NIXL


class NixlTransferBackend:
    """Per-rank NIXL agent owning a registration over the paged KV buffer.

    The whole ``memory_buffer`` is registered once at startup. Per-block
    transfers use ``get_xfer_descs`` to describe sub-ranges by ``(ptr, size,
    device_id)`` triples; NIXL handles the actual RDMA/NVLink path.

    Peer discovery is out-of-band: the prefill engine returns this rank's
    agent metadata inside ``disaggregated_params``, the decode engine calls
    :meth:`pull_blocks` which lazily registers the peer via
    ``add_remote_agent`` on first use.
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

        # TP topology — needed for heterogeneous-TP KV re-sharding (prefill_TP !=
        # decode_TP). When any of these is None the agent only supports matched
        # layouts (the original fast path); export_meta() omits the topology
        # block and pull_blocks() falls back to the whole-slice copy.
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
        # PP topology — used for heterogeneous-PP KV layer re-sharding.
        # layer_start/layer_end are global attention-layer indices (exclusive end)
        # owned by this PP rank. None → PP=1 or unknown, no layer subsetting.
        self._pp_rank = pp_rank
        self._layer_start = layer_start
        self._layer_end = layer_end

        # Locate the blocks axis. Megatron has two layouts:
        #   - MLA:        [layers, blocks, block_size, kv_dim]            → axis 1
        #   - K/V split:  [2, layers, blocks, block_size, n_kv_heads, d]  → axis 2
        # We find it by matching the expected block count rather than
        # hardcoding a position, so future layouts (e.g. EP/PP variations)
        # don't silently mis-identify the layer dim as blocks. Caller passes
        # `kv_block_allocator.total_count` as the source of truth.
        shape = list(memory_buffer.shape)
        candidates = [i for i, dim in enumerate(shape) if dim == expected_num_blocks]
        if not candidates:
            raise RuntimeError(
                f"NixlTransferBackend: no axis in memory_buffer shape {shape} "
                f"matches expected_num_blocks={expected_num_blocks}. Layout "
                "is unrecognized — bug in caller or new Megatron tensor shape."
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

        # The KV buffer is contiguous in C-order. For each (outer-index combo,
        # block i), there is ONE contiguous slice of `bytes_per_slice` bytes
        # containing every position×head×dim element for that (outer, block).
        # Walking the blocks-axis advances one slice; walking the next outer
        # axis advances `num_blocks` slices (one full B-stride). For Megatron's
        # K/V split layout [2, L, B, T, H, d] this gives 2*L slices per block.
        # For MLA [L, B, T, D] it's L slices per block.
        shape = list(memory_buffer.shape)
        elements_per_slice = 1
        for dim in shape[self._blocks_axis + 1 :]:
            elements_per_slice *= dim
        self._bytes_per_slice = self._element_size * elements_per_slice
        num_outer = 1
        for dim in shape[: self._blocks_axis]:
            num_outer *= dim
        self._num_outer = num_outer
        # Bytes to skip to advance one outer-index combination (jumps over a
        # whole B-stride of slices).
        self._outer_stride_bytes = self._num_blocks * self._bytes_per_slice
        # Total bytes per logical block (informational).
        self._per_block_bytes = self._num_outer * self._bytes_per_slice

        # Topology snapshot — passed to kv_reshard_plan functions at transfer time.
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

        # UCX transport selection — must happen before the nixl_agent() constructor
        # creates the UCX worker context.
        #
        # Without this, UCX may wire VRAM-to-VRAM transfers over its TCP transport.
        # The TCP handler (uct_tcp_ep_am_bcopy) does a CPU memcpy from the source
        # address; on aarch64 that crashes with SIGSEGV when the source is VRAM.
        #
        #   cuda_ipc  — direct GPU P2P via NVLink / PCIe P2P (zero-copy preferred path)
        #   cuda_copy — CUDA-mediated staging through host memory (safe fallback)
        #   cma       — cross-memory attach for host-memory control messages
        #   shm       — POSIX shm for intra-node host memory
        #   self      — loopback, needed for NIXL's internal operations
        #
        # setdefault: the operator can override by setting UCX_TLS before launch.
        # TCP is not listed; if cuda_ipc/cuda_copy are unavailable (e.g. missing
        # UCX CUDA backends), NIXL will raise a transport error at connection time
        # rather than silently falling back to TCP and crashing later.
        os.environ.setdefault("UCX_TLS", "cuda_ipc,cuda_copy,cma,shm,self")
        # Disable the UCX memory-type cache: the cache is populated during early
        # UCX init before CUDA is fully ready, and can misclassify VRAM addresses
        # as host memory.  With explicit register_memory() we pay no lookup cost.
        os.environ.setdefault("UCX_MEMTYPE_CACHE", "n")

        self._agent = nixl_agent(agent_name)
        # Pass the torch tensor directly; NIXL detects VRAM + computes the
        # descriptor list internally. Returns a registration handle we hold
        # for the agent's lifetime.
        self._reg_handle = self._agent.register_memory(memory_buffer)

        # Cache our own metadata bytes for export_meta(). NIXL's
        # add_remote_agent() on the peer side requires raw bytes; we
        # base64-encode for msgpack-safe transport in disaggregated_params.
        self._agent_metadata = self._agent.get_agent_metadata()

        # Peer agent_name -> the id returned by add_remote_agent (often the
        # peer's name string). Used to short-circuit duplicate registration.
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

        The receiver passes ``agent_metadata_b64`` to :meth:`pull_blocks` (via
        ``kv_meta``); we decode and register it lazily on first transfer.

        ``bytes_per_slice``, ``num_outer``, ``outer_stride_bytes`` capture the
        scatter-gather layout: each block is ``num_outer`` non-contiguous
        slices of ``bytes_per_slice`` bytes, separated by ``outer_stride_bytes``.
        Peer + local must agree on all of these (enforced in pull_blocks).
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
        # Topology block — only present when the agent was built with TP info.
        # Its presence is what lets a decode peer re-shard a differently-TP'd
        # prefill buffer (see pull_blocks). Absent → matched-layout only.
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
        # PP topology — present when built with layer_start/layer_end. Allows a
        # decode peer at a different PP size to select only the outer (layer)
        # slices it owns from each prefill PP rank's buffer.
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
        # NIXL's add_remote_agent returns the peer agent name in some
        # versions, an opaque id in others. Either way, store it for use in
        # initialize_xfer().
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
                list of such dicts (one per prefill TP rank) when prefill and
                decode run at different TP. For heterogeneous-PP handoffs this is
                ``{"pp_metas": [{"tp_metas": ..., "block_ids": [...]}, ...]}``.
            src_block_ids: Block ids on the peer(s). Unused for the PP path (each
                PP entry carries its own block_ids); required for TP-only paths.
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

        Two execution paths driven by ``seg.n_heads``:

        - ``n_heads == 0``: full-slice matched copy. One descriptor per
          ``(dst_block, outer)`` pair. Uses the peer's own stride/size values for
          src so the method is correct even when prefill and decode have different
          block pool sizes (different ``outer_stride_bytes``).

        - ``n_heads > 0``: head sub-range copy. One descriptor per
          ``(dst_block, outer, token)`` triple. Used when TP differs between
          prefill and decode so each token's ``[H, d]`` slice must be partially
          copied.
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
            # Matched layout: copy one full bytes_per_slice per (block, outer).
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
            # Head sub-range: copy n_heads×head_dim bytes per token.
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
        # READ pulls remote → local. NIXL's signature is (op, LOCAL, REMOTE, peer).
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
    listen_addr: Optional[str],  # accepted but unused — NIXL does its own discovery
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

    ``listen_addr`` is kept in the signature for launcher compatibility but is
    ignored by NIXL — peer discovery is metadata-based (see module docstring).
    Callers pass it as a flag to indicate "transfer enabled" by setting any
    non-empty string.

    The ``tp_*`` / head / block args enable heterogeneous-TP re-sharding; pass
    them from the engine's parallel state + KV context. When omitted the agent
    only supports matched layouts.

    The ``pp_*`` / layer args enable heterogeneous-PP layer re-sharding. Pass
    ``pp_rank``, ``layer_start``, and ``layer_end`` (global attention-layer
    indices, inclusive start / exclusive end) so the agent can expose its layer
    partition to decode peers and select the right outer-slice range when pulling.
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
