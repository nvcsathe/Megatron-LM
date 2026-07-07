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
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import torch

from megatron.core.inference.disaggregation.kv_reshard import (
    KVBufferGeometry,
    KVShardLayout,
    TransferSegment,
    build_reshard_plan,
)
from megatron.core.inference.disaggregation.mamba_reshard import (
    MambaShardLayout,
    build_mamba_reshard_plan,
)
from megatron.core.inference.disaggregation.utils import transfer_peer_records

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


@dataclass
class NixlPullHandle:
    """Pollable handle for one logical pull made of one or more NIXL transfers."""

    agent: Any
    xfers: List[Any]
    contexts: List[str]
    submitted_at: float
    timeout_s: float = _POLL_TIMEOUT_S
    done: bool = False
    error: Optional[str] = None

    def poll(self) -> bool:
        if self.done:
            if self.error is not None:
                raise RuntimeError(self.error)
            return True
        if not self.xfers:
            self.done = True
            return True

        errors: List[str] = []
        pending: List[str] = []
        for xfer, ctx in zip(self.xfers, self.contexts):
            state = self.agent.check_xfer_state(xfer)
            if state == "DONE":
                continue
            if state == "ERR":
                errors.append(ctx)
                continue
            pending.append(f"{ctx}: {state}")

        if not pending:
            self.done = True
            if errors:
                self.error = f"NIXL transfer failed ({', '.join(errors)})"
                raise RuntimeError(self.error)
            return True
        if time.perf_counter() - self.submitted_at > self.timeout_s:
            raise TimeoutError(
                f"NIXL transfer timed out after {self.timeout_s}s; pending={pending}"
            )
        return False

    def wait(self) -> None:
        while not self.poll():
            time.sleep(_POLL_INTERVAL_S)


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
        global_rank: Optional[int] = None,
        pp_size: Optional[int] = None,
        pp_rank: Optional[int] = None,
        num_layers_global: Optional[int] = None,
        layer_start: Optional[int] = None,
        layer_end: Optional[int] = None,
        mamba_layout: Optional[MambaShardLayout] = None,
        mamba_state_kind: Optional[str] = None,
    ):
        if not _HAVE_NIXL:
            raise RuntimeError(
                "NixlTransferBackend requires the nixl Python package. Install the "
                "NIXL runtime and `pip install nixl` before launching "
                "disaggregated workers."
            )
        self.agent_name = agent_name
        self._memory_buffer = memory_buffer

        if (mamba_layout is None) != (mamba_state_kind is None):
            raise ValueError(
                "mamba_layout and mamba_state_kind must be provided together"
            )
        if mamba_state_kind not in (None, "conv", "ssm"):
            raise ValueError("mamba_state_kind must be 'conv' or 'ssm'")

        layout_capable = (
            None
            not in (
                global_rank,
                tp_size,
                tp_rank,
                pp_size,
                pp_rank,
                num_layers_global,
                num_kv_heads_global,
                heads_per_partition,
                head_dim,
                tokens_per_block,
                layer_start,
                layer_end,
            )
            and heads_per_partition * tp_size == num_kv_heads_global
        )  # type: ignore[operator]
        # Locate the blocks axis by allocator size instead of layout position.
        # The inference KV layout is [2, L, B, T, H, d].
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
        blocks_axis = candidates[0]
        self._buf_ptr = memory_buffer.data_ptr()
        self._element_size = memory_buffer.element_size()
        self._device_id = memory_buffer.device.index if memory_buffer.is_cuda else 0

        # Each (outer, block) pair is one contiguous slice. The outer stride
        # skips over the full block pool for that outer index.
        shape = list(memory_buffer.shape)
        elements_per_slice = 1
        for dim in shape[blocks_axis + 1 :]:
            elements_per_slice *= dim
        bytes_per_slice = self._element_size * elements_per_slice
        num_outer = 1
        for dim in shape[:blocks_axis]:
            num_outer *= dim
        self._outer_stride_bytes = expected_num_blocks * bytes_per_slice
        self._geometry = KVBufferGeometry(
            num_outer=num_outer,
            bytes_per_slice=bytes_per_slice,
            blocks_axis=blocks_axis,
            num_blocks=expected_num_blocks,
            heads_per_partition=heads_per_partition,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            element_size=self._element_size,
        )

        # Canonical KV layout. Mamba agents carry their separate typed layout.
        self._layout = None
        if layout_capable:
            self._layout = KVShardLayout(
                num_layers=int(num_layers_global),
                num_heads=int(num_kv_heads_global),
                tp_size=int(tp_size),
                tp_rank=int(tp_rank),
                pp_size=int(pp_size),
                pp_rank=int(pp_rank),
                global_rank=int(global_rank),
                layer_start=int(layer_start),
                num_local_layers=int(layer_end) - int(layer_start),
            )
            if self._geometry.blocks_axis != 2:
                raise ValueError(
                    "inference KV transfers require the [2, L, B, T, H, d] layout"
                )
            if self._layout.local_num_heads() != heads_per_partition:
                raise ValueError(
                    "heads_per_partition does not match the canonical KV layout: "
                    f"{heads_per_partition} vs {self._layout.local_num_heads()}"
                )
            if self._geometry.num_outer % self._layout.local_num_layers() != 0:
                raise ValueError(
                    f"num_outer={self._geometry.num_outer} is not divisible by local layers="
                    f"{self._layout.local_num_layers()}"
                )

        if self._layout is not None and mamba_layout is not None:
            raise ValueError("a transfer backend cannot have both KV and Mamba layouts")
        self._mamba_layout = mamba_layout
        self._mamba_state_kind = mamba_state_kind

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
            self._geometry.num_blocks,
            self._geometry.blocks_axis,
            self._geometry.num_outer,
            self._geometry.bytes_per_slice,
            self._geometry.num_outer * self._geometry.bytes_per_slice,
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
            "outer_stride_bytes": self._outer_stride_bytes,
            "device_id": self._device_id,
        }
        meta.update(self._geometry.to_meta())
        if self._layout is not None:
            meta.update(self._layout.to_meta())
        if self._mamba_layout is not None:
            meta["mamba_layout"] = asdict(self._mamba_layout)
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
        resolved = peer_id if peer_id else peer_name
        self._known_peers[peer_name] = resolved
        logger.info(
            "NixlTransferBackend[%s] registered peer %s", self.agent_name, peer_name
        )
        return resolved

    def _matched_segments(
        self, peer_meta: Any, src_block_ids: List[int], dst_block_ids: List[int]
    ) -> List[TransferSegment]:
        """Build a whole-buffer copy for state with no resharding policy."""

        records = transfer_peer_records(peer_meta, src_block_ids)
        if len(records) != 1:
            raise ValueError("matched-layout transfer requires exactly one source peer")
        meta, blocks = records[0]
        peer_geometry = KVBufferGeometry.from_meta(meta)
        self._geometry.validate_transfer_from(
            peer_geometry,
            blocks,
            dst_block_ids,
            peer_name=meta.get("agent_name"),
            require_matched_layout=True,
        )
        return [
            TransferSegment(
                peer_meta=meta,
                src_block_ids=blocks,
                src_o_start=0,
                dst_o_start=0,
                n_outer=self._geometry.num_outer,
            )
        ]

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
        self.begin_pull_blocks(peer_meta, src_block_ids, dst_block_ids).wait()

    def begin_pull_blocks(
        self,
        peer_meta: Any,
        src_block_ids: List[int],
        dst_block_ids: List[int],
    ) -> NixlPullHandle:
        """Submit a pull and return a handle that can be polled later."""
        if not isinstance(peer_meta, dict) or "pp_metas" not in peer_meta:
            if not src_block_ids and not dst_block_ids:
                return NixlPullHandle(
                    agent=self._agent,
                    xfers=[],
                    contexts=[],
                    submitted_at=time.perf_counter(),
                    done=True,
                )

        if self._mamba_layout is not None:
            segments = build_mamba_reshard_plan(
                peer_meta,
                src_block_ids,
                dst_block_ids,
                self._mamba_layout,
                self._geometry,
                self._mamba_state_kind,
            )
        elif self._layout is not None:
            segments = build_reshard_plan(
                peer_meta,
                src_block_ids,
                dst_block_ids,
                self._layout,
                self._geometry,
            )
        else:
            segments = self._matched_segments(peer_meta, src_block_ids, dst_block_ids)
        xfers: List[Any] = []
        contexts: List[str] = []
        submitted_at = time.perf_counter()
        try:
            for seg in segments:
                xfer, ctx = self._begin_segment(seg, dst_block_ids)
                xfers.append(xfer)
                contexts.append(ctx)
        except Exception as exc:
            if xfers:
                cleanup = NixlPullHandle(
                    agent=self._agent,
                    xfers=xfers,
                    contexts=contexts,
                    submitted_at=submitted_at,
                )
                try:
                    cleanup.wait()
                except TimeoutError:
                    # Tell the owner not to recycle the destination storage while
                    # an already-submitted transfer may still write to it.
                    setattr(exc, "transfer_destinations_safe", False)
                except Exception:
                    # Transfer errors are reported only after every submitted
                    # segment has reached a terminal state.
                    pass
            raise
        return NixlPullHandle(
            agent=self._agent,
            xfers=xfers,
            contexts=contexts,
            submitted_at=submitted_at,
        )

    def _begin_segment(
        self, seg: TransferSegment, dst_block_ids: List[int]
    ) -> tuple[Any, str]:
        """Submit one full-slice or head-fragment transfer segment."""
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
        bps = self._geometry.bytes_per_slice
        local_os = self._outer_stride_bytes

        src_tuples: List[Any] = []
        dst_tuples: List[Any] = []

        if seg.n_heads == 0:
            # One descriptor per block and outer slice.
            for src_b, dst_b in zip(src_block_ids, dst_block_ids):
                for i in range(n_outer):
                    src_o = src_o_start + i
                    dst_o = dst_o_start + i
                    src_tuples.append(
                        (
                            peer_base + src_o * peer_os + src_b * peer_bps,
                            peer_bps,
                            peer_device_id,
                        )
                    )
                    dst_tuples.append(
                        (
                            self._buf_ptr + dst_o * local_os + dst_b * bps,
                            bps,
                            self._device_id,
                        )
                    )
            ctx = (
                f"matched peer={peer_id} outer[{src_o_start}:+{n_outer}] "
                f"blocks={len(src_block_ids)}"
            )
        else:
            # Head sub-range copy: one descriptor per token.
            assert self._geometry.head_dim is not None
            assert self._geometry.heads_per_partition is not None
            assert self._geometry.tokens_per_block is not None
            d_bytes = self._geometry.head_dim * self._element_size
            local_token_stride = self._geometry.heads_per_partition * d_bytes
            peer_token_stride = pm["heads_per_partition"] * d_bytes
            T = self._geometry.tokens_per_block
            frag_bytes = seg.n_heads * d_bytes
            src_h_off = seg.src_h0 * d_bytes
            dst_h_off = seg.dst_h0 * d_bytes

            for src_b, dst_b in zip(src_block_ids, dst_block_ids):
                for i in range(n_outer):
                    src_o = src_o_start + i
                    dst_o = dst_o_start + i
                    src_slice = (
                        peer_base + src_o * peer_os + src_b * peer_bps + src_h_off
                    )
                    dst_slice = (
                        self._buf_ptr + dst_o * local_os + dst_b * bps + dst_h_off
                    )
                    for t in range(T):
                        src_tuples.append(
                            (
                                src_slice + t * peer_token_stride,
                                frag_bytes,
                                peer_device_id,
                            )
                        )
                        dst_tuples.append(
                            (
                                dst_slice + t * local_token_stride,
                                frag_bytes,
                                self._device_id,
                            )
                        )
            ctx = (
                f"reshard peer={peer_id} outer[{src_o_start}:+{n_outer}] "
                f"heads[{seg.src_h0}:+{seg.n_heads}] blocks={len(src_block_ids)}"
            )

        src_descs = self._agent.get_xfer_descs(src_tuples, mem_type="VRAM")
        dst_descs = self._agent.get_xfer_descs(dst_tuples, mem_type="VRAM")
        # READ pulls remote -> local. Signature is (op, local, remote, peer).
        xfer = self._agent.initialize_xfer("READ", dst_descs, src_descs, peer_id)
        try:
            self._agent.transfer(xfer)
        except Exception as exc:
            # The transport may have accepted the operation before surfacing an
            # error, so its destination cannot be proven safe for immediate reuse.
            setattr(exc, "transfer_destinations_safe", False)
            raise
        return xfer, ctx

    def close(self) -> None:
        if self._agent is None:
            return
        try:
            self._agent.deregister_memory(self._reg_handle)
        except Exception:  # noqa: BLE001 - shutdown path
            logger.exception("NixlTransferBackend: deregister_memory failed")
        self._agent = None


# Backward-compatible name for callers/tests that still import the old agent.
KvTransferAgent = NixlTransferBackend
