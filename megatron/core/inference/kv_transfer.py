# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIXL-based KV transfer for disaggregated prefill/decode.

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
import time
from typing import Any, Dict, List, Optional

import torch

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


class KvTransferAgent:
    """Per-rank NIXL agent owning a registration over the paged KV buffer.

    The whole ``memory_buffer`` is registered once at startup. Per-block
    transfers use ``get_xfer_descs`` to describe sub-ranges by ``(ptr, size,
    device_id)`` triples; NIXL handles the actual RDMA/NVLink path.

    Peer discovery is out-of-band: the prefill engine returns this rank's
    agent metadata inside ``disaggregated_params``, the decode engine calls
    :meth:`pull_blocks` which lazily registers the peer via
    ``add_remote_agent`` on first use.
    """

    def __init__(
        self,
        agent_name: str,
        memory_buffer: torch.Tensor,
        expected_num_blocks: int,
    ):
        if not _HAVE_NIXL:
            raise RuntimeError(
                "KvTransferAgent requires the nixl Python package. Install the "
                "NIXL runtime and `pip install nixl` before launching "
                "disaggregated workers."
            )
        self.agent_name = agent_name
        self._memory_buffer = memory_buffer

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
                f"KvTransferAgent: no axis in memory_buffer shape {shape} "
                f"matches expected_num_blocks={expected_num_blocks}. Layout "
                "is unrecognized — bug in caller or new Megatron tensor shape."
            )
        if len(candidates) > 1:
            raise RuntimeError(
                f"KvTransferAgent: ambiguous blocks axis in shape {shape} "
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
            "KvTransferAgent[%s] registered %d-block buffer "
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
        return {
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
            "KvTransferAgent[%s] registered peer %s", self.agent_name, peer_name
        )
        return resolved

    def pull_blocks(
        self,
        peer_meta: Dict[str, Any],
        src_block_ids: List[int],
        dst_block_ids: List[int],
    ) -> None:
        """Synchronously pull blocks from a peer agent into local blocks.

        Args:
            peer_meta: ``export_meta()`` dict from the prefill peer.
            src_block_ids: Block ids on the peer (paired 1:1 with dst_block_ids).
            dst_block_ids: Local block ids to write into.
        """
        if len(src_block_ids) != len(dst_block_ids):
            raise ValueError(
                f"src/dst block_id length mismatch: "
                f"{len(src_block_ids)} vs {len(dst_block_ids)}"
            )
        if not src_block_ids:
            return

        # Layout compatibility: bytes_per_slice, num_outer, and outer_stride
        # must all agree. Per-block bytes alone isn't enough — same total can
        # come from different scatter-gather layouts.
        for key, local in (
            ("bytes_per_slice", self._bytes_per_slice),
            ("num_outer", self._num_outer),
            ("outer_stride_bytes", self._outer_stride_bytes),
        ):
            peer_val = peer_meta.get(key)
            if peer_val != local:
                raise ValueError(
                    f"Peer/local layout mismatch on {key}: "
                    f"peer={peer_val} local={local}. TP/PP/dtype mismatch?"
                )
        peer_base = peer_meta["base_addr"]
        peer_device_id = peer_meta.get("device_id", 0)
        peer_id = self._ensure_peer_registered(peer_meta)

        # Each logical block contributes `num_outer` non-contiguous slices.
        # Generate one descriptor per (block, outer-index) pair on each side.
        bps = self._bytes_per_slice
        os_stride = self._outer_stride_bytes
        src_tuples = []
        dst_tuples = []
        for src_b, dst_b in zip(src_block_ids, dst_block_ids):
            src_block_offset = src_b * bps
            dst_block_offset = dst_b * bps
            for o in range(self._num_outer):
                src_tuples.append(
                    (peer_base + o * os_stride + src_block_offset, bps, peer_device_id)
                )
                dst_tuples.append(
                    (self._buf_ptr + o * os_stride + dst_block_offset, bps, self._device_id)
                )

        src_descs = self._agent.get_xfer_descs(src_tuples, mem_type="VRAM")
        dst_descs = self._agent.get_xfer_descs(dst_tuples, mem_type="VRAM")

        # READ pulls remote → local: src_descs = where on the peer, dst_descs
        # = where to put it locally. The third positional after the operation
        # in NIXL's API is the *remote* descs; the second is *local*. For
        # "READ" the local side is the destination.
        xfer = self._agent.initialize_xfer("READ", dst_descs, src_descs, peer_id)
        self._agent.transfer(xfer)

        deadline = time.perf_counter() + _POLL_TIMEOUT_S
        while True:
            state = self._agent.check_xfer_state(xfer)
            if state == "DONE":
                break
            if state == "ERR":
                raise RuntimeError(
                    f"NIXL transfer failed (peer={peer_id}, "
                    f"blocks={len(src_block_ids)})"
                )
            if time.perf_counter() > deadline:
                raise TimeoutError(
                    f"NIXL transfer timed out after {_POLL_TIMEOUT_S}s "
                    f"(peer={peer_id}, blocks={len(src_block_ids)}, "
                    f"last_state={state})"
                )
            time.sleep(_POLL_INTERVAL_S)

    def close(self) -> None:
        if self._agent is None:
            return
        try:
            self._agent.deregister_memory(self._reg_handle)
        except Exception:  # noqa: BLE001 - shutdown path
            logger.exception("KvTransferAgent: deregister_memory failed")
        self._agent = None


def make_agent(
    role: str,
    rank: int,
    listen_addr: Optional[str],  # accepted but unused — NIXL does its own discovery
    memory_buffer: torch.Tensor,
    expected_num_blocks: int,
) -> Optional[KvTransferAgent]:
    """Construct an agent, or return None when KV transfer is disabled.

    ``listen_addr`` is kept in the signature for launcher compatibility but is
    ignored by NIXL — peer discovery is metadata-based (see module docstring).
    Callers pass it as a flag to indicate "transfer enabled" by setting any
    non-empty string.
    """
    if not listen_addr:
        return None
    agent_name = f"{role}-rank{rank}"
    return KvTransferAgent(agent_name, memory_buffer, expected_num_blocks)
