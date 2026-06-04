# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIXL-based KV transfer for disaggregated prefill/decode.

The Phase-3 disagg flow:
1. Prefill engine runs one prefill step, populates KV blocks for the request.
2. Engine packages ``(block_ids, kv_meta, first_token)`` in the reply and
   pins the blocks until ``RELEASE_KV`` arrives.
3. Decode engine receives the reply, allocates local blocks, NIXL-pulls KV
   data block-by-block from the prefill peer (matched rank-to-rank), then
   continues generation.

This module exposes one class, :class:`KvTransferAgent`, that wraps the NIXL
Python API (``nixl._api``). The agent is created on each TP/PP rank that has
a slice of ``memory_buffer`` — it registers that buffer once at startup and
exposes:

- :meth:`export_meta` returning the JSON-serializable handle metadata that
  the prefill engine ships in ``disaggregated_params["kv_meta"]``.
- :meth:`pull_blocks` performing a synchronous NIXL read from a peer agent.

Constraints (per the Phase-3 plan):
- TP and PP must match between prefill and decode engines (rank-to-rank pull).
- One memory descriptor per rank covers the entire paged buffer.

NIXL availability is checked lazily — the module imports without nixl, and
:class:`KvTransferAgent` only raises if you try to construct one without it.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

try:
    from nixl._api import nixl_agent  # type: ignore[import-not-found]

    _HAVE_NIXL = True
except ImportError:
    nixl_agent = None  # type: ignore[assignment]
    _HAVE_NIXL = False


def have_nixl() -> bool:
    """Return whether the nixl Python package is importable in this env."""
    return _HAVE_NIXL


class KvTransferAgent:
    """Per-rank NIXL agent owning a registration over the paged KV buffer.

    A single :class:`torch.Tensor` (``DynamicInferenceContext.memory_buffer``)
    is registered as one memory descriptor; per-block transfers reference
    sub-regions of that descriptor by ``(block_idx, num_blocks)``.

    The agent identity is ``<role>-<rank>`` so peers can address it
    deterministically.
    """

    def __init__(
        self,
        agent_name: str,
        listen_host: str,
        listen_port: int,
        memory_buffer: torch.Tensor,
    ):
        if not _HAVE_NIXL:
            raise RuntimeError(
                "KvTransferAgent requires the nixl Python package. Install "
                "the NIXL runtime and `pip install nixl` before launching "
                "disaggregated workers."
            )
        self.agent_name = agent_name
        self.listen_host = listen_host
        self.listen_port = listen_port
        self._memory_buffer = memory_buffer

        # Treat layers as the outermost dim and blocks as the second; per-block
        # transfers slice all-layers-for-one-block. Compute the bytes-per-block
        # stride once so pull_blocks can build descriptors quickly.
        # memory_buffer shape (concrete layouts):
        #   - non-paged: [layers, blocks, block_size_tokens, n_kv_heads*head_dim*2]
        #   - paged with K/V split: [2, layers, blocks, block_size_tokens, n_kv_heads*head_dim]
        # In either case, dim 0 spans layers OR (k/v, layers); the "blocks"
        # axis is `1` (or `2` for split). The decode side requires the layout
        # to match the prefill side, which is true under TP=match/PP=1.
        self._buf_dtype = memory_buffer.dtype
        self._buf_ptr = memory_buffer.data_ptr()
        self._buf_numel = memory_buffer.numel()
        self._buf_bytes = memory_buffer.element_size() * self._buf_numel
        # Per-block element count = product of all dims except the "blocks"
        # axis. Default assumption: blocks axis is dim 1.
        shape = list(memory_buffer.shape)
        self._blocks_axis = 1
        self._num_blocks = shape[self._blocks_axis]
        per_block_elems = self._buf_numel // self._num_blocks
        self._per_block_bytes = memory_buffer.element_size() * per_block_elems

        self._agent = nixl_agent(agent_name)
        # Register the entire buffer as one VRAM memory descriptor. NIXL
        # internally chunks transfers; we only need to ship the base address +
        # total size and the agent name.
        self._reg_handle = self._agent.register_memory(
            [(self._buf_ptr, self._buf_bytes, "VRAM")]
        )
        # Optional listen-mode bring-up: each agent listens on a port so peers
        # can address it without an out-of-band rendezvous. Returns immediately;
        # nixl_agent runs the listener in a background thread.
        self._agent.start_listening(self.listen_host, self.listen_port)

        # Cache of peer agent handles, keyed by peer agent_name.
        self._peer_cache: Dict[str, Any] = {}

        logger.info(
            "KvTransferAgent[%s] registered %d-block buffer (%d bytes/block) "
            "listening on %s:%d",
            agent_name,
            self._num_blocks,
            self._per_block_bytes,
            listen_host,
            listen_port,
        )

    def export_meta(self) -> Dict[str, Any]:
        """Return the handle metadata to ship to a decode peer.

        Dict is JSON-friendly so it can ride inside
        ``disaggregated_params["kv_meta"]``.
        """
        return {
            "agent_name": self.agent_name,
            "host": self.listen_host,
            "port": self.listen_port,
            "base_addr": self._buf_ptr,
            "per_block_bytes": self._per_block_bytes,
            "num_blocks": self._num_blocks,
            "blocks_axis": self._blocks_axis,
        }

    def _ensure_peer_handle(self, peer_meta: Dict[str, Any]) -> Any:
        """Look up (or establish) a connection to the peer agent."""
        name = peer_meta["agent_name"]
        handle = self._peer_cache.get(name)
        if handle is None:
            handle = self._agent.connect_to_peer(
                name, peer_meta["host"], peer_meta["port"]
            )
            self._peer_cache[name] = handle
        return handle

    def pull_blocks(
        self,
        peer_meta: Dict[str, Any],
        src_block_ids: List[int],
        dst_block_ids: List[int],
    ) -> None:
        """Synchronously pull blocks from a peer agent into local blocks.

        Args:
            peer_meta: The dict returned by the prefill peer's
                :meth:`export_meta`.
            src_block_ids: Block ids on the peer (1:1 with dst_block_ids).
            dst_block_ids: Block ids on this rank to write into.

        ``src_block_ids`` and ``dst_block_ids`` must have the same length.
        Both endpoints must agree on the same per-block layout (this is
        enforced by the TP-match constraint).
        """
        if len(src_block_ids) != len(dst_block_ids):
            raise ValueError(
                f"src/dst block_id length mismatch: {len(src_block_ids)} vs {len(dst_block_ids)}"
            )
        if not src_block_ids:
            return

        peer_per_block = peer_meta["per_block_bytes"]
        if peer_per_block != self._per_block_bytes:
            raise ValueError(
                "Peer per_block_bytes "
                f"{peer_per_block} != local {self._per_block_bytes}; "
                "TP/PP/dtype mismatch between prefill and decode engines?"
            )
        peer_base = peer_meta["base_addr"]

        # Build descriptor lists: (peer_addr, local_addr, length) for each
        # block. Layout assumption: each block occupies one contiguous span
        # of `per_block_bytes` inside the registered buffer. The paged
        # tensor's blocks axis is contiguous in memory because the outermost
        # dim (layers OR (k/v, layers)) is the slowest-varying, and the
        # block stride equals per_block_bytes.
        local_base = self._buf_ptr
        per = self._per_block_bytes
        src_descs = [(peer_base + sb * per, per, "VRAM") for sb in src_block_ids]
        dst_descs = [(local_base + db * per, per, "VRAM") for db in dst_block_ids]

        peer_handle = self._ensure_peer_handle(peer_meta)
        transfer = self._agent.initialize_xfer(
            "READ", src_descs, dst_descs, peer_handle
        )
        self._agent.post_xfer(transfer)
        self._agent.wait_xfer(transfer)
        self._agent.release_xfer(transfer)

    def close(self) -> None:
        if self._agent is None:
            return
        try:
            self._agent.deregister_memory(self._reg_handle)
        except Exception:  # noqa: BLE001 - shutdown path
            logger.exception("KvTransferAgent: deregister_memory failed")
        try:
            self._agent.stop_listening()
        except Exception:  # noqa: BLE001 - shutdown path
            logger.exception("KvTransferAgent: stop_listening failed")
        self._agent = None


def make_agent(
    role: str,
    rank: int,
    listen_addr: Optional[str],
    memory_buffer: torch.Tensor,
) -> Optional[KvTransferAgent]:
    """Helper: parse ``host:port`` and construct an agent, or return None.

    The launcher passes a `host:port` pair when KV transfer is enabled;
    when it's not, we return None and the engine treats KV transfer as a
    no-op (Phase 0/aggregated path).
    """
    if not listen_addr:
        return None
    host, _, port_str = listen_addr.rpartition(":")
    if not host or not port_str:
        raise ValueError(
            f"--kv-transfer-listen-addr must be host:port, got {listen_addr!r}"
        )
    port = int(port_str)
    agent_name = f"{role}-rank{rank}"
    return KvTransferAgent(agent_name, host, port, memory_buffer)
