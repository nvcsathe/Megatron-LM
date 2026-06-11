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
import os
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
        tp_size: Optional[int] = None,
        tp_rank: Optional[int] = None,
        num_kv_heads_global: Optional[int] = None,
        heads_per_partition: Optional[int] = None,
        head_dim: Optional[int] = None,
        tokens_per_block: Optional[int] = None,
    ):
        if not _HAVE_NIXL:
            raise RuntimeError(
                "KvTransferAgent requires the nixl Python package. Install the "
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
            "KvTransferAgent[%s] registered peer %s", self.agent_name, peer_name
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
                decode run at different TP. Block ids are identical across all
                prefill TP ranks (lockstep allocator), so ``src_block_ids`` /
                ``dst_block_ids`` apply to every peer.
            src_block_ids: Block ids on the peer(s) (paired 1:1 with dst_block_ids).
            dst_block_ids: Local block ids to write into.
        """
        if len(src_block_ids) != len(dst_block_ids):
            raise ValueError(
                f"src/dst block_id length mismatch: "
                f"{len(src_block_ids)} vs {len(dst_block_ids)}"
            )
        if not src_block_ids:
            return

        peer_metas = peer_meta if isinstance(peer_meta, list) else [peer_meta]

        # Fast path: a single peer whose scatter-gather layout matches ours
        # exactly (equal TP/PP/dtype). This is the original, head-agnostic
        # whole-slice copy — cheapest possible, one descriptor per (block,
        # outer-index).
        if len(peer_metas) == 1 and self._layout_matches(peer_metas[0]):
            self._pull_matched(peer_metas[0], src_block_ids, dst_block_ids)
            return

        # Heterogeneous-TP path: re-shard KV heads on the way in.
        self._pull_resharded(peer_metas, src_block_ids, dst_block_ids)

    def _layout_matches(self, peer_meta: Dict[str, Any]) -> bool:
        """True iff peer's scatter-gather layout is byte-identical to ours."""
        return all(
            peer_meta.get(key) == local
            for key, local in (
                ("bytes_per_slice", self._bytes_per_slice),
                ("num_outer", self._num_outer),
                ("outer_stride_bytes", self._outer_stride_bytes),
            )
        )

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

    def _pull_matched(
        self,
        peer_meta: Dict[str, Any],
        src_block_ids: List[int],
        dst_block_ids: List[int],
    ) -> None:
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
        # = where to put it locally. NIXL's signature is (op, LOCAL, REMOTE,
        # peer); for "READ" the local side is the destination.
        xfer = self._agent.initialize_xfer("READ", dst_descs, src_descs, peer_id)
        self._await_xfer(xfer, f"peer={peer_id}, blocks={len(src_block_ids)}")

    def reshard_plan(self, peer_metas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compute which (peer, head-range) fragments this rank must pull.

        KV heads live in a global index space ``[0, num_kv_heads_global)`` (==
        ``num_query_groups`` under GQA). A rank at TP ``t``/``rank r`` owns the
        contiguous range ``[r*Hpp, (r+1)*Hpp)`` where ``Hpp =
        num_kv_heads_global / t``. For each prefill peer we intersect its owned
        range with ours; the overlap is the slice of heads to copy, expressed as
        offsets into the peer's slice (``src_h0``) and our slice (``dst_h0``).

        Returns a list of ``{peer, src_h0, dst_h0, n_heads}``. Raises if the
        union of fragments does not exactly cover our local head range — that
        means an incompatible (non-divisible) TP combination.
        """
        if not self._reshard_capable:
            raise RuntimeError(
                "KvTransferAgent is not reshard-capable (no TP topology). "
                "Heterogeneous-TP handoff requires the decode agent to be built "
                "with tp_size/num_kv_heads_global/etc."
            )
        g = self._num_kv_heads_global
        local_hpp = self._heads_per_partition
        # Replication regime (TP > num_kv_heads_global) repeats whole heads
        # across ranks rather than partitioning them; the simple range-overlap
        # model below doesn't describe it. Only matched TP (fast path) is
        # supported there for now.
        if local_hpp * self._tp_size != g:
            raise NotImplementedError(
                f"KV-head replication regime not supported for re-shard "
                f"(local heads_per_partition={local_hpp} * tp={self._tp_size} "
                f"!= num_kv_heads_global={g}). Use matched TP, or keep TP "
                f"<= num_query_groups on both sides."
            )
        local_lo = self._tp_rank * local_hpp
        local_hi = local_lo + local_hpp

        plan: List[Dict[str, Any]] = []
        for pm in peer_metas:
            for key in ("tp_size", "tp_rank", "heads_per_partition", "head_dim",
                        "tokens_per_block", "num_kv_heads_global"):
                if pm.get(key) is None:
                    raise ValueError(
                        f"peer_meta missing topology field {key!r}; prefill "
                        "engine must be reshard-capable for heterogeneous TP."
                    )
            if pm["num_kv_heads_global"] != g:
                raise ValueError(
                    f"num_kv_heads_global mismatch peer={pm['num_kv_heads_global']} "
                    f"local={g} (different model?)."
                )
            if pm["head_dim"] != self._head_dim or pm["tokens_per_block"] != self._tokens_per_block:
                raise ValueError(
                    "head_dim/tokens_per_block mismatch — only TP may differ "
                    "between prefill and decode, not head dim or block size."
                )
            if pm["num_outer"] != self._num_outer:
                raise ValueError(
                    f"num_outer mismatch peer={pm['num_outer']} local={self._num_outer} "
                    "— PP/layer-count must match (PP-heterogeneous unsupported)."
                )
            p_hpp = pm["heads_per_partition"]
            p_lo = pm["tp_rank"] * p_hpp
            p_hi = p_lo + p_hpp
            lo = max(local_lo, p_lo)
            hi = min(local_hi, p_hi)
            if hi <= lo:
                continue
            plan.append(
                {
                    "peer": pm,
                    "src_h0": lo - p_lo,
                    "dst_h0": lo - local_lo,
                    "n_heads": hi - lo,
                }
            )

        covered = sum(item["n_heads"] for item in plan)
        if covered != local_hpp:
            raise ValueError(
                f"re-shard plan covers {covered} of {local_hpp} local KV heads; "
                f"prefill TP set {[pm.get('tp_size') for pm in peer_metas]} is not "
                f"compatible with decode TP {self._tp_size} (one must divide the "
                "other and both must divide num_kv_heads_global)."
            )
        return plan

    def _pull_resharded(
        self,
        peer_metas: List[Dict[str, Any]],
        src_block_ids: List[int],
        dst_block_ids: List[int],
    ) -> None:
        plan = self.reshard_plan(peer_metas)

        # Equal-TP shortcut: the gathered list is shipped even when TP matches,
        # but the plan then resolves to a single peer covering our full head
        # range with identical layout. Take the cheap whole-slice copy instead
        # of per-token fragments.
        if (
            len(plan) == 1
            and plan[0]["n_heads"] == self._heads_per_partition
            and plan[0]["src_h0"] == 0
            and plan[0]["dst_h0"] == 0
            and self._layout_matches(plan[0]["peer"])
        ):
            self._pull_matched(plan[0]["peer"], src_block_ids, dst_block_ids)
            return

        d_bytes = self._head_dim * self._element_size
        local_token_stride = self._heads_per_partition * d_bytes
        T = self._tokens_per_block
        num_outer = self._num_outer

        # One NIXL transfer per contributing peer (each is a distinct remote
        # agent). Within a peer, a head sub-range is strided across the T tokens
        # of every (outer, block) slice — layout is [T, H, d] — so we emit one
        # descriptor per (block, outer, token).
        for item in plan:
            pm = item["peer"]
            peer_base = pm["base_addr"]
            peer_device_id = pm.get("device_id", 0)
            peer_bps = pm["bytes_per_slice"]
            peer_os = pm["outer_stride_bytes"]
            peer_token_stride = pm["heads_per_partition"] * d_bytes
            peer_id = self._ensure_peer_registered(pm)

            frag_bytes = item["n_heads"] * d_bytes
            src_h_off = item["src_h0"] * d_bytes
            dst_h_off = item["dst_h0"] * d_bytes

            src_tuples = []
            dst_tuples = []
            for src_b, dst_b in zip(src_block_ids, dst_block_ids):
                src_block_off = src_b * peer_bps
                dst_block_off = dst_b * self._bytes_per_slice
                for o in range(num_outer):
                    src_slice = peer_base + o * peer_os + src_block_off + src_h_off
                    dst_slice = self._buf_ptr + o * self._outer_stride_bytes + dst_block_off + dst_h_off
                    for t in range(T):
                        src_tuples.append(
                            (src_slice + t * peer_token_stride, frag_bytes, peer_device_id)
                        )
                        dst_tuples.append(
                            (dst_slice + t * local_token_stride, frag_bytes, self._device_id)
                        )

            src_descs = self._agent.get_xfer_descs(src_tuples, mem_type="VRAM")
            dst_descs = self._agent.get_xfer_descs(dst_tuples, mem_type="VRAM")
            xfer = self._agent.initialize_xfer("READ", dst_descs, src_descs, peer_id)
            self._await_xfer(
                xfer,
                f"reshard peer={peer_id} heads[{item['src_h0']}:+{item['n_heads']}] "
                f"blocks={len(src_block_ids)}",
            )

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
    tp_size: Optional[int] = None,
    tp_rank: Optional[int] = None,
    num_kv_heads_global: Optional[int] = None,
    heads_per_partition: Optional[int] = None,
    head_dim: Optional[int] = None,
    tokens_per_block: Optional[int] = None,
) -> Optional[KvTransferAgent]:
    """Construct an agent, or return None when KV transfer is disabled.

    ``listen_addr`` is kept in the signature for launcher compatibility but is
    ignored by NIXL — peer discovery is metadata-based (see module docstring).
    Callers pass it as a flag to indicate "transfer enabled" by setting any
    non-empty string.

    The ``tp_*`` / head / block args enable heterogeneous-TP re-sharding; pass
    them from the engine's parallel state + KV context. When omitted the agent
    only supports matched layouts.
    """
    if not listen_addr:
        return None
    agent_name = f"{role}-rank{rank}"
    return KvTransferAgent(
        agent_name,
        memory_buffer,
        expected_num_blocks,
        tp_size=tp_size,
        tp_rank=tp_rank,
        num_kv_heads_global=num_kv_heads_global,
        heads_per_partition=heads_per_partition,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
    )
