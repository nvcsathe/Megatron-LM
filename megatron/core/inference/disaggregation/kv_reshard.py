# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KV-cache reshard planning across heterogeneous parallelism.

This module contains the transport-neutral shard-layout planner and the
NIXL segment adapter used by the current KV transfer backend.

No NIXL, no torch required. Functions here compute *what* to pull (which peer,
which block ids, which outer-slice range, which head range); the transfer backend
executes *how* (NIXL descriptor building and submission).

Layout conventions
------------------
Megatron's paged KV buffer has two layouts:

- K/V split:  ``[2, L, B, T, H, d]`` → blocks-axis = 2, num_outer = 2 × L
- MLA:        ``[L, B, T, H, d]``    → blocks-axis = 1, num_outer = L

where L = attention layers on this PP rank, B = total blocks in the pool,
T = tokens per block, H = KV heads per partition, d = head dim.

Within each (outer, block) slice the layout is ``[T, H, d]`` for K/V split or
``[T, D]`` for MLA. Head re-sharding only applies to the K/V-split case (MLA
concatenates K and Q projections; heads are not independently addressable in
the same way — treat it as matched-only for now).

Parallelism dimensions handled here
-------------------------------------
TP (tensor parallelism)
    Each TP rank owns a contiguous range of KV heads ``[r*Hpp, (r+1)*Hpp)``
    where ``Hpp = num_kv_heads_global / tp_size``. Re-sharding computes which
    peer head-ranges overlap with the local range.

PP (pipeline parallelism)
    Each PP rank owns a contiguous range of attention layers ``[layer_start,
    layer_end)``. Re-sharding selects which outer-index slices (= layer × kv_factor)
    to pull from each prefill PP rank.

EP (expert parallelism) — *future*
    Each EP rank owns a subset of MoE expert slots. A placeholder hook is
    provided; add ``ep_reshard_plan`` here when needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from megatron.core.inference.disaggregation.shard_range_intersection import intersect

# ---------------------------------------------------------------------------
# Topology descriptor
# ---------------------------------------------------------------------------


@dataclass
class KvTopology:
    """This rank's KV buffer topology — the source of truth for plan computations.

    All fields that affect re-sharding are captured here so that plan functions
    are pure (no side-effects, no NIXL, no torch) and therefore trivially
    testable.

    Optional fields are ``None`` when the corresponding parallelism dimension is
    absent (PP=1, TP=1, etc.); plan functions gate their logic on non-None checks.

    Future: add ``ep_rank``, ``expert_start``, ``expert_end`` for EP support.
    """

    # --- TP ---
    tp_size: Optional[int] = None
    tp_rank: Optional[int] = None
    num_kv_heads_global: Optional[int] = None
    heads_per_partition: Optional[int] = None  # num_kv_heads_global // tp_size
    head_dim: Optional[int] = None
    tokens_per_block: Optional[int] = None

    # --- PP ---
    pp_rank: Optional[int] = None
    layer_start: Optional[int] = None  # inclusive global attention-layer index
    layer_end: Optional[int] = None    # exclusive

    # --- Buffer geometry (set by KvTransferAgent from the memory buffer shape) ---
    num_outer: Optional[int] = None  # kv_factor × (layer_end − layer_start)

    # ------------------------------------------------------------------ derived

    @property
    def n_layers(self) -> int:
        if self.layer_start is None or self.layer_end is None:
            raise ValueError("KvTopology.n_layers requires layer_start and layer_end")
        return self.layer_end - self.layer_start

    @property
    def kv_factor(self) -> int:
        """Outer slices per attention layer (2 for K/V split, 1 for MLA)."""
        n = self.n_layers
        if n == 0:
            raise ValueError("KvTopology has zero layers")
        if self.num_outer is None:
            raise ValueError("KvTopology.kv_factor requires num_outer")
        factor, rem = divmod(self.num_outer, n)
        if rem != 0:
            raise ValueError(
                f"num_outer={self.num_outer} is not divisible by n_layers={n}; "
                "unexpected KV buffer layout."
            )
        return factor

    @property
    def tp_capable(self) -> bool:
        """True iff TP topology is fully specified and heads are partitioned (not replicated)."""
        return (
            None not in (self.tp_size, self.tp_rank, self.num_kv_heads_global,
                         self.heads_per_partition, self.head_dim, self.tokens_per_block)
            and self.heads_per_partition * self.tp_size == self.num_kv_heads_global  # type: ignore[operator]
        )

    @property
    def pp_capable(self) -> bool:
        """True iff PP topology is fully specified."""
        return None not in (self.pp_rank, self.layer_start, self.layer_end, self.num_outer)


# ---------------------------------------------------------------------------
# Transfer segment
# ---------------------------------------------------------------------------


@dataclass
class TransferSegment:
    """One atomic NIXL pull: a (outer-range × head-range) fragment from one peer.

    the transfer backend turns this into NIXL
    descriptor lists and submits the transfer.

    Outer range
        ``src_o_start`` / ``dst_o_start`` / ``n_outer`` select which consecutive
        outer slices (= layers × kv_factor) to copy. For PP=1 transfers
        ``src_o_start = dst_o_start = 0`` and ``n_outer = local.num_outer``.

    Head range (TP re-sharding)
        ``n_heads == 0`` triggers the fast matched-layout path: one descriptor
        per (block, outer) pair, copying a full ``bytes_per_slice``. No per-token
        iteration is needed.

        ``n_heads > 0`` triggers the head sub-range path: one descriptor per
        (block, outer, token) triple, copying ``n_heads × head_dim × element_size``
        bytes starting at ``src_h0`` / ``dst_h0`` within the ``[T, H, d]`` slice.
    """

    peer_meta: Dict[str, Any]    # export_meta() from the prefill peer
    src_block_ids: List[int]     # block IDs on that prefill peer for this request
    src_o_start: int             # first outer index to read from the peer buffer
    dst_o_start: int             # first outer index to write into the local buffer
    n_outer: int                 # number of consecutive outer slices to transfer
    # Head range — both zero and n_heads=0 → full-slice matched path
    src_h0: int = 0              # first head in the peer's per-token slice (head units)
    dst_h0: int = 0              # first head in the local per-token slice (head units)
    n_heads: int = 0             # 0 → full slice; >0 → head sub-range


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------


def build_reshard_plan(
    kv_meta: Any,
    src_block_ids: List[int],
    local: KvTopology,
) -> List[TransferSegment]:
    """Produce the full list of NIXL transfer segments for one handoff request.

    Parameters
    ----------
    kv_meta:
        The ``kv_meta`` field from ``disaggregated_params``.

        - ``{"pp_metas": [...]}``  — heterogeneous-PP: each entry is one prefill
          PP rank's TP metas + per-rank block ids.
        - list of dicts            — heterogeneous-TP (multiple prefill TP peers,
          same layer range, same block ids for all).
        - single dict              — matched or single-peer (PP=1, TP=1 or
          TP-matched).

    src_block_ids:
        Block IDs on the prefill side. Used for PP=1 paths; PP>1 paths read
        per-rank block ids from inside each ``pp_metas`` entry.

    local:
        This decode rank's :class:`KvTopology`.

    Returns
    -------
    List of :class:`TransferSegment` ready for
    the transfer backend.
    """
    if isinstance(kv_meta, dict) and "pp_metas" in kv_meta:
        return _pp_to_segments(kv_meta["pp_metas"], local)

    peer_metas = kv_meta if isinstance(kv_meta, list) else [kv_meta]
    return _tp_to_segments(peer_metas, src_block_ids, local)


# ---------------------------------------------------------------------------
# TP re-sharding
# ---------------------------------------------------------------------------


def tp_reshard_plan(
    peer_metas: List[Dict[str, Any]],
    local: KvTopology,
    *,
    check_outer: bool = True,
) -> List[Dict[str, Any]]:
    """Compute head-range fragments for heterogeneous-TP KV re-sharding.

    KV heads live in a global index space ``[0, num_kv_heads_global)`` (==
    ``num_query_groups`` under GQA). Rank ``r`` at TP ``t`` owns
    ``[r*Hpp, (r+1)*Hpp)`` where ``Hpp = num_kv_heads_global / t``. For each
    prefill peer we intersect its head range with ours; the overlap is expressed
    as offsets into the peer's slice (``src_h0``) and our slice (``dst_h0``).

    Parameters
    ----------
    peer_metas:
        List of ``export_meta()`` dicts, one per prefill TP rank.
    local:
        Decode rank's topology (must have TP fields set).
    check_outer:
        When True (default), raise if a peer's ``num_outer`` differs from
        ``local.num_outer`` — indicates a PP mismatch in a TP-only context.
        Set to False when called from :func:`pp_reshard_plan` where num_outer
        legitimately differs across PP ranks.

    Returns
    -------
    List of ``{peer, src_h0, dst_h0, n_heads}`` dicts.
    """
    if not local.tp_capable:
        raise RuntimeError(
            "KvTopology is not TP-capable (tp_size/heads_per_partition/etc. missing "
            "or in head-replication regime). Heterogeneous-TP handoff requires "
            "partitioned (not replicated) KV heads."
        )
    g = local.num_kv_heads_global
    local_hpp = local.heads_per_partition
    local_lo = local.tp_rank * local_hpp  # type: ignore[operator]
    local_hi = local_lo + local_hpp       # type: ignore[operator]

    plan: List[Dict[str, Any]] = []
    for pm in peer_metas:
        for key in ("tp_size", "tp_rank", "heads_per_partition", "head_dim",
                    "tokens_per_block", "num_kv_heads_global"):
            if pm.get(key) is None:
                raise ValueError(
                    f"peer_meta missing topology field {key!r}; prefill engine "
                    "must be built with TP topology for heterogeneous-TP handoff."
                )
        if pm["num_kv_heads_global"] != g:
            raise ValueError(
                f"num_kv_heads_global mismatch peer={pm['num_kv_heads_global']} "
                f"local={g} (different model?)."
            )
        if pm["head_dim"] != local.head_dim or pm["tokens_per_block"] != local.tokens_per_block:
            raise ValueError(
                "head_dim/tokens_per_block mismatch — only TP may differ between "
                "prefill and decode, not head dim or block size."
            )
        if check_outer and pm["num_outer"] != local.num_outer:
            raise ValueError(
                f"num_outer mismatch peer={pm['num_outer']} local={local.num_outer} — "
                "use build_reshard_plan with {\"pp_metas\": ...} for heterogeneous PP."
            )
        p_hpp = pm["heads_per_partition"]
        p_lo = pm["tp_rank"] * p_hpp
        p_hi = p_lo + p_hpp
        lo = max(local_lo, p_lo)
        hi = min(local_hi, p_hi)
        if hi <= lo:
            continue
        plan.append({"peer": pm, "src_h0": lo - p_lo, "dst_h0": lo - local_lo, "n_heads": hi - lo})

    covered = sum(item["n_heads"] for item in plan)
    if covered != local_hpp:
        raise ValueError(
            f"TP re-shard plan covers {covered} of {local_hpp} local KV heads; "
            f"prefill TP set {[pm.get('tp_size') for pm in peer_metas]} is not "
            f"compatible with decode TP {local.tp_size} (one must divide the "
            "other and both must divide num_kv_heads_global)."
        )
    return plan


def _tp_to_segments(
    peer_metas: List[Dict[str, Any]],
    src_block_ids: List[int],
    local: KvTopology,
    *,
    check_outer: bool = True,
    src_o_start: int = 0,
    dst_o_start: int = 0,
    n_outer: Optional[int] = None,
) -> List[TransferSegment]:
    """Convert TP peer metas into TransferSegments.

    ``src_o_start`` / ``dst_o_start`` / ``n_outer`` are filled in by the PP
    layer when called from ``_pp_to_segments``; callers at the TP-only level
    leave them at defaults (0 / local.num_outer).
    """
    effective_n_outer = local.num_outer if n_outer is None else n_outer

    # Single peer with no TP topology block → matched layout, full-slice copy.
    if len(peer_metas) == 1 and not _has_tp_topology(peer_metas[0]):
        return [
            TransferSegment(
                peer_meta=peer_metas[0],
                src_block_ids=src_block_ids,
                src_o_start=src_o_start,
                dst_o_start=dst_o_start,
                n_outer=effective_n_outer,
            )
        ]

    # TP-heterogeneous path.
    fragments = tp_reshard_plan(peer_metas, local, check_outer=check_outer)

    # Equal-TP shortcut: plan resolves to a single peer covering our full head
    # range from offset 0, AND the peer's heads_per_partition equals ours so
    # their bytes_per_slice equals ours. Only then is the full-slice matched
    # copy correct — if the peer has more heads per slice (e.g. prefill TP=1
    # vs decode TP=2), peer_bps > local_bps and NIXL rejects the size mismatch.
    if (
        len(fragments) == 1
        and fragments[0]["src_h0"] == 0
        and fragments[0]["dst_h0"] == 0
        and fragments[0]["n_heads"] == local.heads_per_partition
        and fragments[0]["peer"]["heads_per_partition"] == local.heads_per_partition
    ):
        return [
            TransferSegment(
                peer_meta=fragments[0]["peer"],
                src_block_ids=src_block_ids,
                src_o_start=src_o_start,
                dst_o_start=dst_o_start,
                n_outer=effective_n_outer,
            )
        ]

    return [
        TransferSegment(
            peer_meta=frag["peer"],
            src_block_ids=src_block_ids,
            src_o_start=src_o_start,
            dst_o_start=dst_o_start,
            n_outer=effective_n_outer,
            src_h0=frag["src_h0"],
            dst_h0=frag["dst_h0"],
            n_heads=frag["n_heads"],
        )
        for frag in fragments
    ]


# ---------------------------------------------------------------------------
# PP re-sharding
# ---------------------------------------------------------------------------


def pp_reshard_plan(
    pp_metas: List[Dict[str, Any]],
    local: KvTopology,
) -> List[Dict[str, Any]]:
    """Compute outer-range fragments for heterogeneous-PP KV re-sharding.

    Each element of ``pp_metas`` is a dict ``{"tp_metas": ..., "block_ids": [...]}``
    produced by ``dynamic_engine._capture_handoff_meta`` when prefill PP > 1.
    ``tp_metas`` is a single export_meta() dict (TP=1) or a list of such dicts (TP>1).

    Outer indices map to layers via ``kv_factor`` (2 for K/V split, 1 for MLA).
    For each prefill PP rank the overlapping layer range is computed; the outer-
    index arithmetic converts that to byte-addressable slice indices.

    Parameters
    ----------
    pp_metas:
        List of per-prefill-PP-rank entries, as packed by ``_capture_handoff_meta``.
    local:
        Decode rank's topology (must have PP fields set).

    Returns
    -------
    List of raw plan dicts::

        {
            "tp_metas":   list of export_meta() dicts for this prefill PP rank,
            "block_ids":  src block ids on this prefill PP rank,
            "src_o_start": first outer-index to read from the peer buffer,
            "dst_o_start": first outer-index to write into the local buffer,
            "n_outer":     number of consecutive outer slices,
        }
    """
    if not local.pp_capable:
        raise RuntimeError(
            "KvTopology is not PP-capable (pp_rank/layer_start/layer_end/num_outer "
            "missing). Build the decode agent with pp_rank, layer_start, layer_end."
        )
    local_kv_factor = local.kv_factor
    plan: List[Dict[str, Any]] = []
    total_n_outer = 0

    for entry in pp_metas:
        raw_tp = entry.get("tp_metas", entry)
        tp_metas = raw_tp if isinstance(raw_tp, list) else [raw_tp]
        ref = tp_metas[0]
        p_lo = ref.get("layer_start")
        p_hi = ref.get("layer_end")
        if p_lo is None or p_hi is None:
            raise ValueError(
                "pp_metas entry is missing layer_start/layer_end in its tp_metas; "
                "prefill engine must be built with pp_rank/layer_start/layer_end."
            )
        lo = max(local.layer_start, p_lo)  # type: ignore[type-var]
        hi = min(local.layer_end, p_hi)    # type: ignore[type-var]
        if hi <= lo:
            continue

        p_num_outer = ref["num_outer"]
        p_n_layers = p_hi - p_lo
        if p_n_layers <= 0:
            raise ValueError(f"PP meta has zero-length layer range [{p_lo}, {p_hi})")
        p_kv_factor, rem = divmod(p_num_outer, p_n_layers)
        if rem != 0:
            raise ValueError(
                f"PP meta num_outer={p_num_outer} is not divisible by layer count "
                f"{p_n_layers}; unexpected KV buffer layout."
            )
        if p_kv_factor != local_kv_factor:
            raise ValueError(
                f"kv_factor mismatch peer={p_kv_factor} local={local_kv_factor} "
                "(K/V-split vs MLA layout mismatch between prefill and decode)."
            )
        n_outer = p_kv_factor * (hi - lo)
        plan.append(
            {
                "tp_metas": tp_metas,
                "block_ids": entry.get("block_ids", []),
                "src_o_start": p_kv_factor * (lo - p_lo),
                "dst_o_start": local_kv_factor * (lo - local.layer_start),  # type: ignore[operator]
                "n_outer": n_outer,
            }
        )
        total_n_outer += n_outer

    if total_n_outer != local.num_outer:
        raise ValueError(
            f"PP reshard plan covers {total_n_outer} of {local.num_outer} local outer "
            f"slices (decode layers [{local.layer_start}, {local.layer_end})). Ensure "
            "all prefill PP ranks shipped layer_start/layer_end in their tp_metas."
        )
    return plan


def _pp_to_segments(
    pp_metas: List[Dict[str, Any]],
    local: KvTopology,
) -> List[TransferSegment]:
    """Convert PP plan dicts into TransferSegments (including per-PP TP plans)."""
    raw_plan = pp_reshard_plan(pp_metas, local)
    segments: List[TransferSegment] = []
    for item in raw_plan:
        tp_metas = item["tp_metas"]
        tp_segs = _tp_to_segments(
            tp_metas,
            item["block_ids"],
            local,
            check_outer=False,
            src_o_start=item["src_o_start"],
            dst_o_start=item["dst_o_start"],
            n_outer=item["n_outer"],
        )
        segments.extend(tp_segs)
    return segments


# ---------------------------------------------------------------------------
# EP re-sharding (future)
# ---------------------------------------------------------------------------


def ep_reshard_plan(
    ep_metas: List[Dict[str, Any]],
    local: KvTopology,
) -> List[Dict[str, Any]]:
    """Placeholder for expert-parallelism KV re-sharding.

    EP shards MoE expert slots across ranks; the KV buffer layout for MoE
    models differs from dense models (expert dim replaces or extends the head
    dim). Implement here when needed; wire into ``build_reshard_plan`` analogously
    to PP/TP.
    """
    raise NotImplementedError(
        "EP KV re-sharding is not yet implemented. Add ep_reshard_plan logic here "
        "and extend build_reshard_plan to dispatch on ep_metas."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _has_tp_topology(peer_meta: Dict[str, Any]) -> bool:
    """True iff the peer shipped TP topology fields (tp_size, tp_rank, ...)."""
    return peer_meta.get("tp_size") is not None


# ---------------------------------------------------------------------------
# Transport-neutral rank-layout planner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KVShardLayout:
    """A worker's KV-cache ownership within the global model.

    ``num_layers`` / ``num_heads`` are the *global* attention layer count
    and KV-head count (for GQA, the number of KV heads). ``global_rank``
    is the worker's torch rank (used as the transport peer id).
    """

    num_layers: int
    num_heads: int
    tp_size: int
    tp_rank: int
    pp_size: int
    pp_rank: int
    global_rank: int
    # Expert dimensions. KV-replica dimensions only: they shard the MoE
    # expert weights, never the attention KV cache, so they don't affect
    # head_range/layer_range -- only representative (source) selection.
    ep_size: int = 1
    ep_rank: int = 0
    etp_size: int = 1
    etp_rank: int = 0
    # Optional explicit PP layer window for this stage. When None, an even split
    # of num_layers across pp_size is assumed -- correct for pure-attention
    # models. Models that do NOT split attention layers evenly across PP stages
    # (e.g. hybrid Mamba+attention) must pass an explicit (layer_start,
    # num_local_layers); the even-split default would otherwise map the wrong
    # global layer indices.
    layer_start: Optional[int] = None
    num_local_layers: Optional[int] = None

    def __post_init__(self) -> None:
        # TP must divide heads (the head split is always even).
        if self.num_heads % self.tp_size != 0:
            raise ValueError(f"num_heads={self.num_heads} not divisible by tp_size={self.tp_size}")
        # layer_start and num_local_layers are an all-or-nothing explicit window:
        # setting only one would silently fall back to the even-split count and
        # defeat the purpose (uneven stage with an even count).
        if (self.layer_start is None) != (self.num_local_layers is None):
            raise ValueError(
                "layer_start and num_local_layers must be set together (or both omitted)"
            )
        # Only the even-split path requires PP to divide layers; an explicit
        # window may be uneven across stages.
        if self.layer_start is None and self.num_layers % self.pp_size != 0:
            raise ValueError(
                f"num_layers={self.num_layers} not divisible by pp_size={self.pp_size}; "
                "pass an explicit (layer_start, num_local_layers) for uneven PP splits"
            )

    def kv_shard_key(self) -> Tuple[int, int]:
        """The attention shard this rank holds: ``(tp_rank, pp_rank)``.
        Ranks sharing a key hold identical KV (EP/ETP replicas of it)."""
        return (self.tp_rank, self.pp_rank)

    def layer_range(self) -> Tuple[int, int]:
        """Global attention-layer range ``[lo, hi)`` owned by this rank."""
        # num_local_layers is guaranteed set whenever layer_start is (see __post_init__).
        if self.layer_start is not None:
            return (self.layer_start, self.layer_start + self.num_local_layers)
        per = self.num_layers // self.pp_size
        return (self.pp_rank * per, (self.pp_rank + 1) * per)

    def head_range(self) -> Tuple[int, int]:
        """Global KV-head range ``[lo, hi)`` owned by this rank."""
        per = self.num_heads // self.tp_size
        return (self.tp_rank * per, (self.tp_rank + 1) * per)

    def local_num_layers(self) -> int:
        """Number of attention layers held locally by this rank."""
        lo, hi = self.layer_range()
        return hi - lo

    def local_num_heads(self) -> int:
        """Number of KV heads held locally by this rank."""
        lo, hi = self.head_range()
        return hi - lo


@dataclass(frozen=True)
class KVReshardTransfer:
    """One sub-block exchange between a (src, dst) rank pair.

    Global coords identify the intersection; the local-slice helpers
    convert to each side's buffer offsets. There is at most one transfer
    per (src, dst) pair (each owns a contiguous rectangle, so the
    intersection is a single rectangle).
    """

    src_rank: int
    dst_rank: int
    # The transferred sub-block's GLOBAL bounds as half-open ranges:
    # layers [global_layer_lo, global_layer_hi) x kv-heads [global_head_lo, global_head_hi).
    global_layer_lo: int
    global_layer_hi: int
    global_head_lo: int
    global_head_hi: int

    def src_layer_slice(self, src: KVShardLayout) -> slice:
        """Local layer slice on the source side for this transfer."""
        off = src.layer_range()[0]
        return slice(self.global_layer_lo - off, self.global_layer_hi - off)

    def src_head_slice(self, src: KVShardLayout) -> slice:
        """Local KV-head slice on the source side for this transfer."""
        off = src.head_range()[0]
        return slice(self.global_head_lo - off, self.global_head_hi - off)

    def dst_layer_slice(self, dst: KVShardLayout) -> slice:
        """Local layer slice on the destination side for this transfer."""
        off = dst.layer_range()[0]
        return slice(self.global_layer_lo - off, self.global_layer_hi - off)

    def dst_head_slice(self, dst: KVShardLayout) -> slice:
        """Local KV-head slice on the destination side for this transfer."""
        off = dst.head_range()[0]
        return slice(self.global_head_lo - off, self.global_head_hi - off)


def plan_kv_reshard(
    srcs: List[KVShardLayout], dsts: List[KVShardLayout]
) -> List[KVReshardTransfer]:
    """Full reshard plan: every sub-block that must move src -> dst.

    Both sides compute the same plan from the same layouts and filter to
    their own rank (``transfers_for_src`` / ``transfers_for_dst``).

    KV is replicated across the EP and ETP dimensions, so each attention
    shard ``(tp_rank, pp_rank)`` may be held by several source ranks. We
    source each shard from exactly one of them -- the smallest
    ``global_rank`` -- which avoids duplicate sends and is independent of
    how EP/ETP map onto ranks.
    """
    if srcs and dsts:
        if srcs[0].num_layers != dsts[0].num_layers or srcs[0].num_heads != dsts[0].num_heads:
            raise ValueError("src and dst describe different global models")

    # One representative source rank per attention shard (dedupe EP/ETP
    # replicas that hold identical KV).
    rep_rank: dict = {}
    for s in srcs:
        key = s.kv_shard_key()
        if key not in rep_rank or s.global_rank < rep_rank[key]:
            rep_rank[key] = s.global_rank
    source_ranks = set(rep_rank.values())

    transfers: List[KVReshardTransfer] = []
    for d in dsts:
        dl, dh = d.layer_range(), d.head_range()
        for s in srcs:
            if s.global_rank not in source_ranks:
                continue
            li = intersect(s.layer_range(), dl)
            if li is None:
                continue
            hi = intersect(s.head_range(), dh)
            if hi is None:
                continue
            transfers.append(
                KVReshardTransfer(
                    src_rank=s.global_rank,
                    dst_rank=d.global_rank,
                    global_layer_lo=li[0],
                    global_layer_hi=li[1],
                    global_head_lo=hi[0],
                    global_head_hi=hi[1],
                )
            )
    return transfers
