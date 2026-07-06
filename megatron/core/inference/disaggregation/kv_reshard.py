# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KV-cache shard planning and transfer-segment adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from megatron.core.inference.disaggregation.utils import (
    intersect,
    representative_source_ranks,
    transfer_peer_records,
    transfers_for_dst,
    transfers_for_src,
)

__all__ = [
    "KVReshardTransfer",
    "KVShardLayout",
    "KVBufferGeometry",
    "TransferSegment",
    "build_reshard_plan",
    "plan_kv_reshard",
    "transfers_for_dst",
    "transfers_for_src",
]


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
            raise ValueError(
                f"num_heads={self.num_heads} not divisible by tp_size={self.tp_size}"
            )
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

    def to_meta(self) -> Dict[str, int]:
        """Serialize this layout into transfer metadata."""

        layer_start, layer_end = self.layer_range()
        return {
            "global_rank": self.global_rank,
            "tp_size": self.tp_size,
            "tp_rank": self.tp_rank,
            "pp_size": self.pp_size,
            "pp_rank": self.pp_rank,
            "num_layers_global": self.num_layers,
            "num_kv_heads_global": self.num_heads,
            "layer_start": layer_start,
            "layer_end": layer_end,
        }

    @classmethod
    def from_meta(cls, meta: Dict[str, Any]) -> "KVShardLayout":
        """Deserialize a layout exported by :meth:`to_meta`."""

        missing = [key for key in _LAYOUT_META_KEYS if meta.get(key) is None]
        if missing:
            raise ValueError(f"peer metadata missing KV layout fields: {missing}")
        return cls(
            num_layers=int(meta["num_layers_global"]),
            num_heads=int(meta["num_kv_heads_global"]),
            tp_size=int(meta["tp_size"]),
            tp_rank=int(meta["tp_rank"]),
            pp_size=int(meta["pp_size"]),
            pp_rank=int(meta["pp_rank"]),
            global_rank=int(meta["global_rank"]),
            layer_start=int(meta["layer_start"]),
            num_local_layers=int(meta["layer_end"]) - int(meta["layer_start"]),
        )


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

    def layer_slice(self, layout: KVShardLayout) -> slice:
        """Local layer slice for this transfer in ``layout``."""

        off = layout.layer_range()[0]
        return slice(self.global_layer_lo - off, self.global_layer_hi - off)

    def head_slice(self, layout: KVShardLayout) -> slice:
        """Local KV-head slice for this transfer in ``layout``."""

        off = layout.head_range()[0]
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
        if (
            srcs[0].num_layers != dsts[0].num_layers
            or srcs[0].num_heads != dsts[0].num_heads
        ):
            raise ValueError("src and dst describe different global models")

    # One representative source rank per attention shard (dedupe EP/ETP
    # replicas that hold identical KV).
    source_ranks = representative_source_ranks(
        srcs, lambda layout: layout.kv_shard_key()
    )

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


@dataclass(frozen=True)
class KVBufferGeometry:
    """Physical paged-KV geometry needed to realize a logical reshard plan."""

    num_outer: int
    bytes_per_slice: int
    blocks_axis: int
    num_blocks: int
    heads_per_partition: Optional[int] = None
    head_dim: Optional[int] = None
    tokens_per_block: Optional[int] = None
    element_size: Optional[int] = None

    @classmethod
    def from_meta(cls, meta: Dict[str, Any]) -> "KVBufferGeometry":
        return cls(
            num_outer=int(meta["num_outer"]),
            bytes_per_slice=int(meta["bytes_per_slice"]),
            blocks_axis=int(meta["blocks_axis"]),
            num_blocks=int(meta["num_blocks"]),
            heads_per_partition=meta.get("heads_per_partition"),
            head_dim=meta.get("head_dim"),
            tokens_per_block=meta.get("tokens_per_block"),
            element_size=meta.get("element_size"),
        )

    def to_meta(self) -> Dict[str, Any]:
        """Serialize physical geometry into transfer metadata."""

        return {
            "num_outer": self.num_outer,
            "bytes_per_slice": self.bytes_per_slice,
            "blocks_axis": self.blocks_axis,
            "num_blocks": self.num_blocks,
            "heads_per_partition": self.heads_per_partition,
            "head_dim": self.head_dim,
            "tokens_per_block": self.tokens_per_block,
            "element_size": self.element_size,
        }

    def validate_transfer_from(
        self,
        peer: "KVBufferGeometry",
        src_block_ids: List[int],
        dst_block_ids: List[int],
        *,
        peer_name: Optional[str] = None,
        require_matched_layout: bool = False,
    ) -> None:
        """Validate block mappings and physical transfer compatibility."""

        if len(src_block_ids) != len(dst_block_ids):
            source = f" for peer {peer_name!r}" if peer_name is not None else ""
            raise ValueError(
                f"source/destination block_id length mismatch{source}: "
                f"{len(src_block_ids)} vs {len(dst_block_ids)}"
            )
        peer._validate_block_ids(src_block_ids, "source")
        self._validate_block_ids(dst_block_ids, "destination")

        common_fields = ("head_dim", "tokens_per_block", "element_size")
        matched_fields = (
            "num_outer",
            "bytes_per_slice",
            "blocks_axis",
            "heads_per_partition",
        )
        fields = (
            common_fields + matched_fields if require_matched_layout else common_fields
        )
        mismatches = [
            f"{field}: peer={getattr(peer, field)} local={getattr(self, field)}"
            for field in fields
            if getattr(peer, field) is not None
            and getattr(self, field) is not None
            and getattr(peer, field) != getattr(self, field)
        ]
        if mismatches:
            kind = "matched-layout" if require_matched_layout else "transfer"
            raise ValueError(f"{kind} geometry mismatch: {', '.join(mismatches)}")

    def _validate_block_ids(self, block_ids: List[int], side: str) -> None:
        for block in block_ids:
            if not 0 <= block < self.num_blocks:
                raise ValueError(
                    f"{side} block {block} is outside pool [0, {self.num_blocks})"
                )


@dataclass(frozen=True)
class TransferSegment:
    """One physical block/layer/head segment produced from a logical transfer."""

    peer_meta: Dict[str, Any]
    src_block_ids: List[int]
    src_o_start: int
    dst_o_start: int
    n_outer: int
    src_h0: int = 0
    dst_h0: int = 0
    n_heads: int = 0


@dataclass(frozen=True)
class _PeerRecord:
    meta: Dict[str, Any]
    block_ids: List[int]
    layout: KVShardLayout
    geometry: KVBufferGeometry


_LAYOUT_META_KEYS = (
    "global_rank",
    "tp_size",
    "tp_rank",
    "pp_size",
    "pp_rank",
    "num_layers_global",
    "num_kv_heads_global",
    "layer_start",
    "layer_end",
)


def _validate_peer_layout(
    peer: _PeerRecord,
    local: KVBufferGeometry,
    dst_block_ids: List[int],
) -> None:
    local.validate_transfer_from(
        peer.geometry,
        peer.block_ids,
        dst_block_ids,
        peer_name=peer.meta.get("agent_name"),
    )
    if peer.geometry.heads_per_partition != peer.layout.local_num_heads():
        raise ValueError(
            "peer heads_per_partition does not match its KV layout: "
            f"{peer.geometry.heads_per_partition} vs {peer.layout.local_num_heads()}"
        )
    if peer.geometry.num_outer % peer.layout.local_num_layers():
        raise ValueError(
            f"peer num_outer={peer.geometry.num_outer} is not divisible by "
            f"local layers={peer.layout.local_num_layers()}"
        )


def build_reshard_plan(
    peer_meta: Any,
    src_block_ids: List[int],
    dst_block_ids: List[int],
    local_layout: Optional[KVShardLayout],
    local_geometry: KVBufferGeometry,
) -> List[TransferSegment]:
    """Build physical transfer segments using :func:`plan_kv_reshard` slices."""

    raw_records = transfer_peer_records(peer_meta, src_block_ids)
    if not raw_records:
        raise ValueError("KV handoff contains no source peer metadata")

    have_layout = [
        all(meta.get(key) is not None for key in _LAYOUT_META_KEYS)
        for meta, _ in raw_records
    ]
    if not all(have_layout):
        raise ValueError("KV resharding requires complete KV layout metadata")

    if local_layout is None:
        raise ValueError("local transfer agent is missing KV layout metadata")
    if local_geometry.num_outer % local_layout.local_num_layers():
        raise ValueError(
            f"local num_outer={local_geometry.num_outer} is not divisible by "
            f"local layers={local_layout.local_num_layers()}"
        )

    sources: List[_PeerRecord] = []
    source_by_rank: Dict[int, _PeerRecord] = {}
    for meta, blocks in raw_records:
        source = _PeerRecord(
            meta=meta,
            block_ids=blocks,
            layout=KVShardLayout.from_meta(meta),
            geometry=KVBufferGeometry.from_meta(meta),
        )
        _validate_peer_layout(source, local_geometry, dst_block_ids)
        if source.layout.global_rank in source_by_rank:
            raise ValueError(
                f"duplicate source global_rank={source.layout.global_rank} in KV metadata"
            )
        sources.append(source)
        source_by_rank[source.layout.global_rank] = source

    local_planes = local_geometry.num_outer // local_layout.local_num_layers()
    logical_plan = plan_kv_reshard(
        [source.layout for source in sources], [local_layout]
    )
    segments: List[TransferSegment] = []
    for transfer in logical_plan:
        source = source_by_rank[transfer.src_rank]
        src_layout = source.layout
        src_planes = source.geometry.num_outer // src_layout.local_num_layers()
        if src_planes != local_planes:
            raise ValueError(
                f"outer-plane mismatch peer={src_planes} local={local_planes}"
            )

        src_layers = transfer.layer_slice(src_layout)
        dst_layers = transfer.layer_slice(local_layout)
        src_heads = transfer.head_slice(src_layout)
        dst_heads = transfer.head_slice(local_layout)
        assert src_layers.start is not None and src_layers.stop is not None
        assert dst_layers.start is not None and dst_layers.stop is not None
        assert src_heads.start is not None and src_heads.stop is not None
        assert dst_heads.start is not None and dst_heads.stop is not None
        layer_count = src_layers.stop - src_layers.start
        head_count = src_heads.stop - src_heads.start
        full_heads = (
            src_heads.start == 0
            and src_heads.stop == src_layout.local_num_heads()
            and dst_heads.start == 0
            and dst_heads.stop == local_layout.local_num_heads()
            and source.geometry.bytes_per_slice == local_geometry.bytes_per_slice
        )
        full_layers = (
            src_layers.start == 0
            and src_layers.stop == src_layout.local_num_layers()
            and dst_layers.start == 0
            and dst_layers.stop == local_layout.local_num_layers()
        )
        if not full_heads and (
            local_geometry.blocks_axis != 2 or source.geometry.blocks_axis != 2
        ):
            raise NotImplementedError(
                "heterogeneous TP KV handoff requires the K/V-split "
                "[2, L, B, T, H, d] layout; MLA head slicing is unsupported"
            )
        if full_heads and full_layers:
            segments.append(
                TransferSegment(
                    peer_meta=source.meta,
                    src_block_ids=source.block_ids,
                    src_o_start=0,
                    dst_o_start=0,
                    n_outer=local_geometry.num_outer,
                )
            )
            continue

        # [2, L, B, ...] stores all K layers followed by all V layers, so a
        # partial layer range requires one segment per outer plane.
        for plane in range(local_planes):
            segments.append(
                TransferSegment(
                    peer_meta=source.meta,
                    src_block_ids=source.block_ids,
                    src_o_start=(
                        plane * src_layout.local_num_layers() + src_layers.start
                    ),
                    dst_o_start=(
                        plane * local_layout.local_num_layers() + dst_layers.start
                    ),
                    n_outer=layer_count,
                    src_h0=0 if full_heads else src_heads.start,
                    dst_h0=0 if full_heads else dst_heads.start,
                    n_heads=0 if full_heads else head_count,
                )
            )
    return segments
