# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Heterogeneous TP/PP reshard of Mamba conv/ssm state between prefill and
decode shard layouts (the Mamba analog of the attention KV reshard)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from megatron.core.inference.disaggregation.kv_reshard import (
    KVBufferGeometry,
    TransferSegment,
)
from megatron.core.inference.disaggregation.utils import (
    intersect,
    representative_source_ranks,
    transfer_peer_records,
)

# Channel bands of a Mamba layer's state, in the order the conv state
# concatenates them on its channel axis (x, B, C); ssm is the head axis.
# (name, lives_in_conv). conv bands share one tensor; ssm is its own tensor.
_CONV_BANDS = ("x", "B", "C")


@dataclass(frozen=True)
class MambaStateDims:
    """The model's (global, unsharded) Mamba structural dims.

    These belong to the MambaMixer / model config -- carried as one unit (rather
    than loose constants spread across the layout) so there's a single source
    and they can't drift apart. The producer should read them straight from the
    model config (e.g. ``ngroups = config.mamba_num_groups``) rather than
    reverse-deriving from tensor shapes. TP shards ``nheads``/``ngroups``; the
    rest are unsharded.
    """

    nheads: int
    headdim: int
    d_state: int
    ngroups: int
    d_conv: int


@dataclass(frozen=True)
class MambaShardLayout:
    """One rank's Mamba-state ownership: which global layers + TP rank, plus the
    model's structural dims (:class:`MambaStateDims`). Per-rank locals follow by
    dividing by ``tp_size``."""

    global_rank: int
    tp_size: int
    tp_rank: int
    layer_start: int  # global Mamba-layer index of this rank's first layer
    num_layers: int  # Mamba layers held locally (this PP stage)
    dims: MambaStateDims

    def __post_init__(self) -> None:
        # Wire reconstruction (MambaShardLayout(**dict)) hands ``dims`` as a
        # plain dict; coerce it back to MambaStateDims.
        if isinstance(self.dims, dict):
            object.__setattr__(self, "dims", MambaStateDims(**self.dims))
        # TP shards heads and groups; both must divide evenly or the local
        # conv/ssm band sizes truncate to the wrong (or zero) width silently.
        if self.dims.nheads % self.tp_size != 0:
            raise ValueError(
                f"nheads={self.dims.nheads} not divisible by tp_size={self.tp_size}"
            )
        if self.dims.ngroups % self.tp_size != 0:
            raise ValueError(
                f"ngroups={self.dims.ngroups} not divisible by tp_size={self.tp_size}"
            )

    # Convenience proxies onto the dims so callers read ``layout.headdim`` etc.
    @property
    def nheads(self) -> int:
        """Global (unsharded) number of Mamba heads."""
        return self.dims.nheads

    @property
    def headdim(self) -> int:
        """Dimension of each Mamba head."""
        return self.dims.headdim

    @property
    def d_state(self) -> int:
        """SSM state size per head."""
        return self.dims.d_state

    @property
    def ngroups(self) -> int:
        """Global (unsharded) number of B/C groups."""
        return self.dims.ngroups

    @property
    def d_conv(self) -> int:
        """Convolution kernel width."""
        return self.dims.d_conv

    def mamba_shard_key(self) -> Tuple[int, int, int]:
        """The Mamba shard this rank holds. Replica ranks share this key."""

        return (self.tp_rank, self.layer_start, self.num_layers)

    @property
    def d_inner(self) -> int:
        """Global inner dimension (nheads * headdim)."""
        return self.dims.nheads * self.dims.headdim

    @property
    def nheads_local(self) -> int:
        """Number of Mamba heads held by this TP rank."""
        return self.dims.nheads // self.tp_size

    @property
    def d_inner_local(self) -> int:
        """Local inner dimension for this TP rank."""
        return self.d_inner // self.tp_size

    @property
    def ngroups_local(self) -> int:
        """Number of B/C groups held by this TP rank."""
        return self.dims.ngroups // self.tp_size

    @property
    def conv_dim_local(self) -> int:
        """Total local conv channel width (x + B + C bands)."""
        return self.d_inner_local + 2 * self.ngroups_local * self.dims.d_state

    def layer_range(self) -> Tuple[int, int]:
        """Global Mamba-layer range ``[lo, hi)`` owned by this rank."""
        return (self.layer_start, self.layer_start + self.num_layers)

    def _band(self, name: str) -> Tuple[int, int, int]:
        """``(global_total, local_size, conv_local_offset)`` for a band.

        ``conv_local_offset`` is the band's start on the local conv channel
        axis; for the ``ssm`` (head) band it is the start on the local head
        axis (always 0, heads are the whole tensor)."""
        if name == "x":
            g = self.d_inner
            return g, self.d_inner_local, 0
        if name == "B":
            g = self.dims.ngroups * self.dims.d_state
            return g, self.ngroups_local * self.dims.d_state, self.d_inner_local
        if name == "C":
            g = self.dims.ngroups * self.dims.d_state
            return (
                g,
                self.ngroups_local * self.dims.d_state,
                self.d_inner_local + self.ngroups_local * self.dims.d_state,
            )
        if name == "ssm":
            return self.dims.nheads, self.nheads_local, 0
        raise KeyError(name)


@dataclass(frozen=True)
class MambaReshardTransfer:
    """One sub-block move for the reshard.

    ``band`` is ``"x"``/``"B"``/``"C"`` (conv channel axis) or ``"ssm"`` (head
    axis). ``src_layer``/``dst_layer`` are local layer indices on each side;
    ``*_lo``/``*_hi`` are the local channel/head slice bounds.
    """

    src_rank: int
    dst_rank: int
    band: str
    global_layer: int
    src_layer: int
    dst_layer: int
    src_lo: int
    src_hi: int
    dst_lo: int
    dst_hi: int

    @property
    def is_conv(self) -> bool:
        """True if this transfer targets the conv state; False for ssm."""
        return self.band in _CONV_BANDS


def plan_mamba_reshard(
    src_layouts: List[MambaShardLayout], dst_layouts: List[MambaShardLayout]
) -> List[MambaReshardTransfer]:
    """Plan the conv/ssm sub-block moves from the prefill (src) layouts to the
    decode (dst) layouts. One transfer per (src rank, dst rank, global layer,
    band) where both the layer ranges and the channel ranges overlap."""
    if src_layouts and dst_layouts:
        expected_dims = dst_layouts[0].dims
        if any(layout.dims != expected_dims for layout in src_layouts + dst_layouts):
            raise ValueError("source and destination describe different Mamba models")

    # Dedupe replica sources: ranks sharing TP ownership and the same layer
    # window hold identical Mamba state (e.g. EP/DP replicas), so source each
    # shard from exactly one of them.
    source_ranks = representative_source_ranks(
        src_layouts, lambda layout: layout.mamba_shard_key()
    )

    out: List[MambaReshardTransfer] = []
    for s in src_layouts:
        if s.global_rank not in source_ranks:
            continue
        s_lr = s.layer_range()
        for d in dst_layouts:
            layer_ov = intersect(s_lr, d.layer_range())
            if layer_ov is None:
                continue
            for band in (*_CONV_BANDS, "ssm"):
                _, s_size, s_off = s._band(band)
                _, d_size, d_off = d._band(band)
                s_glo = (s.tp_rank * s_size, s.tp_rank * s_size + s_size)
                d_glo = (d.tp_rank * d_size, d.tp_rank * d_size + d_size)
                chan_ov = intersect(s_glo, d_glo)
                if chan_ov is None:
                    continue
                lo, hi = chan_ov
                for g in range(layer_ov[0], layer_ov[1]):
                    out.append(
                        MambaReshardTransfer(
                            src_rank=s.global_rank,
                            dst_rank=d.global_rank,
                            band=band,
                            global_layer=g,
                            src_layer=g - s.layer_start,
                            dst_layer=g - d.layer_start,
                            src_lo=s_off + (lo - s_glo[0]),
                            src_hi=s_off + (hi - s_glo[0]),
                            dst_lo=d_off + (lo - d_glo[0]),
                            dst_hi=d_off + (hi - d_glo[0]),
                        )
                    )
    return out


def build_mamba_reshard_plan(
    peer_meta: Any,
    src_block_ids: List[int],
    dst_block_ids: List[int],
    local_layout: Optional[MambaShardLayout],
    local_geometry: KVBufferGeometry,
    state_kind: str,
) -> List[TransferSegment]:
    """Adapt logical Mamba TP/PP slices into transport segments."""

    if state_kind not in ("conv", "ssm"):
        raise ValueError("state_kind must be 'conv' or 'ssm'")
    if local_layout is None:
        raise ValueError("local transfer agent is missing Mamba layout metadata")
    width = (
        local_layout.conv_dim_local
        if state_kind == "conv"
        else local_layout.nheads_local
    )
    if (
        local_geometry.heads_per_partition != width
        or local_geometry.num_outer != local_layout.num_layers
        or local_geometry.blocks_axis != 1
    ):
        raise ValueError(f"local {state_kind} geometry does not match its Mamba layout")

    records = transfer_peer_records(peer_meta, src_block_ids)
    if not records:
        raise ValueError("Mamba handoff contains no source peer metadata")

    sources = []
    by_rank = {}
    for meta, blocks in records:
        raw_layout = meta.get("mamba_layout")
        if not isinstance(raw_layout, dict):
            raise ValueError("peer metadata is missing mamba_layout")
        layout = MambaShardLayout(**raw_layout)
        geometry = KVBufferGeometry.from_meta(meta)
        peer_width = (
            layout.conv_dim_local if state_kind == "conv" else layout.nheads_local
        )
        local_geometry.validate_transfer_from(
            geometry, blocks, dst_block_ids, peer_name=meta.get("agent_name")
        )
        if (
            geometry.heads_per_partition != peer_width
            or geometry.num_outer != layout.num_layers
            or geometry.blocks_axis != 1
        ):
            raise ValueError(
                f"peer {state_kind} geometry does not match its Mamba layout"
            )
        if layout.global_rank in by_rank:
            raise ValueError(
                f"duplicate source global_rank={layout.global_rank} in Mamba metadata"
            )
        sources.append(layout)
        by_rank[layout.global_rank] = (meta, blocks, geometry)

    for layout in sources:
        if (
            layout.tp_size == local_layout.tp_size
            and layout.tp_rank == local_layout.tp_rank
            and layout.layer_range() == local_layout.layer_range()
            and layout.dims == local_layout.dims
        ):
            meta, blocks, geometry = by_rank[layout.global_rank]
            local_geometry.validate_transfer_from(
                geometry,
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
                    n_outer=local_layout.num_layers,
                )
            ]

    logical_plan = plan_mamba_reshard(sources, [local_layout])
    relevant = [
        transfer
        for transfer in logical_plan
        if transfer.is_conv == (state_kind == "conv")
    ]
    for layer in range(local_layout.num_layers):
        intervals = sorted(
            (transfer.dst_lo, transfer.dst_hi)
            for transfer in relevant
            if transfer.dst_layer == layer
        )
        if not intervals or intervals[0][0] != 0 or intervals[-1][1] != width:
            raise ValueError(
                f"incomplete Mamba {state_kind} coverage for layer {layer}"
            )
        if any(a[1] != b[0] for a, b in zip(intervals, intervals[1:])):
            raise ValueError(
                f"non-contiguous Mamba {state_kind} coverage for layer {layer}"
            )

    return [
        TransferSegment(
            peer_meta=by_rank[transfer.src_rank][0],
            src_block_ids=by_rank[transfer.src_rank][1],
            src_o_start=transfer.src_layer,
            dst_o_start=transfer.dst_layer,
            n_outer=1,
            src_h0=transfer.src_lo,
            dst_h0=transfer.dst_lo,
            n_heads=transfer.src_hi - transfer.src_lo,
        )
        for transfer in relevant
    ]
