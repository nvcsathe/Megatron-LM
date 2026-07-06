# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared helpers for KV-cache and Mamba-state reshard planners."""

from __future__ import annotations

from typing import Any, Callable, Iterable, List, Optional, Tuple


def intersect(a: Tuple[int, int], b: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """Return the overlap of two half-open ranges, or ``None`` if disjoint."""

    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    return (lo, hi) if lo < hi else None


def transfers_for_src(plan, src_rank):
    """Transfers in ``plan`` originating from ``src_rank`` (any KV/Mamba
    reshard transfer -- both expose a ``src_rank`` field)."""
    return [t for t in plan if t.src_rank == src_rank]


def transfers_for_dst(plan, dst_rank):
    """Transfers in ``plan`` destined for ``dst_rank``."""
    return [t for t in plan if t.dst_rank == dst_rank]


def representative_source_ranks(
    layouts: Iterable[Any], key: Callable[[Any], Any]
) -> set[int]:
    """Select the smallest global rank for each replicated shard key."""

    representatives = {}
    for layout in layouts:
        shard_key = key(layout)
        rank = layout.global_rank
        if shard_key not in representatives or rank < representatives[shard_key]:
            representatives[shard_key] = rank
    return set(representatives.values())


def transfer_peer_records(
    peer_meta: Any, src_block_ids: List[int]
) -> List[Tuple[dict, List[int]]]:
    """Normalize flat/TP/PP transfer metadata into peer/block records."""

    def append_metas(raw_metas: Any, default_blocks: List[int]) -> None:
        metas = raw_metas if isinstance(raw_metas, list) else [raw_metas]
        for meta in metas:
            if not isinstance(meta, dict):
                raise ValueError("transfer peer metadata entries must be dictionaries")
            blocks = meta.get("block_ids", default_blocks)
            records.append((meta, [int(block) for block in blocks]))

    records: List[Tuple[dict, List[int]]] = []
    if isinstance(peer_meta, dict) and "pp_metas" in peer_meta:
        for entry in peer_meta["pp_metas"]:
            raw_metas = entry.get("tp_metas", entry)
            blocks = [int(block) for block in entry.get("block_ids", [])]
            append_metas(raw_metas, blocks)
        return records

    if isinstance(peer_meta, dict) and "tp_metas" in peer_meta:
        peer_meta = peer_meta["tp_metas"]
    blocks = [int(block) for block in src_block_ids]
    append_metas(peer_meta, blocks)
    return records


def transfer_block_count(peer_meta: Any, src_block_ids: List[int]) -> int:
    """Return the sequence-block count represented by transfer metadata."""

    records = transfer_peer_records(peer_meta, src_block_ids)
    return len(records[0][1]) if records else 0
