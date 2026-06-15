# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for heterogeneous-TP and heterogeneous-PP KV re-shard planning.

``tp_reshard_plan``  — TP head re-sharding when prefill_TP != decode_TP.
``pp_reshard_plan``  — PP layer re-sharding when prefill_PP != decode_PP.

Both planners are pure index arithmetic (no GPU, no NIXL, no torch) and live
in ``kv_reshard_plan``; this file tests them via their module interface directly.
"""

import base64

import pytest

from megatron.core.inference.kv_reshard_plan import (
    KvTopology,
    TransferSegment,
    build_reshard_plan,
    pp_reshard_plan,
    tp_reshard_plan,
)

G = 8          # num_kv_heads_global (e.g. Llama-3.1-8B num_query_groups)
HEAD_DIM = 128
T = 64         # tokens per block
L = 32         # total attention layers in the model
KV_FACTOR = 2  # K/V split layout: 2 outer slices per layer (K and V)
NUM_OUTER = KV_FACTOR * L  # = 64


# ---------------------------------------------------------------------------
# Synthetic peer-meta helpers
# ---------------------------------------------------------------------------


def _peer_meta(tp_size, tp_rank, num_outer=NUM_OUTER):
    """A synthetic export_meta() dict for a prefill rank at (tp_size, tp_rank)."""
    hpp = G // tp_size
    bytes_per_slice = T * hpp * HEAD_DIM * 2  # 2 bytes (bf16)
    return {
        "agent_name": f"prefill-rank{tp_rank}",
        "agent_metadata_b64": base64.b64encode(b"x").decode(),
        "base_addr": 1_000_000 * (tp_rank + 1),
        "bytes_per_slice": bytes_per_slice,
        "num_outer": num_outer,
        "outer_stride_bytes": bytes_per_slice * 16,
        "num_blocks": 16,
        "device_id": tp_rank,
        "blocks_axis": 2,
        "tp_size": tp_size,
        "tp_rank": tp_rank,
        "num_kv_heads_global": G,
        "heads_per_partition": hpp,
        "head_dim": HEAD_DIM,
        "tokens_per_block": T,
    }


def _make_topology(tp_size, tp_rank, num_outer=NUM_OUTER):
    """KvTopology for a decode rank — TP only (PP=1, all layers)."""
    return KvTopology(
        tp_size=tp_size,
        tp_rank=tp_rank,
        num_kv_heads_global=G,
        heads_per_partition=G // tp_size,
        head_dim=HEAD_DIM,
        tokens_per_block=T,
        pp_rank=0,
        layer_start=0,
        layer_end=L,
        num_outer=num_outer,
    )


def _pp_peer_entry(pp_size, pp_rank, tp_size=1, tp_rank=0, num_blocks=16):
    """Synthetic ``{"tp_metas": ..., "block_ids": [...]}`` entry for one prefill PP rank.

    Mimics what ``_capture_handoff_meta`` packs into ``kv_meta["pp_metas"]``.
    """
    local_layers = L // pp_size
    layer_start = pp_rank * local_layers
    layer_end = layer_start + local_layers
    num_outer = KV_FACTOR * local_layers
    hpp = G // tp_size
    bytes_per_slice = T * hpp * HEAD_DIM * 2
    meta = {
        "agent_name": f"prefill-pp{pp_rank}-tp{tp_rank}",
        "agent_metadata_b64": base64.b64encode(b"x").decode(),
        "base_addr": 1_000_000 * (pp_rank * 10 + tp_rank + 1),
        "bytes_per_slice": bytes_per_slice,
        "num_outer": num_outer,
        "outer_stride_bytes": bytes_per_slice * num_blocks,
        "num_blocks": num_blocks,
        "device_id": pp_rank,
        "blocks_axis": 2,
        "pp_rank": pp_rank,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "tp_size": tp_size,
        "tp_rank": tp_rank,
        "num_kv_heads_global": G,
        "heads_per_partition": hpp,
        "head_dim": HEAD_DIM,
        "tokens_per_block": T,
    }
    block_ids = list(range(pp_rank * num_blocks, (pp_rank + 1) * num_blocks))
    return {"tp_metas": meta, "block_ids": block_ids}


def _make_pp_topology(pp_size, pp_rank, tp_size=1, tp_rank=0):
    """KvTopology for a decode rank with PP topology."""
    local_layers = L // pp_size
    layer_start = pp_rank * local_layers
    layer_end = layer_start + local_layers
    num_outer = KV_FACTOR * local_layers
    return KvTopology(
        tp_size=tp_size,
        tp_rank=tp_rank,
        num_kv_heads_global=G,
        heads_per_partition=G // tp_size,
        head_dim=HEAD_DIM,
        tokens_per_block=T,
        pp_rank=pp_rank,
        layer_start=layer_start,
        layer_end=layer_end,
        num_outer=num_outer,
    )


def _summarize(plan):
    """(peer_tp_rank, src_h0, dst_h0, n_heads) tuples for easy assertions."""
    return [
        (item["peer"]["tp_rank"], item["src_h0"], item["dst_h0"], item["n_heads"])
        for item in plan
    ]


def _summarize_pp(plan):
    """(src_o_start, dst_o_start, n_outer) tuples for easy assertions."""
    return [(item["src_o_start"], item["dst_o_start"], item["n_outer"]) for item in plan]


# ===========================================================================
# TP re-shard plan tests  (tp_reshard_plan)
# ===========================================================================


def test_equal_tp_picks_corresponding_rank_full_slice():
    # decode TP=2, peers are prefill TP=2. Each decode rank pulls exactly its
    # corresponding prefill rank's whole head range.
    peers = [_peer_meta(2, 0), _peer_meta(2, 1)]
    assert _summarize(tp_reshard_plan(peers, _make_topology(2, 0))) == [(0, 0, 0, 4)]
    assert _summarize(tp_reshard_plan(peers, _make_topology(2, 1))) == [(1, 0, 0, 4)]


def test_split_prefill2_to_decode4():
    # prefill TP=2 (4 heads/rank) -> decode TP=4 (2 heads/rank).
    # Each decode rank pulls a sub-range of ONE prefill rank.
    peers = [_peer_meta(2, 0), _peer_meta(2, 1)]
    # decode rank 0 owns heads [0:2) -> prefill rank 0 heads [0:2)
    assert _summarize(tp_reshard_plan(peers, _make_topology(4, 0))) == [(0, 0, 0, 2)]
    # decode rank 1 owns [2:4) -> prefill rank 0 heads [2:4)
    assert _summarize(tp_reshard_plan(peers, _make_topology(4, 1))) == [(0, 2, 0, 2)]
    # decode rank 2 owns [4:6) -> prefill rank 1 heads [0:2)
    assert _summarize(tp_reshard_plan(peers, _make_topology(4, 2))) == [(1, 0, 0, 2)]
    # decode rank 3 owns [6:8) -> prefill rank 1 heads [2:4)
    assert _summarize(tp_reshard_plan(peers, _make_topology(4, 3))) == [(1, 2, 0, 2)]


def test_merge_prefill4_to_decode2():
    # prefill TP=4 (2 heads/rank) -> decode TP=2 (4 heads/rank).
    # Each decode rank gathers from TWO prefill ranks.
    peers = [_peer_meta(4, r) for r in range(4)]
    # decode rank 0 owns [0:4) -> prefill r0 [0:2) and r1 [2:4)
    assert _summarize(tp_reshard_plan(peers, _make_topology(2, 0))) == [
        (0, 0, 0, 2),
        (1, 0, 2, 2),
    ]
    # decode rank 1 owns [4:8) -> prefill r2 [4:6) and r3 [6:8)
    assert _summarize(tp_reshard_plan(peers, _make_topology(2, 1))) == [
        (2, 0, 0, 2),
        (3, 0, 2, 2),
    ]


def test_incomplete_coverage_raises():
    # Only one prefill rank's meta provided but decode rank needs heads beyond
    # it -> plan cannot cover the local range.
    peers = [_peer_meta(2, 0)]  # covers heads [0:4) only
    with pytest.raises(ValueError, match="covers"):
        tp_reshard_plan(peers, _make_topology(2, 1))  # rank 1 needs [4:8)


def test_replication_regime_unsupported():
    # heads_per_partition * tp_size != num_kv_heads_global → not TP-capable.
    bad_topo = KvTopology(
        tp_size=16,
        tp_rank=0,
        num_kv_heads_global=G,
        heads_per_partition=1,   # 1 * 16 != G(8)
        head_dim=HEAD_DIM,
        tokens_per_block=T,
        num_outer=NUM_OUTER,
    )
    with pytest.raises(RuntimeError, match="replication"):
        tp_reshard_plan([_peer_meta(2, 0)], bad_topo)


def test_mismatched_model_raises():
    bad = _peer_meta(2, 0)
    bad["num_kv_heads_global"] = 16  # different model
    with pytest.raises(ValueError, match="num_kv_heads_global mismatch"):
        tp_reshard_plan([bad], _make_topology(2, 0))


# ===========================================================================
# build_reshard_plan segment-type tests
# Regression coverage for the equal-TP shortcut bug:
# prefill TP=1 → decode TP=2 must NOT use the full-slice matched path because
# peer_bps (T×8×d×elem) ≠ local_bps (T×4×d×elem); NIXL rejects the size
# mismatch with NIXL_ERR_INVALID_PARAM / "createXferReq: length mismatch".
# ===========================================================================

SRC_BLOCK_IDS = list(range(8))  # 8 arbitrary src block ids


def test_prefill_tp1_to_decode_tp2_rank0_is_head_subrange():
    # Decode rank 0 (heads [0,4)): src_h0=0, dst_h0=0, n_heads=4 — looks like
    # equal-TP but peer has 8 heads/slice, local has 4 → must NOT be matched.
    peer = _peer_meta(1, 0)   # prefill TP=1: 8 heads/slice
    topo = _make_topology(2, 0)
    segs = build_reshard_plan(peer, SRC_BLOCK_IDS, topo)
    assert len(segs) == 1
    seg = segs[0]
    assert isinstance(seg, TransferSegment)
    assert seg.n_heads == 4, "must use head-subrange path (n_heads>0), not matched path"
    assert seg.src_h0 == 0
    assert seg.dst_h0 == 0


def test_prefill_tp1_to_decode_tp2_rank1_is_head_subrange():
    # Decode rank 1 (heads [4,8)): src_h0=4, dst_h0=0, n_heads=4 — always head-subrange.
    peer = _peer_meta(1, 0)
    topo = _make_topology(2, 1)
    segs = build_reshard_plan(peer, SRC_BLOCK_IDS, topo)
    assert len(segs) == 1
    seg = segs[0]
    assert seg.n_heads == 4
    assert seg.src_h0 == 4
    assert seg.dst_h0 == 0


def test_equal_tp_same_hpp_uses_matched_path():
    # Decode TP=2 rank 0, prefill TP=2 rank 0: heads_per_partition matches (4==4)
    # → shortcut to matched path (n_heads=0) is valid.
    peer = _peer_meta(2, 0)   # prefill TP=2: 4 heads/slice
    topo = _make_topology(2, 0)
    segs = build_reshard_plan(peer, SRC_BLOCK_IDS, topo)
    assert len(segs) == 1
    assert segs[0].n_heads == 0, "equal hpp → matched (full-slice) path"


# ===========================================================================
# PP re-shard plan tests  (pp_reshard_plan)
# ===========================================================================


class TestPpReshardPlan:
    """Pure arithmetic tests for pp_reshard_plan() — no GPU or NIXL needed."""

    def test_matched_pp1(self):
        # Trivial: prefill PP=1, decode PP=1. Single entry covering all layers.
        # src_o_start=0, dst_o_start=0, n_outer = KV_FACTOR*L
        pp_metas = [_pp_peer_entry(pp_size=1, pp_rank=0)]
        plan = pp_reshard_plan(pp_metas, _make_pp_topology(pp_size=1, pp_rank=0))
        assert _summarize_pp(plan) == [(0, 0, KV_FACTOR * L)]

    def test_matched_pp2(self):
        # prefill PP=2, decode PP=2: each decode rank pulls from its matching
        # prefill rank, covering its exact local layer range.
        pp_metas = [_pp_peer_entry(2, 0), _pp_peer_entry(2, 1)]
        half_outer = KV_FACTOR * (L // 2)

        # Decode rank 0 (layers [0, L//2)): pulls from prefill rank 0.
        plan0 = pp_reshard_plan(pp_metas, _make_pp_topology(2, 0))
        assert _summarize_pp(plan0) == [(0, 0, half_outer)]

        # Decode rank 1 (layers [L//2, L)): pulls from prefill rank 1.
        plan1 = pp_reshard_plan(pp_metas, _make_pp_topology(2, 1))
        assert _summarize_pp(plan1) == [(0, 0, half_outer)]

    def test_split_prefill1_to_decode2(self):
        # prefill PP=1 (all 32 layers) → decode PP=2 (16 layers each).
        # Decode rank 0 pulls outer [0, KV_FACTOR*16) from the single prefill.
        # Decode rank 1 pulls outer [KV_FACTOR*16, KV_FACTOR*32) from the same peer.
        pp_metas = [_pp_peer_entry(pp_size=1, pp_rank=0)]
        half = L // 2
        half_outer = KV_FACTOR * half

        plan0 = pp_reshard_plan(pp_metas, _make_pp_topology(2, 0))
        assert _summarize_pp(plan0) == [(0, 0, half_outer)]

        plan1 = pp_reshard_plan(pp_metas, _make_pp_topology(2, 1))
        assert _summarize_pp(plan1) == [(half_outer, 0, half_outer)]

    def test_merge_prefill2_to_decode1(self):
        # prefill PP=2, decode PP=1 (all layers).
        # Decode rank 0 must pull from BOTH prefill ranks.
        pp_metas = [_pp_peer_entry(2, 0), _pp_peer_entry(2, 1)]
        half_outer = KV_FACTOR * (L // 2)

        plan = pp_reshard_plan(pp_metas, _make_pp_topology(1, 0))
        # Two entries: first from prefill-0 (dst 0), then from prefill-1 (dst half_outer).
        assert _summarize_pp(plan) == [
            (0, 0, half_outer),
            (0, half_outer, half_outer),
        ]

    def test_split_prefill1_to_decode4(self):
        # prefill PP=1 → decode PP=4: each decode rank covers L//4 layers.
        pp_metas = [_pp_peer_entry(pp_size=1, pp_rank=0)]
        quarter = L // 4
        qo = KV_FACTOR * quarter
        for dec_rank in range(4):
            plan = pp_reshard_plan(pp_metas, _make_pp_topology(4, dec_rank))
            assert _summarize_pp(plan) == [(dec_rank * qo, 0, qo)], (
                f"decode rank {dec_rank} plan wrong: {_summarize_pp(plan)}"
            )

    def test_block_ids_passed_through(self):
        # Verify that pp_reshard_plan propagates the per-rank block_ids correctly.
        pp_metas = [_pp_peer_entry(2, 0), _pp_peer_entry(2, 1)]
        plan = pp_reshard_plan(pp_metas, _make_pp_topology(1, 0))
        # block_ids for prefill rank 0 and 1 are [0..15] and [16..31] respectively.
        assert plan[0]["block_ids"] == list(range(16))
        assert plan[1]["block_ids"] == list(range(16, 32))

    def test_missing_layer_range_raises(self):
        # An entry without layer_start/layer_end should raise clearly.
        bad_entry = _pp_peer_entry(1, 0)
        bad_meta = dict(bad_entry["tp_metas"])
        del bad_meta["layer_start"]
        bad_entry["tp_metas"] = bad_meta
        with pytest.raises(ValueError, match="layer_start"):
            pp_reshard_plan([bad_entry], _make_pp_topology(1, 0))

    def test_no_pp_topology_on_local_raises(self):
        # Decode topology without layer_start/layer_end cannot plan.
        topo = KvTopology(num_outer=NUM_OUTER)  # layer_start, layer_end, pp_rank all None
        with pytest.raises(RuntimeError, match="PP-capable"):
            pp_reshard_plan([_pp_peer_entry(1, 0)], topo)

    def test_kv_factor_mismatch_raises(self):
        # If the peer's num_outer is not divisible by its layer count, raise.
        bad_entry = _pp_peer_entry(1, 0)
        bad_meta = dict(bad_entry["tp_metas"])
        bad_meta["num_outer"] = bad_meta["num_outer"] + 1  # not divisible
        bad_entry["tp_metas"] = bad_meta
        with pytest.raises(ValueError, match="not divisible"):
            pp_reshard_plan([bad_entry], _make_pp_topology(1, 0))

    def test_incomplete_coverage_raises(self):
        # Only prefill rank 0's data provided; decode PP=1 needs ALL layers.
        # Plan covers half the layers → raise.
        pp_metas = [_pp_peer_entry(2, 0)]  # only first half
        with pytest.raises(ValueError, match="covers"):
            pp_reshard_plan(pp_metas, _make_pp_topology(1, 0))
