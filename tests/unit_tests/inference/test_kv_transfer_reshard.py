# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for heterogeneous-TP KV-head re-shard planning.

These exercise ``KvTransferAgent.reshard_plan`` — the pure index arithmetic that
decides which (prefill_rank, head_range) fragments a decode rank must pull when
prefill_TP != decode_TP. The planner needs no GPU and no NIXL runtime, so we
build a bare agent via ``__new__`` and set only the topology attributes it reads.

Global KV-head space is ``[0, num_kv_heads_global)`` (== num_query_groups under
GQA). Rank ``r`` at TP ``t`` owns ``[r*Hpp, (r+1)*Hpp)`` with ``Hpp == g / t``.
"""

import base64

import pytest

from megatron.core.inference.kv_transfer import KvTransferAgent

G = 8  # num_kv_heads_global (e.g. Llama-3.1-8B num_query_groups)
HEAD_DIM = 128
T = 64  # tokens per block
NUM_OUTER = 64  # 2 (K/V) * 32 layers


def _peer_meta(tp_size, tp_rank):
    """A synthetic export_meta() dict for a prefill rank at (tp_size, tp_rank)."""
    hpp = G // tp_size
    bytes_per_slice = T * hpp * HEAD_DIM * 2  # 2 bytes (bf16)
    return {
        "agent_name": f"prefill-rank{tp_rank}",
        "agent_metadata_b64": base64.b64encode(b"x").decode(),
        "base_addr": 1_000_000 * (tp_rank + 1),
        "bytes_per_slice": bytes_per_slice,
        "num_outer": NUM_OUTER,
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


def _decode_agent(tp_size, tp_rank):
    """Bare agent with just the attributes reshard_plan() reads."""
    a = object.__new__(KvTransferAgent)
    a._reshard_capable = True
    a._tp_size = tp_size
    a._tp_rank = tp_rank
    a._num_kv_heads_global = G
    a._heads_per_partition = G // tp_size
    a._head_dim = HEAD_DIM
    a._tokens_per_block = T
    a._num_outer = NUM_OUTER
    return a


def _summarize(plan):
    """(peer_tp_rank, src_h0, dst_h0, n_heads) tuples for easy assertions."""
    return [
        (item["peer"]["tp_rank"], item["src_h0"], item["dst_h0"], item["n_heads"])
        for item in plan
    ]


def test_equal_tp_picks_corresponding_rank_full_slice():
    # decode TP=2, peers are prefill TP=2. Each decode rank pulls exactly its
    # corresponding prefill rank's whole head range.
    peers = [_peer_meta(2, 0), _peer_meta(2, 1)]
    assert _summarize(_decode_agent(2, 0).reshard_plan(peers)) == [(0, 0, 0, 4)]
    assert _summarize(_decode_agent(2, 1).reshard_plan(peers)) == [(1, 0, 0, 4)]


def test_split_prefill2_to_decode4():
    # prefill TP=2 (4 heads/rank) -> decode TP=4 (2 heads/rank).
    # Each decode rank pulls a sub-range of ONE prefill rank.
    peers = [_peer_meta(2, 0), _peer_meta(2, 1)]
    # decode rank 0 owns heads [0:2) -> prefill rank 0 heads [0:2)
    assert _summarize(_decode_agent(4, 0).reshard_plan(peers)) == [(0, 0, 0, 2)]
    # decode rank 1 owns [2:4) -> prefill rank 0 heads [2:4)
    assert _summarize(_decode_agent(4, 1).reshard_plan(peers)) == [(0, 2, 0, 2)]
    # decode rank 2 owns [4:6) -> prefill rank 1 heads [0:2)
    assert _summarize(_decode_agent(4, 2).reshard_plan(peers)) == [(1, 0, 0, 2)]
    # decode rank 3 owns [6:8) -> prefill rank 1 heads [2:4)
    assert _summarize(_decode_agent(4, 3).reshard_plan(peers)) == [(1, 2, 0, 2)]


def test_merge_prefill4_to_decode2():
    # prefill TP=4 (2 heads/rank) -> decode TP=2 (4 heads/rank).
    # Each decode rank gathers from TWO prefill ranks.
    peers = [_peer_meta(4, r) for r in range(4)]
    # decode rank 0 owns [0:4) -> prefill r0 [0:2) and r1 [2:4)
    assert _summarize(_decode_agent(2, 0).reshard_plan(peers)) == [
        (0, 0, 0, 2),
        (1, 0, 2, 2),
    ]
    # decode rank 1 owns [4:8) -> prefill r2 [4:6) and r3 [6:8)
    assert _summarize(_decode_agent(2, 1).reshard_plan(peers)) == [
        (2, 0, 0, 2),
        (3, 0, 2, 2),
    ]


def test_incomplete_coverage_raises():
    # Only one prefill rank's meta provided but decode rank needs heads beyond
    # it -> plan cannot cover the local range.
    peers = [_peer_meta(2, 0)]  # covers heads [0:4) only
    with pytest.raises(ValueError, match="covers"):
        _decode_agent(2, 1).reshard_plan(peers)  # rank 1 needs [4:8)


def test_replication_regime_unsupported():
    # TP > num_kv_heads_global -> heads replicated, not partitioned.
    a = _decode_agent(2, 0)
    a._tp_size = 16
    a._heads_per_partition = 1  # 1 * 16 != G(8)
    with pytest.raises(NotImplementedError, match="replication"):
        a.reshard_plan([_peer_meta(2, 0)])


def test_mismatched_model_raises():
    a = _decode_agent(2, 0)
    bad = _peer_meta(2, 0)
    bad["num_kv_heads_global"] = 16  # different model
    with pytest.raises(ValueError, match="num_kv_heads_global mismatch"):
        a.reshard_plan([bad])
