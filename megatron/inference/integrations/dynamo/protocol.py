# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Private engine-service protocol for the Megatron Dynamo backend."""

from __future__ import annotations

PROTOCOL_VERSION = 3


def logical_replica_group(args, pg_collection):
    """Select the group whose size counts complete logical model replicas."""

    if getattr(args, "expert_model_parallel_size", 1) > 1:
        return pg_collection.expt_dp
    return pg_collection.dp


def _bos_token_id(tokenizer) -> int:
    for name in ("bos", "bos_token_id", "eod"):
        value = getattr(tokenizer, name, None)
        if value is not None:
            return int(value)
    return 0


def build_engine_metadata(engine, args) -> dict:
    """Build the static engine capabilities advertised to the Dynamo parent."""

    allocator = engine.context.kv_block_allocator
    return {
        "context_length": int(engine.context.max_sequence_length),
        "kv_cache_block_size": int(engine.context.block_size_tokens),
        "total_kv_blocks": max(0, int(allocator.total_count) - 1),
        "max_num_seqs": int(engine.context.max_requests),
        "max_num_batched_tokens": int(engine.context.max_tokens),
        "role": str(args.dynamo_role),
        "bos_token_id": _bos_token_id(engine.controller.tokenizer),
        "enable_prefix_caching": bool(engine.context.enable_prefix_caching),
        "logical_data_parallel_size": 1,
    }


def build_ready_payload(coordinator_address: str, engine_metadata: dict) -> dict:
    """Build the first rank-zero message sent to the supervising parent."""

    return {
        "version": PROTOCOL_VERSION,
        "coordinator_address": coordinator_address,
        "engine": dict(engine_metadata),
    }
