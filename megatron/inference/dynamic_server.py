# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Megatron-only distributed engine service for external inference clients.

This module is intended to be launched by ``torch.distributed.run``.  It owns
the model-parallel rank group and private inference coordinator, but no Dynamo
runtime objects.  Rank zero advertises coordinator readiness through an atomic
JSON file supplied by the supervising process.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from megatron.core.utils import get_pg_size
from megatron.inference.utils import add_inference_args, get_dynamic_inference_engine
from megatron.post_training.arguments import add_modelopt_args
from megatron.training import get_args
from megatron.training.arguments import parse_and_validate_args
from megatron.training.initialize import initialize_megatron


def _extra_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = add_inference_args(add_modelopt_args(parser))
    parser.add_argument("--dynamo-ready-file", required=True)
    parser.add_argument(
        "--dynamo-role",
        choices=["aggregated", "prefill", "decode"],
        default="aggregated",
    )
    parser.add_argument("--dynamo-coordinator-host", default=None)
    parser.add_argument("--dynamo-coordinator-port", type=int, default=None)
    parser.add_argument("--dynamo-kv-transfer-listen-addr", default=None)
    return parser


def _logical_replica_group(args, pg_collection):
    if getattr(args, "expert_model_parallel_size", 1) > 1:
        return pg_collection.expt_dp
    return pg_collection.dp


def _bos_token_id(tokenizer) -> int:
    for name in ("bos", "bos_token_id", "eod"):
        value = getattr(tokenizer, name, None)
        if value is not None:
            return int(value)
    return 0


def _engine_metadata(engine, args) -> dict:
    allocator = engine.context.kv_block_allocator
    return {
        "protocol_version": 1,
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


def _write_ready(path: str, coordinator_address: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            {
                "protocol_version": 1,
                "coordinator_address": coordinator_address,
            }
        )
    )
    os.replace(temporary, target)


async def _report_status(engine) -> None:
    while not engine.engine_loop_task.done():
        engine.publish_engine_status()
        await asyncio.sleep(0.1)


async def _serve() -> None:
    args = get_args()
    args.return_log_probs = True
    engine = get_dynamic_inference_engine()

    replica_group = _logical_replica_group(args, engine.pg_collection)
    replica_count = get_pg_size(replica_group)
    if replica_count != 1:
        raw_dp_size = get_pg_size(engine.pg_collection.dp)
        raise ValueError(
            "Megatron Dynamo service requires one complete model replica; "
            f"got logical DP={replica_count}, regular DP={raw_dp_size}, "
            f"EP={args.expert_model_parallel_size}"
        )

    if args.dynamo_role in ("prefill", "decode"):
        engine.setup_kv_transfer(
            role=args.dynamo_role,
            listen_addr=args.dynamo_kv_transfer_listen_addr,
        )

    def publish_metrics(snapshot: dict) -> None:
        engine.publish_metrics_snapshot(snapshot)
        engine.publish_engine_status()

    engine.add_metrics_listener(publish_metrics)
    engine.add_kv_event_listener(engine.publish_kv_event)

    coordinator_address = await engine.start_listening_to_data_parallel_coordinator(
        inference_coordinator_port=args.dynamo_coordinator_port,
        hostname=args.dynamo_coordinator_host,
        engine_metadata=_engine_metadata(engine, args),
    )

    if dist.get_rank() == 0:
        _write_ready(args.dynamo_ready_file, coordinator_address)

    reporter = asyncio.create_task(_report_status(engine))
    try:
        await engine.engine_loop_task
    finally:
        reporter.cancel()
        await asyncio.gather(reporter, return_exceptions=True)


def main() -> None:
    parse_and_validate_args(
        extra_args_provider=_extra_args,
        args_defaults={"no_load_rng": True, "no_load_optim": True},
    )
    initialize_megatron()
    try:
        with torch.inference_mode():
            asyncio.run(_serve())
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
