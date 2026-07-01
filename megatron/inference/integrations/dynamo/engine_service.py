# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Dynamo-owned Megatron distributed engine service.

This module is intended to be launched by ``torch.distributed.run``.  It owns
the model-parallel rank group and private inference coordinator, but no Dynamo
runtime objects.  Rank zero advertises coordinator readiness through an atomic
JSON file supplied by the supervising process.
"""

from __future__ import annotations

import argparse
import asyncio

import torch
import torch.distributed as dist

from megatron.core.utils import get_pg_size
from megatron.inference.integrations.dynamo.protocol import (
    build_engine_metadata,
    logical_replica_group,
    write_ready_descriptor,
)
from megatron.inference.integrations.dynamo.telemetry import (
    attach_engine_telemetry,
    report_engine_status,
)
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


async def _serve() -> None:
    args = get_args()
    args.return_log_probs = True
    engine = get_dynamic_inference_engine()

    replica_group = logical_replica_group(args, engine.pg_collection)
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

    attach_engine_telemetry(engine)

    coordinator_address = await engine.start_listening_to_data_parallel_coordinator(
        inference_coordinator_port=args.dynamo_coordinator_port,
        hostname=args.dynamo_coordinator_host,
        engine_metadata=build_engine_metadata(engine, args),
    )

    if dist.get_rank() == 0:
        write_ready_descriptor(args.dynamo_ready_file, coordinator_address)

    reporter = asyncio.create_task(report_engine_status(engine))
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
