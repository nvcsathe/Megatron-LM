# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

from megatron.inference.integrations.dynamo.protocol import (
    build_ready_payload,
    logical_replica_group,
)


def test_logical_replica_group_uses_expert_dp_for_moe():
    groups = SimpleNamespace(dp=object(), expt_dp=object())
    assert (
        logical_replica_group(
            SimpleNamespace(expert_model_parallel_size=2), groups
        )
        is groups.expt_dp
    )
    assert (
        logical_replica_group(
            SimpleNamespace(expert_model_parallel_size=1), groups
        )
        is groups.dp
    )


def test_ready_payload_contains_startup_contract():
    assert build_ready_payload(
        "tcp://127.0.0.1:5555",
        {"context_length": 8192},
    ) == {
        "version": 3,
        "coordinator_address": "tcp://127.0.0.1:5555",
        "engine": {"context_length": 8192},
    }
