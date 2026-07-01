# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
from types import SimpleNamespace

from megatron.inference.integrations.dynamo.protocol import (
    logical_replica_group,
    write_ready_descriptor,
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


def test_ready_descriptor_is_valid_json(tmp_path):
    path = tmp_path / "runtime" / "ready.json"
    write_ready_descriptor(str(path), "tcp://127.0.0.1:5555")
    assert json.loads(path.read_text()) == {
        "protocol_version": 1,
        "coordinator_address": "tcp://127.0.0.1:5555",
    }
