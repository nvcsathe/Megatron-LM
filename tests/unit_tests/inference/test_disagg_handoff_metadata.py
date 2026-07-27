# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Engine and metadata behavior specific to disaggregated state handoff."""

from types import SimpleNamespace
from unittest import mock

from megatron.core.inference.disaggregation.config import DisaggregationConfig
from megatron.core.inference.disaggregation.engine import DisaggDynamicInferenceEngine


def test_push_handoff_reuses_mamba_slots_advertised_during_capture():
    """SEND_KV must not discover additional Mamba slots after metadata capture."""
    engine = DisaggDynamicInferenceEngine.__new__(DisaggDynamicInferenceEngine)
    engine._initialize_disaggregation_state()
    engine.context = SimpleNamespace(mamba_slot_allocator=mock.Mock())
    engine.context.mamba_slot_allocator.get_slot.side_effect = AssertionError(
        "SEND_KV must use the captured Mamba slots"
    )

    kv_handle = mock.Mock()
    mamba_handle = mock.Mock()
    engine._kv_transfer_agent = mock.Mock()
    engine._kv_transfer_agent.begin_push_blocks.return_value = kv_handle
    mamba_agent = mock.Mock()
    mamba_agent.begin_push_blocks.return_value = mamba_handle
    engine._mamba_transfer_agents = {"conv": mamba_agent}
    engine._pinned_handoff_blocks[7] = [20, 21]
    engine._pinned_handoff_mamba_slots[7] = [3]
    decode_metas = [{"mamba": {"conv": {"agent": "decode"}}}]

    engine.push_handoff_kv(7, decode_metas)

    engine._kv_transfer_agent.begin_push_blocks.assert_called_once_with(
        {"tp_metas": decode_metas}, [20, 21]
    )
    mamba_agent.begin_push_blocks.assert_called_once_with({"tp_metas": [{"agent": "decode"}]}, [3])
    assert engine._pending_kv_pushes == [(7, [kv_handle, mamba_handle])]


def test_capture_handoff_keeps_request_mamba_metadata_independent():
    """A later TP=1 handoff must not replace an earlier request's Mamba positions."""
    engine = DisaggDynamicInferenceEngine.__new__(DisaggDynamicInferenceEngine)
    engine._initialize_disaggregation_state()
    engine.pg_collection = SimpleNamespace(tp=None, pp=None)
    engine._kv_peer_metas = {"transport": "nccl", "global_rank": 0}
    engine._mamba_transfer_agents = {"conv": mock.Mock(), "ssm": mock.Mock()}
    engine._mamba_peer_metas = {
        "conv": {"transport": "nccl", "state": "conv"},
        "ssm": {"transport": "nccl", "state": "ssm"},
    }
    engine.context = SimpleNamespace(mamba_slot_allocator=mock.Mock(), block_size_tokens=4)

    first = SimpleNamespace(request_id=2, prompt_tokens=[0] * 10, disaggregated_params=None)
    second = SimpleNamespace(request_id=3, prompt_tokens=[0] * 6, disaggregated_params=None)
    engine.context.mamba_slot_allocator.get_slot.side_effect = [4, 5, 6]

    pg_size = "megatron.core.inference.disaggregation.inference_state_handoff.get_pg_size"
    with mock.patch(pg_size, return_value=1):
        engine._capture_handoff_meta(first, [10, 11])
        engine._capture_handoff_meta(second, [12])

    assert first.disaggregated_params["kv_meta"]["mamba"]["positions"] == [1]
    assert second.disaggregated_params["kv_meta"]["mamba"]["positions"] == [0]
    assert first.disaggregated_params["kv_meta"] is not second.disaggregated_params["kv_meta"]
    assert "mamba" not in engine._kv_peer_metas


def test_capture_handoff_compacts_only_native_nixl_metadata():
    """Dynamo keeps self-contained metadata; native NIXL uses engine registration."""
    engine = DisaggDynamicInferenceEngine.__new__(DisaggDynamicInferenceEngine)
    engine._initialize_disaggregation_state()
    engine.pg_collection = SimpleNamespace(tp=None, pp=None)
    engine._kv_peer_metas = {
        "agent_name": "prefill-rank0",
        "agent_metadata_b64": "registered",
        "global_rank": 0,
    }
    engine._mamba_transfer_agents = {}
    engine.context = SimpleNamespace()
    dynamo_request = SimpleNamespace(request_id=4, disaggregated_params=None)
    native_request = SimpleNamespace(request_id=5, disaggregated_params=None)

    pg_size = "megatron.core.inference.disaggregation.inference_state_handoff.get_pg_size"
    with mock.patch(pg_size, return_value=1):
        engine._capture_handoff_meta(dynamo_request, [10])
        engine._disagg_config = DisaggregationConfig(
            role="prefill",
            identity="prefill",
            spawn_coordinator=True,
            router="round_robin",
            kv_transport_backend="nixl",
        )
        engine._capture_handoff_meta(native_request, [11])

    assert dynamo_request.disaggregated_params["kv_meta"]["agent_metadata_b64"] == "registered"
    assert "agent_metadata_b64" not in native_request.disaggregated_params["kv_meta"]


def test_capture_handoff_uses_mamba_positions_common_to_tp_and_pp():
    """Only checkpoints present on every source model shard are transferable."""
    engine = DisaggDynamicInferenceEngine.__new__(DisaggDynamicInferenceEngine)
    engine._initialize_disaggregation_state()
    tp_group = object()
    pp_group = object()
    engine.pg_collection = SimpleNamespace(tp=tp_group, pp=pp_group)
    engine._kv_peer_metas = [{"global_rank": 0}, {"global_rank": 1}]
    engine._mamba_transfer_agents = {"conv": mock.Mock(), "ssm": mock.Mock()}
    engine._mamba_peer_metas = {"conv": {"transport": "nccl"}, "ssm": {"transport": "nccl"}}
    engine.context = SimpleNamespace(mamba_slot_allocator=mock.Mock(), block_size_tokens=4)
    engine.context.mamba_slot_allocator.get_slot.side_effect = [4, 5, 6]
    request = SimpleNamespace(request_id=8, prompt_tokens=[0] * 6, disaggregated_params=None)

    remote_tp = {
        "positions": [0, 2],
        "conv": {"transport": "nccl", "block_ids": [40, 42]},
        "ssm": {"transport": "nccl", "block_ids": [40, 42]},
    }

    def gather_with_different_cache_occupancy(output, local_entry, group):
        if group is tp_group:
            output[:] = [local_entry, remote_tp]
            return
        assert group is pp_group
        remote_stage = {
            "positions": [0],
            "conv": [
                {"transport": "nccl", "block_ids": [50]},
                {"transport": "nccl", "block_ids": [60]},
            ],
            "ssm": [
                {"transport": "nccl", "block_ids": [50]},
                {"transport": "nccl", "block_ids": [60]},
            ],
        }
        output[:] = [
            local_entry,
            {
                "kv_meta": [{"global_rank": 2}, {"global_rank": 3}],
                "block_ids": [10, 11, 12],
                "mamba_meta": remote_stage,
            },
        ]

    pg_size = "megatron.core.inference.disaggregation.inference_state_handoff.get_pg_size"
    with (
        mock.patch(pg_size, return_value=2),
        mock.patch("torch.distributed.is_initialized", return_value=True),
        mock.patch(
            "torch.distributed.all_gather_object", side_effect=gather_with_different_cache_occupancy
        ),
    ):
        engine._capture_handoff_meta(request, [10, 11, 12])

    mamba_meta = request.disaggregated_params["kv_meta"]["mamba"]
    assert mamba_meta["positions"] == [0]
    assert engine._pinned_handoff_mamba_slots[8] == [4]
    assert mamba_meta["conv"]["pp_metas"][0]["tp_metas"][0]["block_ids"] == [4]
    assert mamba_meta["conv"]["pp_metas"][0]["tp_metas"][1]["block_ids"] == [40]
    assert mamba_meta["conv"]["pp_metas"][1]["tp_metas"][0]["block_ids"] == [50]
