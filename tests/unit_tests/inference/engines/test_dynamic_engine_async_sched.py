# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest import mock

import pytest

from megatron.core.inference.config import AsyncScheduleMode
from megatron.core.inference.disaggregation.engine import DisaggDynamicInferenceEngine
from megatron.core.inference.disaggregation.inference_state_handoff import (
    InferenceStateHandoffMixin,
)
from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.sampling_params import SamplingParams


def _make_engine(async_sched_mode=AsyncScheduleMode.SERIAL, **overrides):
    engine = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    context = SimpleNamespace(
        config=SimpleNamespace(async_sched_mode=async_sched_mode),
        is_hybrid_model=False,
        enable_prefix_caching=False,
    )
    model_config = SimpleNamespace(
        expert_model_parallel_size=1, num_moe_experts=None, moe_enable_routing_replay=False
    )
    engine.context = context
    engine.controller = SimpleNamespace(
        inference_wrapped_model=SimpleNamespace(model=SimpleNamespace(config=model_config))
    )
    engine.num_speculative_tokens = 0
    engine.materialize_only_last_token_logits = True

    for name, value in overrides.items():
        if name.startswith("context_"):
            setattr(context, name.removeprefix("context_"), value)
        elif name.startswith("model_config_"):
            setattr(model_config, name.removeprefix("model_config_"), value)
        else:
            setattr(engine, name, value)
    return engine


@pytest.mark.parametrize(
    "overrides, should_raise",
    [
        ({"async_sched_mode": AsyncScheduleMode.LEGACY, "num_speculative_tokens": 1}, False),
        ({}, False),
        ({"async_sched_mode": AsyncScheduleMode.OVERLAP}, False),
        ({"num_speculative_tokens": 1}, True),
        ({"async_sched_mode": AsyncScheduleMode.OVERLAP, "num_speculative_tokens": 1}, True),
        ({"context_is_hybrid_model": True}, True),
        ({"context_enable_prefix_caching": True}, True),
        ({"materialize_only_last_token_logits": False}, True),
        ({"model_config_expert_model_parallel_size": 2}, True),
        ({"model_config_num_moe_experts": 4}, True),
        ({"model_config_moe_enable_routing_replay": True}, True),
    ],
)
def test_validate_async_sched_support_for_config(overrides, should_raise):
    """Ensure engine config validation accepts only supported async scheduling configs."""
    engine = _make_engine(**overrides)

    if should_raise:
        with pytest.raises(ValueError, match="Async scheduling"):
            engine._validate_async_sched_support_for_config()
    else:
        engine._validate_async_sched_support_for_config()


@pytest.mark.parametrize(
    "async_sched_mode, sampling_params, should_raise",
    [
        (AsyncScheduleMode.LEGACY, SamplingParams(top_k=0, top_p=0.5), False),
        (AsyncScheduleMode.SERIAL, SamplingParams(top_k=1, top_p=0.0), False),
        (AsyncScheduleMode.OVERLAP, SamplingParams(top_k=1, top_p=0.0), False),
        (AsyncScheduleMode.SERIAL, SamplingParams(top_k=0, top_p=0.0), True),
        (AsyncScheduleMode.OVERLAP, SamplingParams(top_k=0, top_p=0.0), True),
        (AsyncScheduleMode.SERIAL, SamplingParams(top_k=1, top_p=0.5), True),
        (AsyncScheduleMode.SERIAL, SamplingParams(top_k=1, top_p=0.0, return_log_probs=True), True),
        (AsyncScheduleMode.SERIAL, SamplingParams(top_k=1, top_p=0.0, top_n_logprobs=1), True),
        (AsyncScheduleMode.SERIAL, SamplingParams(top_k=1, top_p=0.0, stop_words=["END"]), True),
    ],
)
def test_validate_async_sched_support_for_request(async_sched_mode, sampling_params, should_raise):
    """Ensure engine request validation accepts only supported async scheduling requests."""
    engine = _make_engine(async_sched_mode=async_sched_mode)
    request = SimpleNamespace(sampling_params=sampling_params)

    if should_raise:
        with pytest.raises(ValueError, match="Async scheduling"):
            engine._validate_async_sched_support_for_request(request)
    else:
        engine._validate_async_sched_support_for_request(request)


def test_add_request_runs_async_sched_request_validation():
    """Ensure request validation is called before mutating engine request state."""
    engine = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    engine._validate_async_sched_support_for_request = mock.Mock(
        side_effect=RuntimeError("validated")
    )
    request = SimpleNamespace(request_id=10)

    with pytest.raises(RuntimeError, match="validated"):
        engine._add_request(request)

    engine._validate_async_sched_support_for_request.assert_called_once_with(request)


def test_base_engine_rejects_kv_handoff_commands():
    engine = DynamicInferenceEngine.__new__(DynamicInferenceEngine)

    assert InferenceStateHandoffMixin not in DynamicInferenceEngine.mro()
    assert engine.pending_kv_import_count == 0
    with pytest.raises(RuntimeError, match="SUBMIT_REQUEST_WITH_KV"):
        engine.add_request_with_kv_handoff(1, [], SamplingParams(), {}, [])
    with pytest.raises(RuntimeError, match="RELEASE_KV"):
        engine.release_handoff_blocks(1)


def test_disagg_engine_resolves_handoff_methods_from_mixin():
    assert DisaggDynamicInferenceEngine.mro()[:3] == [
        DisaggDynamicInferenceEngine,
        InferenceStateHandoffMixin,
        DynamicInferenceEngine,
    ]
    assert (
        DisaggDynamicInferenceEngine.add_request_with_kv_handoff
        is InferenceStateHandoffMixin.add_request_with_kv_handoff
    )


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
    mamba_agent.begin_push_blocks.assert_called_once_with(
        {"tp_metas": [{"agent": "decode"}]}, [3]
    )
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
    engine.context = SimpleNamespace(mamba_slot_allocator=mock.Mock())

    first = SimpleNamespace(request_id=2, disaggregated_params=None)
    second = SimpleNamespace(request_id=3, disaggregated_params=None)
    engine.context.mamba_slot_allocator.get_slot.side_effect = [4, 5, 6]

    pg_size = "megatron.core.inference.disaggregation.inference_state_handoff.get_pg_size"
    with mock.patch(pg_size, return_value=1):
        engine._capture_handoff_meta(first, [10, 11])
        engine._capture_handoff_meta(second, [12])

    assert first.disaggregated_params["kv_meta"]["mamba"]["positions"] == [0, 1]
    assert second.disaggregated_params["kv_meta"]["mamba"]["positions"] == [0]
    assert first.disaggregated_params["kv_meta"] is not second.disaggregated_params["kv_meta"]
    assert "mamba" not in engine._kv_peer_metas
