from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine, EngineState


def _context_with_listener():
    context = DynamicInferenceContext.__new__(DynamicInferenceContext)
    listener = Mock()
    context._kv_event_listeners = [listener]
    context._pending_kv_stored_events = []
    return context, listener


def test_stored_event_is_published_only_after_forward_completion():
    context, listener = _context_with_listener()
    payload = {"block_hashes": [101], "token_ids": [1, 2]}

    context._pending_kv_stored_events.append(payload)

    listener.assert_not_called()
    context.publish_pending_kv_stored_events()
    listener.assert_called_once_with("stored", payload)
    assert context._pending_kv_stored_events == []


def test_cache_clear_discards_unpublished_stored_events():
    context, listener = _context_with_listener()

    context._pending_kv_stored_events.append({"block_hashes": [101]})
    context.notify_kv_cache_cleared()

    listener.assert_called_once_with("cleared", {})
    assert context._pending_kv_stored_events == []


def test_next_forward_can_discard_events_left_by_a_failed_forward():
    context, listener = _context_with_listener()

    context._pending_kv_stored_events.append({"block_hashes": [101]})
    context.discard_pending_kv_stored_events()

    listener.assert_not_called()
    assert context._pending_kv_stored_events == []


@pytest.mark.asyncio
async def test_async_forward_discards_before_scheduling_and_publishes_after_forward(monkeypatch):
    context, listener = _context_with_listener()
    context.step_count = 0
    context.prefix_cache_lru_clock = 0
    context.active_token_count = 0
    context.is_decode_only = lambda: False
    context._pending_kv_stored_events.append({"block_hashes": [7]})

    order = []
    payload = {"block_hashes": [101], "token_ids": [1, 2]}
    discard = context.discard_pending_kv_stored_events
    publish = context.publish_pending_kv_stored_events

    def discard_pending():
        order.append("discard")
        discard()

    def schedule():
        order.append("schedule")
        assert context._pending_kv_stored_events == []
        context._pending_kv_stored_events.append(payload)

    async def forward():
        order.append("forward")
        listener.assert_not_called()
        return {"output": True}

    def publish_pending():
        order.append("publish")
        publish()

    context.discard_pending_kv_stored_events = discard_pending
    context.publish_pending_kv_stored_events = publish_pending

    engine = object.__new__(DynamicInferenceEngine)
    engine.state = EngineState.RUNNING
    engine.context = context
    engine.logging_step_interval = 0
    engine.schedule_waiting_requests = schedule
    engine.controller = SimpleNamespace(async_generate_output_tokens_dynamic_batch=forward)

    monkeypatch.setattr(
        "megatron.core.inference.engines.dynamic_engine.nvtx_range_push", lambda *_: None
    )
    monkeypatch.setattr(
        "megatron.core.inference.engines.dynamic_engine.nvtx_range_pop", lambda *_: None
    )

    result, _, _ = await DynamicInferenceEngine.async_forward(engine)

    assert result == {"output": True}
    assert order == ["discard", "schedule", "forward", "publish"]
    listener.assert_called_once_with("stored", payload)
