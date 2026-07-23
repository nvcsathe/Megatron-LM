# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Decode-local capacity tests for disaggregated state handoff."""

import asyncio
from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.contexts.mamba_slot_allocator import MambaSlotCapacityError
from megatron.core.inference.disaggregation.inference_state_handoff import (
    InferenceStateHandoffMixin,
)
from megatron.core.inference.sampling_params import SamplingParams


class _PendingHandle:
    def poll(self):
        return False

    def wait(self):
        return None


class _TransferAgent:
    def __init__(self):
        self.calls = []

    def begin_pull_blocks(self, peer_meta, src_block_ids, dst_block_ids):
        self.calls.append((peer_meta, list(src_block_ids), list(dst_block_ids)))
        return _PendingHandle()


class _KvAllocator:
    enable_prefix_caching = True

    def __init__(self):
        self.next_block = 10
        self.releases = []

    def allocate_memory_blocks(self, count):
        blocks = torch.arange(self.next_block, self.next_block + count, dtype=torch.int32)
        self.next_block += count
        return blocks

    def release_memory_blocks(self, blocks):
        self.releases.append(blocks.tolist())


class _MambaAllocator:
    def __init__(self, available):
        self.available = available
        self.next_slot = 20
        self.invalidated = []

    def allocate_slots_batch(self, block_ids):
        required = len(set(block_ids))
        if required > self.available:
            raise MambaSlotCapacityError(required=required, available=self.available)
        slots = list(range(self.next_slot, self.next_slot + required))
        self.next_slot += required
        self.available -= required
        return slots

    def invalidate_block(self, block_id):
        self.invalidated.append(block_id)


class _HandoffHarness(InferenceStateHandoffMixin):
    def __init__(self, loop, available):
        self._loop = loop
        self._initialize_disaggregation_state()
        self.context = SimpleNamespace(
            block_size_tokens=4,
            kv_block_allocator=_KvAllocator(),
            mamba_slot_allocator=_MambaAllocator(available),
            memory_buffer=torch.empty(1),
        )
        self._kv_transfer_agent = _TransferAgent()
        self._mamba_transfer_agents = {"conv": _TransferAgent(), "ssm": _TransferAgent()}
        self.pg_collection = SimpleNamespace(mp=None)

    async def _notify_cond_for_new_request(self):
        return None


def _meta(request_id, positions):
    return {
        "request_id": request_id,
        "mamba": {
            "positions": positions,
            "conv": {"request_id": request_id},
            "ssm": {"request_id": request_id},
        },
    }


def _drain_loop(loop):
    loop.run_until_complete(asyncio.sleep(0))


@pytest.fixture
def handoff_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def test_capacity_miss_defers_before_any_transfer(handoff_loop):
    engine = _HandoffHarness(handoff_loop, available=1)
    future = engine.add_request_with_kv_handoff(
        7, [1, 2, 3, 4, 5], SamplingParams(num_tokens_to_generate=2), _meta(7, [0, 1]), [100, 101]
    )

    assert not future.done()
    assert engine.pending_kv_import_count == 1
    assert len(engine._deferred_kv_handoffs) == 1
    assert not engine._pending_kv_imports
    assert not engine._kv_transfer_agent.calls
    assert not engine._mamba_transfer_agents["conv"].calls
    assert engine.context.kv_block_allocator.releases == [[10, 11]]

    engine.context.mamba_slot_allocator.available = 2
    assert engine._poll_pending_kv_imports() == 0
    _drain_loop(handoff_loop)

    assert not engine._deferred_kv_handoffs
    assert len(engine._pending_kv_imports) == 1
    assert engine.pending_kv_import_count == 1
    assert len(engine._kv_transfer_agent.calls) == 1
    assert len(engine._mamba_transfer_agents["conv"].calls) == 1
    assert len(engine._mamba_transfer_agents["ssm"].calls) == 1


def test_attention_only_handoff_has_no_mamba_admission_overhead(handoff_loop):
    engine = _HandoffHarness(handoff_loop, available=0)
    engine.context.mamba_slot_allocator = None
    engine._mamba_transfer_agents = {}

    future = engine.add_request_with_kv_handoff(
        4, [1, 2, 3, 4], SamplingParams(num_tokens_to_generate=2), {"request_id": 4}, [100]
    )
    _drain_loop(handoff_loop)

    assert not future.done()
    assert not engine._deferred_kv_handoffs
    assert len(engine._pending_kv_imports) == 1
    assert len(engine._kv_transfer_agent.calls) == 1
    assert engine.context.kv_block_allocator.releases == []


def test_capacity_queue_is_fifo(handoff_loop):
    engine = _HandoffHarness(handoff_loop, available=1)
    engine.add_request_with_kv_handoff(
        1, [1] * 8, SamplingParams(num_tokens_to_generate=2), _meta(1, [0, 1]), [100, 101]
    )
    engine.add_request_with_kv_handoff(
        2, [2] * 4, SamplingParams(num_tokens_to_generate=2), _meta(2, [0]), [102]
    )

    assert [item.request_id for item in engine._deferred_kv_handoffs] == [1, 2]
    assert not engine._kv_transfer_agent.calls

    engine.context.mamba_slot_allocator.available = 2
    engine._poll_pending_kv_imports()
    _drain_loop(handoff_loop)
    assert [item.request_id for item in engine._deferred_kv_handoffs] == [2]
    assert [call[0]["request_id"] for call in engine._kv_transfer_agent.calls] == [1]

    engine.context.mamba_slot_allocator.available = 1
    engine._poll_pending_kv_imports()
    _drain_loop(handoff_loop)
    assert not engine._deferred_kv_handoffs
    assert [call[0]["request_id"] for call in engine._kv_transfer_agent.calls] == [1, 2]


def test_peer_capacity_miss_rolls_back_before_any_transfer(handoff_loop, monkeypatch):
    engine = _HandoffHarness(handoff_loop, available=2)
    engine.pg_collection.mp = object()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)

    def report_peer_capacity_miss(agreement, op, group):
        agreement.copy_(torch.tensor([0, 1, -2], dtype=agreement.dtype))

    monkeypatch.setattr(torch.distributed, "all_reduce", report_peer_capacity_miss)

    future = engine.add_request_with_kv_handoff(
        8, [1, 2, 3, 4, 5], SamplingParams(num_tokens_to_generate=2), _meta(8, [0, 1]), [100, 101]
    )

    assert not future.done()
    assert [item.request_id for item in engine._deferred_kv_handoffs] == [8]
    assert not engine._kv_transfer_agent.calls
    assert not engine._mamba_transfer_agents["conv"].calls
    assert engine.context.mamba_slot_allocator.invalidated == [10, 11]
    assert engine.context.kv_block_allocator.releases == [[10, 11]]


def test_reset_cancels_capacity_queued_handoffs(handoff_loop):
    engine = _HandoffHarness(handoff_loop, available=0)
    future = engine.add_request_with_kv_handoff(
        3, [3] * 4, SamplingParams(num_tokens_to_generate=2), _meta(3, [0]), [103]
    )

    engine._reset_pending_kv_imports()

    assert future.cancelled()
    assert engine.pending_kv_import_count == 0
