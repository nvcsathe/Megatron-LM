# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Decode-local capacity tests for disaggregated state handoff."""

import asyncio
from collections import deque
from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.contexts.mamba_slot_allocator import MambaSlotCapacityError
from megatron.core.inference.disaggregation.inference_state_handoff import (
    InferenceStateHandoffMixin,
)
from megatron.core.inference.disaggregation.pending_handoff_imports import PendingKvImport
from megatron.core.inference.inference_request import compute_block_hashes_batched
from megatron.core.inference.sampling_params import SamplingParams


class _PendingHandle:
    def poll(self):
        return False

    def wait(self):
        return None


class _TransferAgent:
    def __init__(self):
        self.calls = []
        self.is_push = False

    def begin_pull_blocks(self, peer_meta, src_block_ids, dst_block_ids):
        self.calls.append((peer_meta, list(src_block_ids), list(dst_block_ids)))
        return _PendingHandle()


class _KvAllocator:
    enable_prefix_caching = True

    def __init__(self):
        self.next_block = 10
        self.releases = []
        self.block_ref_counts = torch.zeros(256, dtype=torch.int32)
        self.kv_hash_to_block_id = {}

    def allocate_memory_blocks(self, count):
        blocks = torch.arange(self.next_block, self.next_block + count, dtype=torch.int32)
        self.next_block += count
        self.block_ref_counts[blocks] = 1
        return blocks

    def release_memory_blocks(self, blocks):
        assert torch.all(self.block_ref_counts[blocks] > 0)
        self.block_ref_counts[blocks] -= 1
        self.releases.append(blocks.tolist())

    def register_kv_block_hashes(self, block_ids, block_hashes):
        self.kv_hash_to_block_id.update(zip(block_hashes, block_ids))

    def update_timestamps(self, block_ids):
        return None


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


class _SchedulerHarness:
    def schedule_waiting_requests(self):
        if not self.waiting_request_ids:
            return
        request_id = self.waiting_request_ids[0]
        block_ids = torch.tensor(self.blocks_to_bind[request_id], dtype=torch.int32)
        self.context.kv_block_allocator.block_ref_counts[block_ids] += 1
        if request_id in self.partial_admissions:
            self.get_request(request_id).finished_chunk_token_count += 4
        else:
            self.waiting_request_ids.popleft()


class _HandoffHarness(InferenceStateHandoffMixin, _SchedulerHarness):
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
        self.waiting_request_ids = deque()
        self.requests = {}
        self.blocks_to_bind = {}
        self.precomputed_hashes = {}
        self.partial_admissions = set()

    async def _notify_cond_for_new_request(self):
        return None

    def add_request(self, request_id, prompt, sampling_params, precomputed_block_hashes=None):
        self.requests[request_id] = SimpleNamespace(finished_chunk_token_count=0)
        self.precomputed_hashes[request_id] = precomputed_block_hashes
        self.waiting_request_ids.append(request_id)
        return self._loop.create_future()

    def get_request(self, request_id):
        return self.requests[request_id]


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


def _pending_import(engine, request_id, block_id, block_hash):
    return PendingKvImport(
        request_id=request_id,
        prompt=[1, 2, 3, 4],
        sampling_params=SamplingParams(num_tokens_to_generate=2),
        local_blocks=[block_id],
        hashes=[block_hash],
        hashes_to_register=1,
        hash_registration_start=0,
        handle=None,
        future=engine._loop.create_future(),
    )


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


def test_nixl_handoff_reuses_decode_cached_prefix(handoff_loop):
    engine = _HandoffHarness(handoff_loop, available=0)
    engine.context.mamba_slot_allocator = None
    engine._mamba_transfer_agents = {}
    prompt = [1] * 12
    hashes = compute_block_hashes_batched(
        torch.tensor(prompt), engine.context.block_size_tokens, include_partial=True
    )
    cached = engine.context.kv_block_allocator.allocate_memory_blocks(2)
    engine.context.kv_block_allocator.release_memory_blocks(cached)
    engine.context.kv_block_allocator.kv_hash_to_block_id.update(zip(hashes[:2], cached.tolist()))

    engine.add_request_with_kv_handoff(
        5, prompt, SamplingParams(num_tokens_to_generate=2), {"request_id": 5}, [100, 101, 102]
    )
    _drain_loop(handoff_loop)

    pending = engine._pending_kv_imports[0]
    assert engine._kv_transfer_agent.calls == [({"request_id": 5}, [102], [12])]
    assert pending.local_blocks == [10, 11, 12]
    assert pending.hash_registration_start == 2
    assert pending.hashes_to_register == 1
    engine._finalize_kv_handoff_import(pending)
    assert engine.precomputed_hashes[5] == hashes


def test_nixl_handoff_trims_pipeline_stage_block_lists(handoff_loop):
    engine = _HandoffHarness(handoff_loop, available=0)
    engine.context.mamba_slot_allocator = None
    engine._mamba_transfer_agents = {}
    prompt = [2] * 12
    hashes = compute_block_hashes_batched(
        torch.tensor(prompt), engine.context.block_size_tokens, include_partial=True
    )
    cached = engine.context.kv_block_allocator.allocate_memory_blocks(1)
    engine.context.kv_block_allocator.release_memory_blocks(cached)
    engine.context.kv_block_allocator.kv_hash_to_block_id[hashes[0]] = int(cached[0])
    kv_meta = {
        "pp_metas": [
            {"tp_metas": {"rank": 0}, "block_ids": [100, 101, 102]},
            {"tp_metas": {"rank": 1}, "block_ids": [200, 201, 202]},
        ]
    }

    engine.add_request_with_kv_handoff(
        6, prompt, SamplingParams(num_tokens_to_generate=2), kv_meta, [100, 101, 102]
    )
    _drain_loop(handoff_loop)

    submitted_meta, src_blocks, dst_blocks = engine._kv_transfer_agent.calls[0]
    assert src_blocks == [101, 102]
    assert dst_blocks == [11, 12]
    assert [stage["block_ids"] for stage in submitted_meta["pp_metas"]] == [[101, 102], [201, 202]]


def test_nccl_handoff_does_not_filter_source_push(handoff_loop):
    engine = _HandoffHarness(handoff_loop, available=0)
    engine.context.mamba_slot_allocator = None
    engine._mamba_transfer_agents = {}
    engine._kv_transfer_agent.is_push = True
    prompt = [3] * 8
    hashes = compute_block_hashes_batched(
        torch.tensor(prompt), engine.context.block_size_tokens, include_partial=True
    )
    engine.context.kv_block_allocator.kv_hash_to_block_id[hashes[0]] = 4

    engine.add_request_with_kv_handoff(
        9, prompt, SamplingParams(num_tokens_to_generate=2), {"request_id": 9}, [100, 101]
    )
    _drain_loop(handoff_loop)

    assert engine._kv_transfer_agent.calls == [({"request_id": 9}, [100, 101], [10, 11])]


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


def test_import_owner_survives_hash_replacement_until_request_admission(handoff_loop):
    engine = _HandoffHarness(handoff_loop, available=0)
    first_block = int(engine.context.kv_block_allocator.allocate_memory_blocks(1)[0])
    second_block = int(engine.context.kv_block_allocator.allocate_memory_blocks(1)[0])

    engine._finalize_kv_handoff_import(_pending_import(engine, 1, first_block, 101))
    engine._finalize_kv_handoff_import(_pending_import(engine, 2, second_block, 101))

    assert engine.context.kv_block_allocator.kv_hash_to_block_id[101] == second_block
    engine.blocks_to_bind[1] = [second_block]
    engine.schedule_waiting_requests()

    ref_counts = engine.context.kv_block_allocator.block_ref_counts
    assert ref_counts[first_block] == 0
    assert ref_counts[second_block] == 2
    assert 1 not in engine._handoff_import_owners
    assert 2 in engine._handoff_import_owners


def test_chunked_admission_releases_import_owner(handoff_loop):
    engine = _HandoffHarness(handoff_loop, available=0)
    block_id = int(engine.context.kv_block_allocator.allocate_memory_blocks(1)[0])
    engine._finalize_kv_handoff_import(_pending_import(engine, 3, block_id, 103))
    engine.blocks_to_bind[3] = [block_id]
    engine.partial_admissions.add(3)

    engine.schedule_waiting_requests()

    assert list(engine.waiting_request_ids) == [3]
    assert engine.get_request(3).finished_chunk_token_count == 4
    assert engine.context.kv_block_allocator.block_ref_counts[block_id] == 1
    assert not engine._handoff_import_owners


def test_handoff_import_listener_runs_after_transfer_finalization(handoff_loop):
    engine = _HandoffHarness(handoff_loop, available=0)
    block_id = int(engine.context.kv_block_allocator.allocate_memory_blocks(1)[0])
    events = []
    engine.add_handoff_import_listener(lambda kind, payload: events.append((kind, payload)))

    engine._finalize_kv_handoff_import(_pending_import(engine, 4, block_id, 104))

    assert events == [("handoff_imported", {"request_id": 4})]
