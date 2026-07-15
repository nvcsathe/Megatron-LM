from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("dynamo")

from megatron.core.inference.disaggregation.inference_state_handoff import (
    InferenceStateHandoffMixin,
)
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
from megatron.inference.integrations.dynamo.args import Config
from megatron.inference.integrations.dynamo.dynamic_engine import DynamoDynamicInferenceEngine
from megatron.inference.integrations.dynamo.llm_engine import (
    MegatronLLMEngine,
    build_sampling_params,
)


def test_dynamo_engine_resolves_handoff_methods_from_mixin():
    assert DynamoDynamicInferenceEngine.mro()[:3] == [
        DynamoDynamicInferenceEngine,
        InferenceStateHandoffMixin,
        DynamicInferenceEngine,
    ]
    assert (
        DynamoDynamicInferenceEngine.add_request_with_kv_handoff
        is InferenceStateHandoffMixin.add_request_with_kv_handoff
    )


def _config(role="aggregated"):
    return Config(
        model="model",
        served_model_name="served",
        namespace="dynamo",
        component="prefill" if role == "prefill" else "backend",
        endpoint="generate",
        discovery_backend="etcd",
        request_plane="nats",
        event_plane="nats",
        role=role,
        nproc_per_node=1,
        coordinator_host=None,
        coordinator_port=None,
        worker_id_file=None,
        megatron_root="/opt/megatron-lm",
        drain_timeout=0.1,
        megatron_argv=["--load", "/checkpoint"],
    )


def test_sampling_params_maps_greedy_and_limits():
    params = build_sampling_params(
        {
            "token_ids": [1],
            "sampling_options": {"temperature": 0.0, "top_p": 0.9},
            "stop_conditions": {"max_tokens": 7},
        }
    )
    assert params.top_k == 1
    assert params.top_p == 0.0
    assert params.num_tokens_to_generate == 7


@pytest.mark.asyncio
async def test_start_uses_parent_event_socket_and_base_client(tmp_path):
    config = _config()
    config.megatron_root = str(tmp_path)
    engine = MegatronLLMEngine(config)
    stdout = asyncio.StreamReader()
    stderr = asyncio.StreamReader()
    stdout.feed_eof()
    stderr.feed_eof()
    process_exit = asyncio.Event()

    async def wait_for_process():
        await process_exit.wait()
        return 0

    process = SimpleNamespace(stdout=stdout, stderr=stderr, returncode=None, wait=wait_for_process)
    metadata = {
        "context_length": 8192,
        "kv_cache_block_size": 64,
        "total_kv_blocks": 100,
        "max_num_seqs": 32,
        "max_num_batched_tokens": 4096,
    }
    client = SimpleNamespace(start=MagicMock())
    event_receiver = SimpleNamespace(start=MagicMock(return_value="tcp://127.0.0.1:5556"))
    engine._wait_for_readiness = AsyncMock(
        return_value={
            "version": 3,
            "coordinator_address": "tcp://127.0.0.1:5555",
            "engine": metadata,
        }
    )

    try:
        with (
            patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)),
            patch(
                "megatron.inference.integrations.dynamo.llm_engine.InferenceClient",
                return_value=client,
            ) as client_class,
            patch(
                "megatron.inference.integrations.dynamo.llm_engine.EngineEventReceiver",
                return_value=event_receiver,
            ) as receiver_class,
        ):
            await engine.start(worker_id=0)

        client_class.assert_called_once_with("tcp://127.0.0.1:5555", deserialize=False)
        client.start.assert_called_once()
        receiver_class.assert_called_once_with(engine._on_engine_event, "127.0.0.1")
        event_receiver.start.assert_called_once()
    finally:
        if engine._process_monitor is not None:
            engine._process_monitor.cancel()
            await asyncio.gather(engine._process_monitor, return_exceptions=True)
        await asyncio.gather(*engine._log_tasks)


class _Context:
    def __init__(self, request_id="dynamo-request"):
        self.request_id = request_id

    def id(self):
        return self.request_id


class _FinalStream:
    request_id = 31

    def __init__(self, final):
        self.final = final
        self.done = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.done:
            raise StopAsyncIteration
        self.done = True
        return {"final": self.final}

    async def aclose(self):
        return None


@pytest.mark.asyncio
async def test_aggregated_usage_forwards_cached_tokens():
    engine = MegatronLLMEngine(_config())
    params = SimpleNamespace(num_tokens_to_generate=2)
    stream = _FinalStream({"generated_tokens": [7, 8], "num_cached_tokens": 64})

    chunks = [chunk async for chunk in engine._stream_chunks(stream, [1, 2, 3], params)]

    assert chunks[-1]["completion_usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
        "prompt_tokens_details": {"cached_tokens": 64},
    }


@pytest.mark.asyncio
async def test_decode_usage_uses_prefill_cache_count_not_handoff_match_count():
    engine = MegatronLLMEngine(_config())
    params = SimpleNamespace(num_tokens_to_generate=1)
    stream = _FinalStream({"generated_tokens": [9], "num_cached_tokens": 128})

    chunks = [
        chunk
        async for chunk in engine._stream_chunks(
            stream, [1, 2, 3], params, cached_tokens_override=32
        )
    ]

    assert chunks[-1]["completion_usage"]["prompt_tokens_details"] == {"cached_tokens": 32}


@pytest.mark.asyncio
async def test_decode_health_probe_bypasses_kv_handoff():
    handoff_called = False

    class Stream:
        request_id = 31

        def __aiter__(self):
            return self

        async def __anext__(self):
            if getattr(self, "done", False):
                raise StopAsyncIteration
            self.done = True
            return {"final": {"generated_tokens": [9]}}

        async def aclose(self):
            return None

    def add_request_with_kv_handoff(*_args, **_kwargs):
        nonlocal handoff_called
        handoff_called = True
        raise AssertionError("health probe must not import KV")

    engine = MegatronLLMEngine(_config("decode"))
    engine.client = SimpleNamespace(
        add_request_streaming=lambda *_args, **_kwargs: Stream(),
        add_request_with_kv_handoff=add_request_with_kv_handoff,
    )
    request = {
        "token_ids": [1],
        "_HEALTH_CHECK": True,
        "sampling_options": {},
        "stop_conditions": {"max_tokens": 1},
    }

    chunks = [chunk async for chunk in engine.generate(request, _Context())]

    assert chunks[-1]["token_ids"] == [9]
    assert chunks[-1]["finish_reason"] == "length"
    assert not handoff_called


@pytest.mark.asyncio
async def test_abort_uses_megatron_request_id_recorded_for_context():
    aborted = []

    engine = MegatronLLMEngine(_config())
    engine.client = SimpleNamespace(abort_request=aborted.append)
    engine._request_ids["dynamo-request"] = 77

    await engine.abort(_Context())

    assert aborted == [77]
