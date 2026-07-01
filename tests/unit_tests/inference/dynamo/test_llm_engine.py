from __future__ import annotations

from types import SimpleNamespace

import pytest

from megatron.inference.integrations.dynamo.args import Config
from megatron.inference.integrations.dynamo.llm_engine import (
    MegatronLLMEngine,
    build_sampling_params,
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
        kv_transfer_listen_addr="127.0.0.1:7000" if role != "aggregated" else None,
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


class _Context:
    def __init__(self, request_id="dynamo-request"):
        self.request_id = request_id

    def id(self):
        return self.request_id


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
