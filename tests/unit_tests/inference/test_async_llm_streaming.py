from types import SimpleNamespace

import pytest

from megatron.core.inference.apis.async_llm import MegatronAsyncLLM
from megatron.core.inference.inference_client import InferenceStream


class _LoopManager:
    async def run_async(self, coro):
        return await coro


class _Stream:
    request_id = 17

    def __init__(self, items):
        self.items = iter(items)
        self.closed = False

    async def __anext__(self):
        try:
            return next(self.items)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_iterate_stream_emits_deltas_and_terminal_tail():
    llm = object.__new__(MegatronAsyncLLM)
    llm._loop_manager = _LoopManager()
    final = SimpleNamespace(generated_tokens=[10, 11, 12])
    stream = _Stream(
        [
            {"partial": {"request_id": 17, "new_tokens": [10, 11]}},
            {"final": final},
        ]
    )

    outputs = [item async for item in llm._iterate_stream(stream)]

    assert outputs[0].token_ids == [10, 11]
    assert not outputs[0].finished
    assert outputs[1].token_ids == [12]
    assert outputs[1].finished
    assert outputs[1].final_request is final
    assert not stream.closed


@pytest.mark.asyncio
async def test_iterate_stream_closes_remote_request_when_consumer_disconnects():
    llm = object.__new__(MegatronAsyncLLM)
    llm._loop_manager = _LoopManager()
    stream = _Stream([{"partial": {"request_id": 17, "new_tokens": [10]}}])
    generator = llm._iterate_stream(stream)

    await generator.__anext__()
    await generator.aclose()

    assert stream.closed


@pytest.mark.asyncio
async def test_inference_stream_aclose_aborts_exact_request():
    aborted = []
    client = SimpleNamespace(abort_request=aborted.append)
    stream = InferenceStream(client, 42, __import__("asyncio").Queue())

    await stream.aclose()

    assert aborted == [42]
