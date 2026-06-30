# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Async high-level inference API for Megatron (``MegatronAsyncLLM``)."""

import copy
from dataclasses import dataclass
from typing import AsyncIterator, Callable, List, Optional, Union

from megatron.core.inference.apis._llm_base import _MegatronLLMBase
from megatron.core.inference.apis.serve_config import ServeConfig
from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams


@dataclass(frozen=True)
class StreamingInferenceOutput:
    """One token delta or the terminal result for a streaming request."""

    request_id: int
    token_ids: List[int]
    finished: bool
    final_request: Optional[DynamicInferenceRequest] = None


@dataclass(frozen=True)
class EngineMetadata:
    """Capacity and addressing information for one coordinator-backed engine."""

    context_length: int
    block_size_tokens: int
    total_kv_blocks: int
    max_requests: int
    max_tokens: int
    coordinator_address: str
    role: str


class MegatronAsyncLLM(_MegatronLLMBase):
    """Async high-level inference API for Megatron.

    Asyncio-native wrapper over the shared engine + runtime managed by
    :class:`_MegatronLLMBase` -- see that class for caller responsibilities
    and the ``model.eval()`` contract. Requires ``use_coordinator=True``;
    direct mode is rejected at ``__init__`` (see Known Limitations in the
    package README).

    On top of the base, this class provides:

    - ``async generate`` accepting single or batched prompts.
    - ``async`` lifecycle controls: ``pause`` / ``unpause`` / ``suspend`` /
      ``resume`` / ``shutdown`` / ``wait_for_shutdown``.
    - :meth:`serve` for OpenAI-compatible HTTP serving on the primary rank.
    - ``async with`` context-manager protocol; exit calls :meth:`shutdown`.
    """

    def __init__(
        self,
        *,
        model,
        tokenizer,
        inference_config: Optional[InferenceConfig] = None,
        use_coordinator: bool = True,
        coordinator_host: Optional[str] = None,
        coordinator_port: Optional[int] = None,
    ) -> None:
        # MegatronAsyncLLM requires coordinator mode: direct mode invokes the
        # synchronous ``engine.generate()`` from inside the caller's asyncio
        # loop, which collides with the engine's loop-bound internal state
        # (``_cond``, ``_state_events``). Coordinator mode rebinds those to a
        # daemon-thread loop via ``start_listening_to_data_parallel_coordinator``
        # and avoids the conflict.
        if not use_coordinator:
            raise ValueError(
                "MegatronAsyncLLM requires use_coordinator=True. Direct mode is "
                "not supported in async because the underlying engine's "
                "asyncio primitives bind to the caller's loop and collide with "
                "the synchronous engine.generate() path. Use MegatronLLM for "
                "sync direct/coordinator workflows."
            )
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            inference_config=inference_config,
            use_coordinator=use_coordinator,
            coordinator_host=coordinator_host,
            coordinator_port=coordinator_port,
        )
        # Set in serve() when this rank starts the HTTP frontend; consulted by shutdown().
        self._serve_started: bool = False

    async def generate(
        self,
        prompts: Union[str, List[int], List[str], List[List[int]]],
        sampling_params: Optional[SamplingParams] = None,
    ) -> Union["DynamicInferenceRequest", List["DynamicInferenceRequest"]]:
        """Run inference for one prompt or a batch of prompts.

        Single input (``str`` or ``list[int]``) returns a single
        ``DynamicInferenceRequest``; batched input (``list[str]`` or
        ``list[list[int]]``) returns ``list[DynamicInferenceRequest]`` in
        input order.

        Raises:
            RuntimeError: if called on a non-primary rank.
        """
        self._assert_primary()
        if sampling_params is None:
            sampling_params = SamplingParams()

        normalized, is_batch = self._normalize_prompts(prompts)

        if not normalized:
            # Empty batch: nothing to schedule. ``is_batch`` is always True
            # here since single input is wrapped to a one-element list.
            return []

        assert self._loop_manager is not None
        results = await self._loop_manager.run_async(
            self._generate_impl(normalized, sampling_params)
        )
        return results if is_batch else results[0]

    async def _iterate_stream(self, stream) -> AsyncIterator[StreamingInferenceOutput]:
        """Bridge an InferenceClient stream from the runtime loop to the caller loop."""
        assert self._loop_manager is not None
        emitted = 0
        completed = False
        try:
            while True:
                try:
                    item = await self._loop_manager.run_async(stream.__anext__())
                except StopAsyncIteration:
                    return
                if "partial" in item:
                    tokens = list(item["partial"].get("new_tokens") or [])
                    emitted += len(tokens)
                    yield StreamingInferenceOutput(
                        request_id=stream.request_id,
                        token_ids=tokens,
                        finished=False,
                    )
                elif "final" in item:
                    final = item["final"]
                    generated = list(final.generated_tokens or [])
                    completed = True
                    yield StreamingInferenceOutput(
                        request_id=stream.request_id,
                        token_ids=generated[emitted:],
                        finished=True,
                        final_request=final,
                    )
                    return
        finally:
            if not completed:
                await self._loop_manager.run_async(stream.aclose())

    async def generate_stream(
        self,
        prompt: Union[str, List[int]],
        sampling_params: Optional[SamplingParams] = None,
        on_request_started: Optional[Callable[[int], None]] = None,
    ) -> AsyncIterator[StreamingInferenceOutput]:
        """Stream token deltas for one prompt."""
        self._assert_primary()
        if sampling_params is None:
            sampling_params = SamplingParams()
        assert self._loop_manager is not None
        stream = await self._loop_manager.run_async(
            self._start_stream_impl(prompt, copy.deepcopy(sampling_params))
        )
        if on_request_started is not None:
            on_request_started(stream.request_id)
        async for output in self._iterate_stream(stream):
            yield output

    async def prefill_for_handoff(
        self,
        prompt: Union[str, List[int]],
        sampling_params: Optional[SamplingParams] = None,
        on_request_started: Optional[Callable[[int], None]] = None,
    ) -> DynamicInferenceRequest:
        """Populate and pin prompt KV, returning transfer metadata."""
        self._assert_primary()
        params = copy.deepcopy(sampling_params or SamplingParams())
        params.do_kv_handoff = True
        params.streaming = False
        params.num_tokens_to_generate = 0
        assert self._loop_manager is not None
        stream = await self._loop_manager.run_async(self._start_stream_impl(prompt, params))
        if on_request_started is not None:
            on_request_started(stream.request_id)
        async for output in self._iterate_stream(stream):
            if output.finished and output.final_request is not None:
                return output.final_request
        raise RuntimeError("prefill stream ended without a final request")

    async def generate_stream_with_kv_handoff(
        self,
        prompt: Union[str, List[int]],
        sampling_params: SamplingParams,
        kv_meta: dict,
        src_block_ids: List[int],
        on_request_started: Optional[Callable[[int], None]] = None,
    ) -> AsyncIterator[StreamingInferenceOutput]:
        """Import remote KV and stream decode output."""
        self._assert_primary()
        assert self._loop_manager is not None
        stream = await self._loop_manager.run_async(
            self._start_stream_impl(
                prompt,
                copy.deepcopy(sampling_params),
                kv_meta=kv_meta,
                src_block_ids=src_block_ids,
            )
        )
        if on_request_started is not None:
            on_request_started(stream.request_id)
        async for output in self._iterate_stream(stream):
            yield output

    async def abort(self, request_id: int) -> None:
        """Abort an in-flight request by client-visible request id."""
        self._assert_primary()
        assert self._loop_manager is not None
        await self._loop_manager.run_async(self._abort_impl(request_id))

    async def release_handoff(self, request_id: int) -> None:
        """Release KV blocks pinned by a completed prefill request."""
        self._assert_primary()
        assert self._loop_manager is not None
        await self._loop_manager.run_async(self._release_handoff_impl(request_id))

    def add_kv_event_listener(self, listener: Callable[[str, dict], None]) -> None:
        """Register a primary-rank callback for prefix-cache block events."""
        self._assert_primary()
        self.engine.add_kv_event_listener(listener)

    def add_metrics_listener(self, listener: Callable[[dict], None]) -> None:
        """Register a primary-rank callback for per-step load snapshots."""
        self._assert_primary()
        self.engine.add_metrics_listener(listener)

    @property
    def active_request_count(self) -> int:
        """Return the number of scheduled or waiting requests."""
        return len(self.engine.requests)

    @property
    def pinned_handoff_count(self) -> int:
        """Return the number of prefill handoffs awaiting source release."""
        return len(self.engine._pinned_handoff_blocks)

    @property
    def metadata(self) -> EngineMetadata:
        """Return capacity metadata after coordinator startup."""
        self._assert_primary()
        assert self._coord_runtime is not None and self._coord_runtime.coord_addr is not None
        allocator = self.context.kv_block_allocator
        return EngineMetadata(
            context_length=int(self.context.max_sequence_length),
            block_size_tokens=int(self.context.block_size_tokens),
            total_kv_blocks=max(0, int(allocator.total_count) - 1),
            max_requests=int(self.context.max_requests),
            max_tokens=int(self.context.max_tokens),
            coordinator_address=self._coord_runtime.coord_addr,
            role=self.engine.role,
        )

    async def pause(self) -> None:
        """Transition the engine to ``PAUSED``.

        Raises:
            RuntimeError: in direct mode (``use_coordinator=False``).
        """
        self._assert_coordinator()
        assert self._loop_manager is not None
        await self._loop_manager.run_async(self._pause_impl())

    async def unpause(self) -> None:
        """Transition the engine from ``PAUSED`` back to ``RUNNING``.

        Raises:
            RuntimeError: in direct mode (``use_coordinator=False``).
        """
        self._assert_coordinator()
        assert self._loop_manager is not None
        await self._loop_manager.run_async(self._unpause_impl())

    async def suspend(self) -> None:
        """Transition the engine to ``SUSPENDED`` (offloads GPU buffers).

        The caller must ``pause()`` first; this method does not enforce that.

        Raises:
            RuntimeError: in direct mode (``use_coordinator=False``).
        """
        self._assert_coordinator()
        assert self._loop_manager is not None
        await self._loop_manager.run_async(self._suspend_impl())

    async def resume(self) -> None:
        """Transition the engine from ``SUSPENDED`` to ``RESUMED``.

        Raises:
            RuntimeError: in direct mode (``use_coordinator=False``).
        """
        self._assert_coordinator()
        assert self._loop_manager is not None
        await self._loop_manager.run_async(self._resume_impl())

    async def shutdown(self) -> None:
        """Stop the engine, tear down the coordinator, and join the runtime thread.

        Idempotent. No-op in direct mode.
        """
        if self._shutdown_called:
            return
        self._shutdown_called = True

        # If we started an HTTP frontend, stop it first so no new requests
        # arrive while we tear down the coordinator. Invariant:
        # ``_serve_started`` can only be True when ``use_coordinator=True``
        # because ``serve()`` raises otherwise.
        if self._serve_started:
            from megatron.core.inference.text_generation_server.dynamic_text_gen_server.text_generation_server import (  # pylint: disable=line-too-long
                stop_text_gen_server,
            )

            stop_text_gen_server()
            self._serve_started = False

        if not self._use_coordinator:
            return
        assert self._loop_manager is not None
        await self._loop_manager.run_async(self._shutdown_impl())
        self._loop_manager.stop()

    async def serve(self, serve_config: ServeConfig, *, blocking: bool = True) -> None:
        """Start the OpenAI-compatible HTTP frontend.

        Coordinator mode only. The HTTP frontend runs only on the primary
        rank (global rank 0); other ranks no-op the HTTP setup but still
        respect ``blocking`` (so all ranks return together).

        With ``blocking=True`` (default), this awaits the engine loop until
        :meth:`shutdown` is called -- suitable for standalone serving scripts.
        With ``blocking=False``, this returns once the HTTP frontend is up
        (primary) or immediately (workers); the engine loop continues in the
        background runtime, and the user can call :meth:`generate` /
        :meth:`shutdown` afterward.

        Raises:
            ValueError: if ``use_coordinator=False`` (HTTP serving requires
                the coordinator path).
        """
        if not self._use_coordinator:
            raise ValueError("MegatronAsyncLLM.serve() requires use_coordinator=True")

        if self._is_primary_rank:
            # Lazy import: keep the module importable in environments where
            # the HTTP server backend (Quart/Hypercorn) isn't installed.
            import torch.distributed as dist

            from megatron.core.inference.text_generation_server.dynamic_text_gen_server.text_generation_server import (  # pylint: disable=line-too-long
                start_text_gen_server,
            )

            assert self._coord_runtime is not None
            start_text_gen_server(
                coordinator_addr=self._coord_runtime.coord_addr,
                tokenizer=self._controller.tokenizer,
                rank=dist.get_rank(),
                server_port=serve_config.port,
                parsers=serve_config.parsers,
                verbose=serve_config.verbose,
                num_replicas=serve_config.frontend_replicas,
                hostname=serve_config.host,
            )
            self._serve_started = True

        if blocking:
            # Block until the engine loop terminates (shutdown was invoked
            # somewhere in this process; for serve(blocking=True) typically by
            # SIGINT or out-of-band orchestration).
            await self.wait_for_shutdown()

    async def wait_for_shutdown(self) -> None:
        """Block until the engine's background loop task terminates.

        No-op in direct mode.
        """
        if not self._use_coordinator:
            return
        assert self._loop_manager is not None
        await self._loop_manager.run_async(self._wait_for_shutdown_impl())

    async def __aenter__(self) -> "MegatronAsyncLLM":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.shutdown()
