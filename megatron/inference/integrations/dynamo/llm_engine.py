# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import signal
import sys
import tempfile
import threading
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from megatron.core.inference.sampling_params import SamplingParams

from dynamo._core import Context
from dynamo.common.backend.disagg import require_prefill_result
from dynamo.common.backend.engine import EngineConfig, GenerateChunk, GenerateRequest, LLMEngine
from dynamo.common.backend.health_check import build_health_check_payload, is_probe
from dynamo.common.backend.publisher import KvEventSource, PushSource
from dynamo.common.backend.worker import WorkerConfig
from dynamo.common.constants import DisaggregationMode
from dynamo.llm import KvEventPublisher, ModelInput

from megatron.inference.integrations.dynamo.args import Config, parse_args
from megatron.inference.integrations.dynamo.client import DynamoInferenceClient

logger = logging.getLogger(__name__)


def build_sampling_params(request: GenerateRequest) -> SamplingParams:
    from megatron.core.inference.sampling_params import SamplingParams

    sampling = request.get("sampling_options") or {}
    stop = request.get("stop_conditions") or {}
    params = SamplingParams()
    if sampling.get("temperature") is not None:
        params.temperature = float(sampling["temperature"])
    if sampling.get("top_p") is not None:
        params.top_p = float(sampling["top_p"])
    if sampling.get("top_k") is not None:
        params.top_k = max(0, int(sampling["top_k"]))
    if stop.get("max_tokens") is not None:
        params.num_tokens_to_generate = int(stop["max_tokens"])
    if stop.get("stop"):
        params.stop_words = list(stop["stop"])
    stop_token_ids = list(stop.get("stop_token_ids") or [])
    if stop.get("ignore_eos"):
        params.termination_id = -1
    elif stop_token_ids:
        params.termination_id = int(stop_token_ids[0])
        if len(stop_token_ids) > 1:
            logger.warning(
                "Megatron supports one termination token; ignoring %d additional IDs",
                len(stop_token_ids) - 1,
            )
    output = request.get("output_options") or {}
    token_logprobs = output.get("logprobs")
    prompt_logprobs = output.get("prompt_logprobs")
    params.top_n_logprobs = max(
        int(token_logprobs or 0), int(prompt_logprobs or 0)
    )
    params.return_log_probs = token_logprobs is not None
    params.skip_prompt_log_probs = prompt_logprobs is None
    params.add_attributes({})
    if params.temperature == 0.0:
        params.top_k = 1
        params.top_p = 0.0
    return params


class MegatronLLMEngine(LLMEngine):
    """Unified Dynamo backend for one self-owned Megatron DP replica."""

    def __init__(
        self,
        config: Config,
    ) -> None:
        self.config = config
        self.client: Any = None
        self._process: Optional[asyncio.subprocess.Process] = None
        self._process_monitor: Optional[asyncio.Task] = None
        self._log_tasks: list[asyncio.Task] = []
        self._runtime_dir: Optional[tempfile.TemporaryDirectory] = None
        self._shutting_down = False
        self._metadata: dict[str, Any] = {}
        self._kv_queue: queue.Queue[tuple[str, dict]] = queue.Queue()
        self._publisher_stop = threading.Event()
        self._publisher_thread: Optional[threading.Thread] = None
        self._release_clients: dict[str, Any] = {}
        self._request_ids: dict[str, int] = {}

    @classmethod
    async def from_args(
        cls, argv: list[str] | None = None
    ) -> tuple["MegatronLLMEngine", WorkerConfig]:
        config = parse_args(argv)
        return cls(config), cls.worker_config(config)

    @staticmethod
    def worker_config(config: Config) -> WorkerConfig:
        mode = {
            "aggregated": DisaggregationMode.AGGREGATED,
            "prefill": DisaggregationMode.PREFILL,
            "decode": DisaggregationMode.DECODE,
        }[config.role]
        return WorkerConfig(
            namespace=config.namespace,
            component=config.component,
            endpoint=config.endpoint,
            model_name=config.model,
            served_model_name=config.served_model_name,
            model_input=ModelInput.Tokens,
            discovery_backend=config.discovery_backend,
            request_plane=config.request_plane,
            event_plane=config.event_plane,
            disaggregation_mode=mode,
            enable_kv_routing=True,
        )

    async def start(self, worker_id: int) -> EngineConfig:
        del worker_id
        if not os.path.isdir(self.config.megatron_root):
            raise FileNotFoundError(
                f"Megatron root does not exist: {self.config.megatron_root}"
            )

        self._runtime_dir = tempfile.TemporaryDirectory(prefix="dynamo-megatron-")
        ready_path = Path(self._runtime_dir.name) / "ready.json"
        command = self._engine_command(ready_path)
        logger.info("Launching owned Megatron engine: %s", " ".join(command))
        self._process = await asyncio.create_subprocess_exec(
            *command,
            cwd=self.config.megatron_root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        assert self._process.stdout is not None
        assert self._process.stderr is not None
        self._log_tasks = [
            asyncio.create_task(self._forward_logs(self._process.stdout, logging.INFO)),
            asyncio.create_task(self._forward_logs(self._process.stderr, logging.WARNING)),
        ]

        readiness = await self._wait_for_readiness(ready_path)
        from megatron.inference.integrations.dynamo.protocol import PROTOCOL_VERSION

        self.client = DynamoInferenceClient(
            str(readiness["coordinator_address"]), deserialize=False
        )
        self.client.start(
            loop=asyncio.get_running_loop(),
            connect_timeout_seconds=min(30.0, self.config.engine_start_timeout),
        )
        self._metadata = dict(self.client.metadata)
        protocol_version = int(self._metadata.get("protocol_version", 0))
        if protocol_version != PROTOCOL_VERSION:
            raise RuntimeError(
                "Megatron coordinator protocol mismatch: "
                f"expected {PROTOCOL_VERSION}, got {protocol_version}"
            )
        self.client.subscribe_telemetry(kv_event_listener=self._on_kv_event)
        self._process_monitor = asyncio.create_task(self._monitor_process())

        return EngineConfig(
            model=self.config.model,
            served_model_name=self.config.served_model_name,
            context_length=int(self._metadata["context_length"]),
            kv_cache_block_size=int(self._metadata["kv_cache_block_size"]),
            total_kv_blocks=int(self._metadata["total_kv_blocks"]),
            max_num_seqs=int(self._metadata["max_num_seqs"]),
            max_num_batched_tokens=int(self._metadata["max_num_batched_tokens"]),
            data_parallel_size=1,
            data_parallel_start_rank=0,
            runtime_data={"role": self.config.role},
        )

    def _engine_command(self, ready_path: Path) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={self.config.nproc_per_node}",
            "--module",
            "megatron.inference.integrations.dynamo.engine_service",
            "--dynamo-ready-file",
            str(ready_path),
            "--dynamo-role",
            self.config.role,
        ]
        if self.config.coordinator_host is not None:
            command.extend(
                ["--dynamo-coordinator-host", self.config.coordinator_host]
            )
        if self.config.coordinator_port is not None:
            command.extend(
                ["--dynamo-coordinator-port", str(self.config.coordinator_port)]
            )
        if self.config.kv_transfer_listen_addr is not None:
            command.extend(
                [
                    "--dynamo-kv-transfer-listen-addr",
                    self.config.kv_transfer_listen_addr,
                ]
            )
        command.extend(self.config.megatron_argv)
        return command

    async def _forward_logs(
        self, stream: asyncio.StreamReader, level: int
    ) -> None:
        while line := await stream.readline():
            logger.log(level, "[megatron-engine] %s", line.decode(errors="replace").rstrip())

    async def _wait_for_readiness(self, ready_path: Path) -> dict[str, Any]:
        deadline = asyncio.get_running_loop().time() + self.config.engine_start_timeout
        while True:
            if ready_path.exists():
                return json.loads(ready_path.read_text())
            if self._process is None:
                raise RuntimeError("Megatron process disappeared during startup")
            if self._process.returncode is not None:
                raise RuntimeError(
                    "Megatron engine exited before readiness "
                    f"with code {self._process.returncode}"
                )
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(
                    "Timed out waiting for the owned Megatron engine to become ready"
                )
            await asyncio.sleep(0.05)

    async def _monitor_process(self) -> None:
        assert self._process is not None
        returncode = await self._process.wait()
        if not self._shutting_down:
            logger.error("Owned Megatron engine exited unexpectedly: %d", returncode)
            os.kill(os.getpid(), signal.SIGTERM)

    async def generate(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        if self.client is None:
            raise RuntimeError("Megatron engine is not initialized")
        token_ids = list(request.get("token_ids") or [])
        if not token_ids:
            raise ValueError("Megatron backend requires token_ids")
        params = build_sampling_params(request)
        n = int((request.get("sampling_options") or {}).get("n") or 1)
        if n != 1:
            raise ValueError("Megatron Dynamo backend currently supports sampling n=1")
        context_id = str(context.id())

        probe = is_probe(request)
        if self.config.role == "prefill" and not probe:
            params.do_kv_handoff = True
            params.num_tokens_to_generate = 0
            stream = self.client.add_request_streaming(token_ids, params)
            self._request_ids[context_id] = stream.request_id
            final = None
            try:
                async for item in stream:
                    if "final" in item:
                        final = item["final"]
                        break
                if final is None:
                    raise RuntimeError("Megatron prefill stream ended without a result")
                disagg = dict(final.get("disaggregated_params") or {})
                disagg["release"] = {
                    "coordinator_addr": self._metadata["coordinator_address"],
                    "request_id": int(
                        disagg.get("request_id", final.get("request_id", stream.request_id))
                    ),
                }
                yield {
                    "token_ids": [],
                    "index": 0,
                    "finish_reason": "stop",
                    "completion_usage": {
                        "prompt_tokens": len(token_ids),
                        "completion_tokens": 0,
                        "total_tokens": len(token_ids),
                    },
                    "disaggregated_params": disagg,
                }
            finally:
                if final is None:
                    await stream.aclose()
                self._request_ids.pop(context_id, None)
            return

        release: dict[str, Any] = {}
        if self.config.role == "decode" and not probe:
            prefill = require_prefill_result(request, DisaggregationMode.DECODE)
            disagg = prefill.get("disaggregated_params") or {}
            release = disagg.get("release") or {}
            stream = self.client.add_request_with_kv_handoff(
                token_ids,
                params,
                disagg.get("kv_meta") or {},
                list(disagg.get("block_ids") or []),
            )
        else:
            stream = self.client.add_request_streaming(token_ids, params)
        self._request_ids[context_id] = stream.request_id

        released = False
        try:
            async for chunk in self._stream_chunks(stream, token_ids, params):
                if not released and self.config.role == "decode":
                    address = release.get("coordinator_addr")
                    request_id = release.get("request_id")
                    if address is not None and request_id is not None:
                        self._release_remote_handoff(str(address), int(request_id))
                        released = True
                yield chunk
        finally:
            self._request_ids.pop(context_id, None)

    async def _stream_chunks(
        self, stream, prompt_token_ids: list[int], params: SamplingParams
    ) -> AsyncGenerator[GenerateChunk, None]:
        completion_tokens = 0
        completed = False
        try:
            async for item in stream:
                if "partial" in item:
                    tokens = list(item["partial"].get("new_tokens") or [])
                    completion_tokens += len(tokens)
                    yield {"token_ids": tokens, "index": 0}
                    continue

                final = item.get("final")
                if final is None:
                    continue
                all_tokens = list(final.get("generated_tokens") or [])
                tokens = all_tokens[completion_tokens:]
                completion_tokens = len(all_tokens)
                max_tokens = params.num_tokens_to_generate
                chunk: GenerateChunk = {
                    "token_ids": tokens,
                    "index": 0,
                    "finish_reason": (
                        "length"
                        if max_tokens is not None and completion_tokens >= max_tokens
                        else "stop"
                    ),
                    "completion_usage": {
                        "prompt_tokens": len(prompt_token_ids),
                        "completion_tokens": completion_tokens,
                        "total_tokens": len(prompt_token_ids) + completion_tokens,
                    },
                }
                completed = True
                yield chunk
                return
            raise RuntimeError("Megatron stream ended without a terminal result")
        finally:
            if not completed:
                await stream.aclose()

    def _release_remote_handoff(self, address: str, request_id: int) -> None:
        from megatron.core.inference.inference_client import InferenceClient

        client = self._release_clients.get(address)
        if client is None:
            client = InferenceClient(address, deserialize=False)
            client.start()
            self._release_clients[address] = client
        client.release_handoff(request_id)

    async def abort(self, context: Context) -> None:
        request_id = self._request_ids.pop(str(context.id()), None)
        if request_id is not None and self.client is not None:
            self.client.abort_request(request_id)

    async def drain(self) -> None:
        if self.client is None:
            return
        deadline = asyncio.get_running_loop().time() + self.config.drain_timeout
        while True:
            status = await self.client.get_status()
            if not int(status.get("active_request_count") or 0) and not int(
                status.get("pinned_handoff_count") or 0
            ):
                return
            if asyncio.get_running_loop().time() >= deadline:
                logger.warning("Timed out draining Megatron requests and KV handoffs")
                return
            await asyncio.sleep(0.05)

    async def cleanup(self) -> None:
        self._shutting_down = True
        self._publisher_stop.set()
        if self._publisher_thread is not None:
            self._publisher_thread.join(timeout=2)
            self._publisher_thread = None
        for client in self._release_clients.values():
            try:
                client.stop()
            except Exception:
                logger.exception("Failed to close Megatron handoff client")
        self._release_clients.clear()
        if self.client is not None:
            client = self.client
            self.client = None
            try:
                await asyncio.wait_for(
                    client.pause_engines_and_wait(),
                    timeout=self.config.engine_shutdown_timeout,
                )
                await asyncio.wait_for(
                    client.stop_engines_and_wait(),
                    timeout=self.config.engine_shutdown_timeout,
                )
                client.shutdown_coordinator()
            except Exception:
                logger.exception("Graceful Megatron coordinator shutdown failed")
            finally:
                try:
                    client.stop()
                except Exception:
                    logger.exception("Failed to close Megatron coordinator client")

        await self._wait_or_terminate_process()
        if self._process_monitor is not None:
            self._process_monitor.cancel()
            await asyncio.gather(self._process_monitor, return_exceptions=True)
            self._process_monitor = None
        if self._log_tasks:
            await asyncio.gather(*self._log_tasks, return_exceptions=True)
            self._log_tasks.clear()
        if self._runtime_dir is not None:
            self._runtime_dir.cleanup()
            self._runtime_dir = None

    async def _wait_or_terminate_process(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            await asyncio.wait_for(
                process.wait(), timeout=self.config.engine_shutdown_timeout
            )
        except asyncio.TimeoutError:
            logger.warning("Terminating unresponsive Megatron engine process group")
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                logger.error("Killing unresponsive Megatron engine process group")
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.wait()
        finally:
            self._process = None

    async def health_check_payload(self) -> dict[str, Any] | None:
        if self.client is None:
            return None
        return build_health_check_payload(int(self._metadata.get("bos_token_id", 0)))

    async def kv_event_sources(self) -> list[KvEventSource]:
        if self.config.role == "decode" or self.client is None:
            return []
        if not self._metadata.get("enable_prefix_caching", False):
            return []
        return [PushSource(on_ready=self._start_publisher_thread, dp_rank=0)]

    def _on_kv_event(self, rank: int, kind: str, payload: dict) -> None:
        from megatron.inference.integrations.dynamo.telemetry import authoritative_kv_event

        event = authoritative_kv_event(rank, kind, payload)
        if event is None:
            return
        self._kv_queue.put(event)

    def _start_publisher_thread(self, publisher: KvEventPublisher) -> None:
        self._publisher_thread = threading.Thread(
            target=self._publish_loop,
            args=(publisher,),
            daemon=True,
            name="megatron-kv-publisher",
        )
        self._publisher_thread.start()

    def _publish_loop(self, publisher: KvEventPublisher) -> None:
        while not self._publisher_stop.is_set():
            try:
                kind, payload = self._kv_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if kind == "stored":
                    publisher.publish_stored(**payload)
                elif kind == "removed":
                    publisher.publish_removed(**payload)
                elif kind == "cleared":
                    publisher.publish_all_cleared()
            except Exception:
                logger.exception("Failed to publish Megatron KV event %s", kind)
