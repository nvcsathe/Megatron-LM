# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Dynamo readiness and acknowledged coordinator-control protocol."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from megatron.core.inference.headers import Headers

try:
    import msgpack
except ImportError:
    msgpack = None


PROTOCOL_VERSION = 1


def logical_replica_group(args, pg_collection):
    """Select the group whose size counts complete logical model replicas."""

    if getattr(args, "expert_model_parallel_size", 1) > 1:
        return pg_collection.expt_dp
    return pg_collection.dp


def _bos_token_id(tokenizer) -> int:
    for name in ("bos", "bos_token_id", "eod"):
        value = getattr(tokenizer, name, None)
        if value is not None:
            return int(value)
    return 0


def build_engine_metadata(engine, args) -> dict:
    """Build the Dynamo-visible descriptor for one owned Megatron engine."""

    allocator = engine.context.kv_block_allocator
    return {
        "protocol_version": PROTOCOL_VERSION,
        "context_length": int(engine.context.max_sequence_length),
        "kv_cache_block_size": int(engine.context.block_size_tokens),
        "total_kv_blocks": max(0, int(allocator.total_count) - 1),
        "max_num_seqs": int(engine.context.max_requests),
        "max_num_batched_tokens": int(engine.context.max_tokens),
        "role": str(args.dynamo_role),
        "bos_token_id": _bos_token_id(engine.controller.tokenizer),
        "enable_prefix_caching": bool(engine.context.enable_prefix_caching),
        "logical_data_parallel_size": 1,
    }


def write_ready_descriptor(path: str, coordinator_address: str) -> None:
    """Atomically publish the owned engine's coordinator address."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            {
                "protocol_version": PROTOCOL_VERSION,
                "coordinator_address": coordinator_address,
            }
        )
    )
    os.replace(temporary, target)


class DynamoClientProtocolMixin:
    """Acknowledged management operations added to ``InferenceClient``."""

    def _initialize_dynamo_protocol(self) -> None:
        self.metadata: dict = {}
        self._next_management_id = 0
        self._management_futures: dict[int, asyncio.Future] = {}

    def _handle_dynamo_protocol_message(self, header: Headers, data: list) -> bool:
        if header not in (
            Headers.METADATA_REPLY,
            Headers.STATUS_REPLY,
            Headers.CONTROL_ACK,
        ):
            return False
        operation_id, payload = data[1:]
        future = self._management_futures.pop(int(operation_id), None)
        if future is not None and not future.done():
            future.set_result(payload)
        return True

    def _management_request(self, header: Headers) -> asyncio.Future:
        if self._loop is None:
            raise RuntimeError("InferenceClient.start() must be called first")
        operation_id = self._next_management_id
        self._next_management_id += 1
        future = self._loop.create_future()
        self._management_futures[operation_id] = future
        self._send_signal_to_engines(header, operation_id)
        return future

    async def get_status(self) -> dict:
        return dict(await self._management_request(Headers.GET_STATUS))

    async def pause_engines_and_wait(self) -> dict:
        return dict(await self._management_request(Headers.PAUSE))

    async def stop_engines_and_wait(self) -> dict:
        return dict(await self._management_request(Headers.STOP))

    def _cancel_dynamo_protocol_futures(self) -> None:
        for future in self._management_futures.values():
            if not future.done():
                future.cancel()
        self._management_futures.clear()


class DynamoCoordinatorProtocolMixin:
    """Metadata/status request handling added to the Megatron coordinator."""

    def _initialize_dynamo_protocol(self, engine_metadata, data_parallel_size: int) -> None:
        self.engine_metadata = dict(engine_metadata or {})
        self.engine_metadata.setdefault("protocol_version", PROTOCOL_VERSION)
        self.engine_metadata.setdefault("coordinator_address", self.addr)
        self.engine_metadata.setdefault("data_parallel_size", data_parallel_size)
        self._pending_controls: dict[tuple[bytes, int], tuple[str, set[bytes]]] = {}

    def _dynamo_connect_ack(self) -> bytes:
        return msgpack.packb(
            [Headers.CONNECT_ACK.value, self.engine_metadata], use_bin_type=True
        )

    def _handle_dynamo_management_request(
        self, header: Headers, sender_identity: bytes, payload: list, known_clients: set
    ) -> bool:
        if header not in (Headers.GET_METADATA, Headers.GET_STATUS):
            return False
        if sender_identity not in known_clients:
            return True
        operation_id = int(payload[1])
        if header == Headers.GET_METADATA:
            reply_header = Headers.METADATA_REPLY
            reply = self.engine_metadata
        else:
            reply_header = Headers.STATUS_REPLY
            reply = self._status_payload()
        self.router_socket.send_multipart(
            [
                sender_identity,
                msgpack.packb(
                    [reply_header.value, operation_id, reply], use_bin_type=True
                ),
            ]
        )
        return True

    def _send_dynamo_control_ack(self, client: bytes, operation_id: int) -> None:
        self.router_socket.send_multipart(
            [
                client,
                msgpack.packb(
                    [Headers.CONTROL_ACK.value, operation_id, self._status_payload()],
                    use_bin_type=True,
                ),
            ]
        )

    def _acknowledge_idempotent_dynamo_control(
        self, sender_identity: bytes, payload: list
    ) -> None:
        if len(payload) > 1:
            self._send_dynamo_control_ack(sender_identity, int(payload[1]))

    def _track_dynamo_control(
        self, header: Headers, sender_identity: bytes, payload: list
    ) -> None:
        if header not in (Headers.PAUSE, Headers.STOP) or len(payload) <= 1:
            return
        operation_id = int(payload[1])
        target = "paused" if header == Headers.PAUSE else "stopped"
        pending = set(self.identities_of_data_parallel_ranks)
        control_key = (sender_identity, operation_id)
        self._pending_controls[control_key] = (target, pending)
        for identity, status in self._latest_engine_status.items():
            if str(status.get("state", "")).lower() == target:
                pending.discard(identity)
        if not pending:
            self._send_dynamo_control_ack(sender_identity, operation_id)
            del self._pending_controls[control_key]
