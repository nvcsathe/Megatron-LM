# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Dynamo engine telemetry publication and coordinator forwarding."""

from __future__ import annotations

import asyncio

from megatron.core.inference.headers import Headers

try:
    import msgpack
except ImportError:
    msgpack = None

try:
    import zmq
except ImportError:
    zmq = None


class DynamoEngineTelemetryMixin:
    """Publish rank-tagged KV events and engine status."""

    def _send_coordinator_event(self, header: Headers, *payload) -> None:
        if not getattr(self, "is_mp_coordinator", False):
            return
        socket = getattr(self, "socket_for_receiving_requests", None)
        if socket is None or socket.closed:
            return
        socket.send(msgpack.packb([header.value, *payload], use_bin_type=True))

    def publish_kv_event(self, kind: str, payload: dict) -> None:
        tagged = dict(payload)
        tagged.setdefault("source_global_rank", self.rank)
        self._send_coordinator_event(Headers.KV_EVENT, kind, tagged)

    def engine_status(self) -> dict:
        return {
            "state": self.state.name.lower(),
            "active_request_count": max(
                0, len(self.requests) - len(self.waiting_request_ids)
            ),
            "waiting_request_count": len(self.waiting_request_ids),
            "pinned_handoff_count": self.pinned_handoff_count,
            "global_rank": self.rank,
        }

    def publish_engine_status(self) -> None:
        self._send_coordinator_event(Headers.ENGINE_STATUS, self.engine_status())


class DynamoClientTelemetryMixin:
    """Receive coordinator-forwarded telemetry in ``InferenceClient``."""

    def _initialize_dynamo_telemetry(self) -> None:
        self._kv_event_listeners = []

    def _handle_dynamo_telemetry_message(self, header: Headers, data: list) -> bool:
        if header == Headers.KV_EVENT:
            rank, kind, payload = data[1:]
            for listener in tuple(self._kv_event_listeners):
                listener(int(rank), str(kind), payload)
            return True
        return False

    def subscribe_telemetry(self, *, kv_event_listener=None) -> None:
        if kv_event_listener is not None:
            self._kv_event_listeners.append(kv_event_listener)
        self._send_signal_to_engines(Headers.SUBSCRIBE_TELEMETRY)


class DynamoCoordinatorTelemetryMixin:
    """Aggregate status and forward process-tagged telemetry to Dynamo."""

    def _initialize_dynamo_telemetry(self) -> None:
        self._latest_engine_status: dict[bytes, dict] = {}
        self._telemetry_clients: set[bytes] = set()

    def _handle_dynamo_telemetry_subscription(
        self, header: Headers, sender_identity: bytes, known_clients: set
    ) -> bool:
        if header != Headers.SUBSCRIBE_TELEMETRY:
            return False
        if sender_identity in known_clients:
            self._telemetry_clients.add(sender_identity)
        return True

    def _status_payload(self) -> dict:
        statuses = list(self._latest_engine_status.values())
        if statuses:
            authoritative = dict(
                min(statuses, key=lambda status: int(status.get("global_rank", 0)))
            )
            authoritative["rank_statuses"] = statuses
            return authoritative
        return {
            "state": self.state.name.lower(),
            "active_request_count": int(self._pending_counts.sum()),
            "waiting_request_count": 0,
            "pinned_handoff_count": 0,
            "rank_statuses": [],
        }

    def _forward_to_telemetry_clients(self, payload: bytes) -> None:
        for client in list(self._telemetry_clients):
            try:
                self.router_socket.send_multipart([client, payload])
            except zmq.error.ZMQError as exc:
                if exc.errno == zmq.EHOSTUNREACH:
                    self._telemetry_clients.discard(client)
                    continue
                raise

    def _observe_engine_status(self, identity: bytes, status: dict) -> None:
        self._latest_engine_status[identity] = dict(status)
        state = str(status.get("state", "")).lower()
        for (client, operation_id), (target, pending) in list(
            self._pending_controls.items()
        ):
            if state == target:
                pending.discard(identity)
            if not pending:
                self._send_dynamo_control_ack(client, operation_id)
                del self._pending_controls[(client, operation_id)]

    def _handle_dynamo_engine_telemetry(
        self, header: Headers, sender_identity: bytes, payload: list
    ) -> bool:
        if header not in (Headers.KV_EVENT, Headers.ENGINE_STATUS):
            return False
        assert sender_identity in self.identities_of_data_parallel_ranks
        if header == Headers.KV_EVENT:
            rank = self.identity_to_rank_index[sender_identity]
            kind, event_payload = payload[1:]
            packed = msgpack.packb(
                [Headers.KV_EVENT.value, rank, kind, event_payload], use_bin_type=True
            )
            self._forward_to_telemetry_clients(packed)
            return True
        if header == Headers.ENGINE_STATUS:
            self._observe_engine_status(sender_identity, payload[1])
            return True
        return True


def authoritative_kv_event(rank: int, kind: str, payload: dict):
    """Return a normalized KV event only for the authoritative rank."""

    if rank != 0:
        return None
    event = dict(payload)
    event.pop("source_global_rank", None)
    return str(kind), event


def attach_engine_telemetry(engine) -> None:
    """Connect Megatron engine callbacks to the Dynamo telemetry channel."""
    engine.add_kv_event_listener(engine.publish_kv_event)


async def report_engine_status(engine) -> None:
    """Publish status periodically until the owned engine loop exits."""

    while not engine.engine_loop_task.done():
        engine.publish_engine_status()
        await asyncio.sleep(0.1)
