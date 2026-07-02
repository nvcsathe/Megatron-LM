# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
from unittest.mock import MagicMock, patch

import msgpack
import pytest
import zmq

from megatron.core.inference.headers import Headers
from megatron.inference.integrations.dynamo.client import DynamoInferenceClient

pytestmark = pytest.mark.asyncio


def _make_client():
    fake_socket = MagicMock(name="zmq_socket")
    fake_context = MagicMock(name="zmq_context")
    fake_context.socket.return_value = fake_socket
    with patch("megatron.core.inference.inference_client.zmq.Context", return_value=fake_context):
        client = DynamoInferenceClient("tcp://127.0.0.1:5555", deserialize=False)
    return client, fake_socket


async def test_metadata_status_and_kv_events():
    client, fake_socket = _make_client()
    assert callable(client.get_status)
    assert callable(client.subscribe_telemetry)
    metadata = {"protocol_version": 1, "context_length": 8192, "logical_data_parallel_size": 1}
    recv_queue = [msgpack.packb([Headers.CONNECT_ACK.value, metadata], use_bin_type=True)]
    status_reply = msgpack.packb(
        [Headers.STATUS_REPLY.value, 0, {"state": "running", "active_request_count": 2}],
        use_bin_type=True,
    )
    kv_reply = msgpack.packb(
        [Headers.KV_EVENT.value, 0, "stored", {"block_hashes": [11]}], use_bin_type=True
    )

    def fake_recv(*args, **kwargs):
        if recv_queue:
            return recv_queue.pop(0)
        raise zmq.Again()

    fake_socket.recv.side_effect = fake_recv
    client.start()
    assert client.metadata == metadata

    kv_events = []
    client.subscribe_telemetry(
        kv_event_listener=lambda rank, kind, payload: kv_events.append((rank, kind, payload))
    )
    status_task = asyncio.create_task(client.get_status())
    await asyncio.sleep(0)
    recv_queue.extend([status_reply, kv_reply])
    status = await asyncio.wait_for(status_task, timeout=2)
    await asyncio.sleep(0.02)

    assert status == {"state": "running", "active_request_count": 2}
    assert kv_events == [(0, "stored", {"block_hashes": [11]})]
    client.stop()


async def test_stop_cancels_pending_management_request():
    client, fake_socket = _make_client()
    recv_queue = [msgpack.packb([Headers.CONNECT_ACK.value, {}], use_bin_type=True)]

    def fake_recv(*args, **kwargs):
        if recv_queue:
            return recv_queue.pop(0)
        raise zmq.Again()

    fake_socket.recv.side_effect = fake_recv
    client.start()

    status_task = asyncio.create_task(client.get_status())
    await asyncio.sleep(0)
    client.stop()
    await asyncio.sleep(0)

    assert status_task.cancelled()
