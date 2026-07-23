# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Coordinator-side 2-hop handlers (CPU; fake socket, no ZMQ/engines).

Drives the disaggregated routing methods on a coordinator built without
__init__ (which needs ZMQ) and asserts the right control messages go to the
right engines: SUBMIT (with do_kv_handoff) to the prefill, then on the
prefill's reply a SUBMIT_REQUEST_WITH_KV to the decode carrying the hand-off
metadata, and RELEASE_KV back to the prefill on KV_READ_DONE.
"""

import pytest

msgpack = pytest.importorskip("msgpack")

from megatron.core.inference.data_parallel_inference_coordinator import (
    DataParallelInferenceCoordinator,
)
from megatron.core.inference.disaggregation.coordinator_flow_control import DisaggMambaFlowControl
from megatron.core.inference.disaggregation.coordinator_routing import DisaggRouting
from megatron.core.inference.disaggregation.handoff_wire_protocol import (
    restore_registered_nixl_agent_metadata,
    strip_registered_nixl_agent_metadata,
)
from megatron.core.inference.headers import Headers

HANDOFF = {"kv_meta": {"agent": "p0"}, "block_ids": [4, 5], "request_id": 5}


def _coord(max_outstanding=32, mamba_capacity=None):
    """Coordinator with just the disagg state populated (bypass ZMQ __init__)."""
    c = DataParallelInferenceCoordinator.__new__(DataParallelInferenceCoordinator)
    c.disaggregated = True
    c._disagg = DisaggRouting()
    c._req_meta = {}
    c._disagg_hop1 = set()
    c._disagg_prefill_of = {}
    c._disagg_max_outstanding = max_outstanding
    c._disagg_mamba_flow = DisaggMambaFlowControl()
    c.block_size_tokens = 256
    c.request_id_to_client_id = {}
    c.request_id_to_client_request_id = {}
    c.client_request_to_request_id = {}
    c.identities_of_data_parallel_ranks = [b"p0", b"d0"]
    c._engine_transport = {b"p0": "nixl", b"d0": "nixl"}
    prefill_meta = {"global_rank": 0}
    decode_meta = {"global_rank": 2}
    if mamba_capacity is not None:
        for meta in (prefill_meta, decode_meta):
            meta["mamba_slot_capacity"] = mamba_capacity
    c._engine_metas = {b"p0": [prefill_meta], b"d0": [decode_meta]}
    c._disagg.register(b"p0", "prefill")
    c._disagg.register(b"d0", "decode")
    c._disagg_mamba_flow.register_engine(b"p0", "prefill", c._engine_metas[b"p0"])
    c._disagg_mamba_flow.register_engine(b"d0", "decode", c._engine_metas[b"d0"])
    c.sent = []  # (identity, [header, *parts])
    c._send_to_engine = lambda ident, payload: (
        c.sent.append((ident, msgpack.unpackb(payload, raw=False))) or True
    )
    return c


def test_submit_routes_to_prefill_with_handoff_params():
    c = _coord()
    c._route_submit_disagg(5, [1, 2, 3], {"temperature": 0.0, "num_tokens_to_generate": 64})
    assert c._req_meta[5] == ([1, 2, 3], {"temperature": 0.0, "num_tokens_to_generate": 64})
    assert 5 in c._disagg_hop1
    ident, msg = c.sent[0]
    assert ident == b"p0"
    assert Headers(msg[0]) == Headers.SUBMIT_REQUEST and msg[1] == 5
    # The prefill copy stops after the prompt KV and pins it for the hand-off.
    assert msg[3]["do_kv_handoff"] is True
    assert msg[3]["num_tokens_to_generate"] == 0


def test_prefill_reply_resubmits_to_decode_with_kv():
    c = _coord()
    c._route_submit_disagg(5, [1, 2, 3], {"temperature": 0.0})
    c.sent.clear()
    c._handle_prefill_done(5, {"request_id": 5, "disaggregated_params": HANDOFF})
    assert 5 not in c._disagg_hop1
    ident, msg = c.sent[0]
    assert ident == b"d0"
    assert Headers(msg[0]) == Headers.SUBMIT_REQUEST_WITH_KV
    # [header, request_id, prompt, original sampling params, kv_meta, src blocks]
    assert msg[1] == 5 and msg[2] == [1, 2, 3]
    assert msg[3] == {"temperature": 0.0}
    assert msg[4] == HANDOFF["kv_meta"] and msg[5] == [4, 5]


def test_nixl_handoff_restores_registered_agent_metadata_across_tp_and_pp():
    registered = [
        {
            "agent_name": "prefill-rank0",
            "agent_metadata_b64": "a3Yw",
            "mamba": {
                "conv": {
                    "agent_name": "prefill-conv-rank0",
                    "agent_metadata_b64": "Y29udjA=",
                },
                "ssm": {
                    "agent_name": "prefill-ssm-rank0",
                    "agent_metadata_b64": "c3NtMA==",
                },
            },
        },
        {
            "agent_name": "prefill-rank1",
            "agent_metadata_b64": "a3Yx",
            "mamba": {
                "conv": {
                    "agent_name": "prefill-conv-rank1",
                    "agent_metadata_b64": "Y29udjE=",
                },
                "ssm": {
                    "agent_name": "prefill-ssm-rank1",
                    "agent_metadata_b64": "c3NtMQ==",
                },
            },
        },
    ]
    handoff = {
        "pp_metas": [
            {
                "tp_metas": [
                    {
                        "agent_name": registered[0]["agent_name"],
                        "agent_metadata_b64": registered[0]["agent_metadata_b64"],
                        "block_ids": [7, 8],
                    },
                    {
                        "agent_name": registered[1]["agent_name"],
                        "agent_metadata_b64": registered[1]["agent_metadata_b64"],
                        "block_ids": [9, 10],
                    },
                ],
                "block_ids": [7, 8],
            }
        ],
        "mamba": {
            "positions": [0, 2],
            "conv": {
                "pp_metas": [
                    {
                        "tp_metas": [
                            {**registered[0]["mamba"]["conv"], "block_ids": [3, 4]},
                            {**registered[1]["mamba"]["conv"], "block_ids": [5, 6]},
                        ]
                    }
                ]
            },
            "ssm": {
                "pp_metas": [
                    {
                        "tp_metas": [
                            {**registered[0]["mamba"]["ssm"], "block_ids": [3, 4]},
                            {**registered[1]["mamba"]["ssm"], "block_ids": [5, 6]},
                        ]
                    }
                ]
            },
        },
    }

    compact = strip_registered_nixl_agent_metadata(handoff)
    assert "agent_metadata_b64" not in repr(compact)
    restored = restore_registered_nixl_agent_metadata(compact, registered)

    assert restored["pp_metas"][0]["tp_metas"][0]["block_ids"] == [7, 8]
    assert restored["mamba"]["positions"] == [0, 2]
    assert (
        restored["mamba"]["ssm"]["pp_metas"][0]["tp_metas"][1]["agent_metadata_b64"]
        == "c3NtMQ=="
    )


def test_nixl_handoff_rejects_unregistered_or_conflicting_agent_metadata():
    registered = [{"agent_name": "prefill-rank0", "agent_metadata_b64": "registered"}]

    with pytest.raises(ValueError, match="absent from instance registration"):
        restore_registered_nixl_agent_metadata(
            {"agent_name": "unknown-rank", "block_ids": [1]}, registered
        )
    with pytest.raises(ValueError, match="differs from its registration"):
        restore_registered_nixl_agent_metadata(
            {
                "agent_name": "prefill-rank0",
                "agent_metadata_b64": "different",
                "block_ids": [1],
            },
            registered,
        )


def test_nixl_handoff_only_strips_metadata_from_agent_records():
    handoff = {
        "agent_name": "prefill-rank0",
        "agent_metadata_b64": "registered",
        "application_payload": {"agent_metadata_b64": "unrelated"},
    }

    compact = strip_registered_nixl_agent_metadata(handoff)

    assert "agent_metadata_b64" not in compact
    assert compact["application_payload"]["agent_metadata_b64"] == "unrelated"


def test_coordinator_hydrates_compact_nixl_handoff_from_prefill_registration():
    c = _coord()
    registered = {
        "agent_name": "prefill-rank0",
        "agent_metadata_b64": "cHJlZmlsbA==",
        "global_rank": 0,
    }
    c._engine_metas[b"p0"] = [registered]
    c._route_submit_disagg(5, [1, 2, 3], {"temperature": 0.0})
    c.sent.clear()
    handoff = {
        "kv_meta": {
            "agent_name": "prefill-rank0",
            "global_rank": 0,
            "block_ids": [4, 5],
        },
        "block_ids": [4, 5],
    }

    c._handle_prefill_done(5, {"request_id": 5, "disaggregated_params": handoff})

    _, message = c.sent[0]
    assert message[4]["agent_metadata_b64"] == "cHJlZmlsbA=="
    assert message[4]["block_ids"] == [4, 5]


def test_push_prefill_reply_also_emits_send_kv():
    c = _coord()
    c._engine_transport[b"p0"] = "nccl"
    c._route_submit_disagg(5, [1, 2, 3], {})
    c.sent.clear()
    c._handle_prefill_done(5, {"request_id": 5, "disaggregated_params": HANDOFF})
    headers = [(ident, Headers(m[0])) for ident, m in c.sent]
    assert (b"d0", Headers.SUBMIT_REQUEST_WITH_KV) in headers
    # Two-sided transport: the prefill is told where to send, with the decode
    # instance's transfer metas.
    send = [m for ident, m in c.sent if Headers(m[0]) == Headers.SEND_KV]
    assert send and send[0][1] == 5 and send[0][2] == [{"global_rank": 2}]


def test_prefill_reply_without_handoff_drops_request():
    c = _coord()
    c._route_submit_disagg(5, [1, 2, 3], {})
    c.sent.clear()
    c._handle_prefill_done(5, {"request_id": 5})
    # No decode submit; the request's state is cleared and the flow-control
    # slot returned.
    assert not [m for _, m in c.sent if Headers(m[0]) == Headers.SUBMIT_REQUEST_WITH_KV]
    assert 5 not in c._req_meta and 5 not in c._disagg_prefill_of
    assert c._disagg_mamba_flow.prefill_usage(b"p0") == 0


def test_kv_read_done_releases_prefill_and_drains_queue():
    c = _coord(max_outstanding=1)
    c._route_submit_disagg(5, [1], {})
    c._route_submit_disagg(6, [2], {})  # window full: queued
    assert c._disagg_mamba_flow.has_queued_prefill(b"p0")
    c.sent.clear()
    c._handle_kv_read_done(5)
    headers = [(ident, Headers(m[0])) for ident, m in c.sent]
    assert (b"p0", Headers.RELEASE_KV) in headers
    # The queued request was submitted once the slot freed.
    assert (b"p0", Headers.SUBMIT_REQUEST) in headers
    assert c._disagg_prefill_of[6] == b"p0"
    # A duplicate or late ack is a no-op.
    c.sent.clear()
    c._handle_kv_read_done(5)
    assert not c.sent


def test_removed_engine_sweeps_queued_and_inflight_requests():
    c = _coord(max_outstanding=1)
    c._route_submit_disagg(5, [1], {})
    c._route_submit_disagg(6, [2], {})  # queued behind the window
    c._remove_engine(b"p0")
    assert 5 not in c._req_meta and 6 not in c._req_meta
    assert not c._disagg_prefill_of
    assert c._disagg_mamba_flow.prefill_usage(b"p0") == 0
    # The prefill is gone from the routing pool.
    with pytest.raises(RuntimeError):
        c._disagg.route_submit(7)


def test_decode_mamba_capacity_queues_until_generation_finishes():
    c = _coord(mamba_capacity=4)
    handoff = {"kv_meta": {"mamba": {"positions": [0, 1, 2]}}, "block_ids": [4, 5, 6]}
    for request_id in (5, 6):
        c._route_submit_disagg(request_id, [1, 2, 3], {})

    c.sent.clear()
    c._handle_prefill_done(5, {"request_id": 5, "disaggregated_params": handoff})
    c._handle_prefill_done(6, {"request_id": 6, "disaggregated_params": handoff})

    decode_submits = [
        message
        for identity, message in c.sent
        if identity == b"d0" and Headers(message[0]) == Headers.SUBMIT_REQUEST_WITH_KV
    ]
    assert [message[1] for message in decode_submits] == [5]
    assert c._disagg_mamba_flow.decode_usage(b"d0") == 3
    assert c._disagg_mamba_flow.has_queued(b"d0")

    c._release_decode_slot_reservation(5)

    decode_submits = [
        message
        for identity, message in c.sent
        if identity == b"d0" and Headers(message[0]) == Headers.SUBMIT_REQUEST_WITH_KV
    ]
    assert [message[1] for message in decode_submits] == [5, 6]
    assert c._disagg_mamba_flow.decode_usage(b"d0") == 3
    assert not c._disagg_mamba_flow.has_queued(b"d0")


def test_nccl_send_waits_for_decode_mamba_capacity():
    c = _coord(mamba_capacity=2)
    c._engine_transport[b"p0"] = "nccl"
    handoff = {"kv_meta": {"mamba": {"positions": [0, 1]}}, "block_ids": [4, 5]}
    for request_id in (5, 6):
        c._route_submit_disagg(request_id, [1, 2, 3], {})

    c.sent.clear()
    c._handle_prefill_done(5, {"request_id": 5, "disaggregated_params": handoff})
    c._handle_prefill_done(6, {"request_id": 6, "disaggregated_params": handoff})
    sends = [message for _, message in c.sent if Headers(message[0]) == Headers.SEND_KV]
    assert [message[1] for message in sends] == [5]

    c._release_decode_slot_reservation(5)

    sends = [message for _, message in c.sent if Headers(message[0]) == Headers.SEND_KV]
    assert [message[1] for message in sends] == [5, 6]


def test_mamba_capacity_reduces_prefill_outstanding_window():
    c = _coord(max_outstanding=32, mamba_capacity=8)
    prompt = list(range(513))  # Three 256-token blocks per reservation.
    for request_id in (5, 6, 7):
        c._route_submit_disagg(request_id, prompt, {})

    prefill_submits = [
        message
        for identity, message in c.sent
        if identity == b"p0" and Headers(message[0]) == Headers.SUBMIT_REQUEST
    ]
    assert [message[1] for message in prefill_submits] == [5, 6]
    assert c._disagg_mamba_flow.has_queued_prefill(b"p0")
    assert c._disagg_mamba_flow.prefill_usage(b"p0") == 6
