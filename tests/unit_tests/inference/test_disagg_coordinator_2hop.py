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
from megatron.core.inference.disaggregation.coordinator_routing import DisaggRouting
from megatron.core.inference.headers import Headers

HANDOFF = {"kv_meta": {"agent": "p0"}, "block_ids": [4, 5], "request_id": 5}


def _coord(max_outstanding=32):
    """Coordinator with just the disagg state populated (bypass ZMQ __init__)."""
    c = DataParallelInferenceCoordinator.__new__(DataParallelInferenceCoordinator)
    c.disaggregated = True
    c._disagg = DisaggRouting()
    c._req_meta = {}
    c._disagg_hop1 = set()
    c._disagg_prefill_of = {}
    c._disagg_outstanding = {}
    c._disagg_submit_queue = {}
    c._disagg_max_outstanding = max_outstanding
    c.request_id_to_client_id = {}
    c.request_id_to_client_request_id = {}
    c.client_request_to_request_id = {}
    c.identities_of_data_parallel_ranks = [b"p0", b"d0"]
    c._engine_transport = {b"p0": "nixl", b"d0": "nixl"}
    c._engine_metas = {b"p0": [{"global_rank": 0}], b"d0": [{"global_rank": 2}]}
    c._disagg.register(b"p0", "prefill")
    c._disagg.register(b"d0", "decode")
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
    assert c._disagg_outstanding[b"p0"] == 0


def test_kv_read_done_releases_prefill_and_drains_queue():
    c = _coord(max_outstanding=1)
    c._route_submit_disagg(5, [1], {})
    c._route_submit_disagg(6, [2], {})  # window full: queued
    assert len(c._disagg_submit_queue[b"p0"]) == 1
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
    # The prefill is gone from the routing pool.
    with pytest.raises(RuntimeError):
        c._disagg.route_submit(7)
