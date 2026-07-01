# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Disaggregated Megatron functional test.

Runs against an already-up Dynamo+Megatron disaggregated stack. Reads
``/tmp/phase3.env`` or matching environment variables.

Checks the model endpoint, streaming completion, prefill handoff marker, and
decode-side KV import marker.

Run inside the container:

    pytest -q /opt/megatron-lm/examples/inference/dynamo/phase3/test_phase3.py

or pointed at a remote stack:

    PHASE3_FRONTEND_URL=http://frontend:8080 \
        PHASE3_MODEL_NAME=llama-3.1-8b-instruct \
        PHASE3_PREFILL_LOG=/path/to/coordinator-prefill.log \
        PHASE3_DECODE_LOG=/path/to/coordinator-decode.log \
        pytest -q test_phase3.py
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import pytest
import requests


REQUIRED_VARS = (
    "PHASE3_FRONTEND_URL",
    "PHASE3_MODEL_NAME",
    "PHASE3_PREFILL_LOG",
    "PHASE3_DECODE_LOG",
)


def _load_phase3_env() -> dict[str, str]:
    env: dict[str, str] = {}
    p = Path("/tmp/phase3.env")
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :]
            key, _, val = line.partition("=")
            env[key.strip()] = val.strip().strip('"').strip("'")
    for k in REQUIRED_VARS:
        if k in os.environ:
            env[k] = os.environ[k]
    return env


@pytest.fixture(scope="module")
def stack() -> dict[str, str]:
    env = _load_phase3_env()
    missing = [k for k in REQUIRED_VARS if k not in env]
    if missing:
        pytest.skip(
            "Disaggregated stack not detected. Start it with orchestrate.sh, or set "
            f"{', '.join(REQUIRED_VARS)} in the environment. Missing: {missing}"
        )
    return env


def test_models_endpoint_lists_served_model(stack):
    r = requests.get(f"{stack['PHASE3_FRONTEND_URL']}/v1/models", timeout=10)
    r.raise_for_status()
    body = r.json()
    ids = {m.get("id") for m in body.get("data", [])}
    assert stack["PHASE3_MODEL_NAME"] in ids, f"served model missing from {ids}"


def _iter_stream(response) -> list[dict]:
    chunks: list[dict] = []
    for raw in response.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data:"):
            continue
        payload = raw[len("data:") :].strip()
        if payload == "[DONE]":
            break
        chunks.append(json.loads(payload))
    return chunks


def _send_completion(stack, prompt: str, max_tokens: int = 32) -> list[dict]:
    body = {
        "model": stack["PHASE3_MODEL_NAME"],
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
    }
    with requests.post(
        f"{stack['PHASE3_FRONTEND_URL']}/v1/chat/completions",
        json=body,
        stream=True,
        timeout=180,
    ) as r:
        r.raise_for_status()
        return _iter_stream(r)


def test_streaming_completion_through_disagg_pipeline(stack):
    """Use a prompt long enough to span multiple KV blocks."""
    long_prompt = " ".join(["Hello"] * 200)  # ~200 tokens; enough to cross 64-token blocks
    chunks = _send_completion(stack, prompt=f"{long_prompt} Summarize.", max_tokens=32)
    assert chunks, "no SSE chunks received"

    delta_chunks = [
        c for c in chunks if c.get("choices", [{}])[0].get("delta", {}).get("content")
    ]
    assert delta_chunks, f"no content deltas in stream: {chunks!r}"

    finish_reasons = [
        c["choices"][0].get("finish_reason") for c in chunks if c.get("choices")
    ]
    assert any(f for f in finish_reasons), f"no finish_reason: {chunks!r}"


_PREFILL_MARKER = re.compile(r"DISAGG_PREFILL_HANDOFF\s+request_id=(\d+)\s+pinned_blocks=(\d+)")
_DECODE_MARKER = re.compile(
    r"DISAGG_DECODE_IMPORT\s+request_id=(\d+)\s+prompt_tokens=(\d+)\s+imported_blocks=(\d+)\s+hashes_registered=(\d+)"
)
def _read_log_with_retry(path: str, max_seconds: int = 20) -> str:
    """Engines flush logs asynchronously. Re-read for up to `max_seconds`."""
    deadline = time.time() + max_seconds
    last = ""
    while time.time() < deadline:
        try:
            last = Path(path).read_text()
        except OSError:
            pass
        if last:
            return last
        time.sleep(1)
    return last


def test_prefill_engine_pinned_blocks_for_handoff(stack):
    """One full request must have driven the prefill engine to pin KV blocks."""
    # Drive a request first so the log has something to assert on.
    _send_completion(stack, prompt="Prefill marker probe.", max_tokens=8)
    log = _read_log_with_retry(stack["PHASE3_PREFILL_LOG"])
    matches = _PREFILL_MARKER.findall(log)
    assert matches, (
        "no DISAGG_PREFILL_HANDOFF markers in prefill engine log; prefill "
        "side did not pin KV blocks for a request. Check that "
        "sampling_params.do_kv_handoff was set by the prefill worker handler."
    )
    assert any(int(blk) > 0 for _, blk in matches), (
        f"prefill saw the disagg path but pinned 0 blocks: {matches}"
    )


def test_decode_engine_imported_kv_and_skipped_prefill(stack):
    """The decode engine must have taken the NIXL-import + prefix-cache match path.

    This proves the decode side used ``add_request_with_kv_handoff`` rather than
    regular full-prompt admission.
    """
    _send_completion(stack, prompt="Decode marker probe.", max_tokens=8)
    log = _read_log_with_retry(stack["PHASE3_DECODE_LOG"])
    matches = _DECODE_MARKER.findall(log)
    assert matches, (
        "no DISAGG_DECODE_IMPORT markers in decode engine log; decode side "
        "did not take the kv-handoff import path. Either the frontend routed "
        "the request as aggregated (PrefillRouter not active?), or the "
        "decode worker's --role flag isn't `decode`."
    )
    assert any(int(reg) > 0 for _, _, _, reg in matches), (
        f"decode imported blocks but registered 0 hashes; prefix-cache "
        f"match path can't skip prefill in this case: {matches}"
    )
