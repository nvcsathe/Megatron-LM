# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase-0 functional test.

Runs against an already-up Dynamo+Megatron stack started by launch_phase0.sh.
Reads `/tmp/phase0.env` (or env vars `PHASE0_FRONTEND_URL` / `PHASE0_MODEL_NAME`)
to find the frontend, then exercises three scenarios:

1. ``/v1/models`` advertises the served model.
2. A streaming chat completion produces at least one delta and terminates
   with a ``finish_reason``.
3. The streamed token count respects the requested ``max_tokens`` ceiling.

Run inside the container:

    pytest -q /opt/megatron-lm/examples/inference/dynamo/phase0/test_phase0.py

or pointed at a remote stack:

    PHASE0_FRONTEND_URL=http://frontend:8080 PHASE0_MODEL_NAME=llama-3.1-8b \
        pytest -q test_phase0.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import requests


def _load_phase0_env() -> dict[str, str]:
    """Source /tmp/phase0.env if present; fall back to environment variables."""
    env: dict[str, str] = {}
    p = Path("/tmp/phase0.env")
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :]
            key, _, val = line.partition("=")
            env[key.strip()] = val.strip().strip('"').strip("'")
    for k in ("PHASE0_FRONTEND_URL", "PHASE0_MODEL_NAME"):
        if k in os.environ:
            env[k] = os.environ[k]
    return env


@pytest.fixture(scope="module")
def stack() -> dict[str, str]:
    env = _load_phase0_env()
    if "PHASE0_FRONTEND_URL" not in env or "PHASE0_MODEL_NAME" not in env:
        pytest.skip(
            "Phase-0 stack not detected. Start it with launch_phase0.sh, or set "
            "PHASE0_FRONTEND_URL + PHASE0_MODEL_NAME in the environment."
        )
    return env


def test_models_endpoint_lists_served_model(stack):
    r = requests.get(f"{stack['PHASE0_FRONTEND_URL']}/v1/models", timeout=10)
    r.raise_for_status()
    body = r.json()
    ids = {m.get("id") for m in body.get("data", [])}
    assert stack["PHASE0_MODEL_NAME"] in ids, f"served model missing from {ids}"


def _iter_stream(response) -> list[dict]:
    """Parse an OpenAI-style SSE response into a list of decoded JSON chunks."""
    chunks: list[dict] = []
    for raw in response.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data:"):
            continue
        payload = raw[len("data:") :].strip()
        if payload == "[DONE]":
            break
        chunks.append(json.loads(payload))
    return chunks


def test_streaming_completion_emits_deltas_and_finishes(stack):
    body = {
        "model": stack["PHASE0_MODEL_NAME"],
        "messages": [
            {"role": "user", "content": "Say hi in one short sentence."}
        ],
        "stream": True,
        "max_tokens": 32,
    }
    with requests.post(
        f"{stack['PHASE0_FRONTEND_URL']}/v1/chat/completions",
        json=body,
        stream=True,
        timeout=120,
    ) as r:
        r.raise_for_status()
        chunks = _iter_stream(r)

    assert chunks, "no SSE chunks received"

    delta_chunks = [
        c for c in chunks if c.get("choices", [{}])[0].get("delta", {}).get("content")
    ]
    assert len(delta_chunks) >= 1, f"expected at least one content delta, got {chunks}"

    finish_reasons = [
        c["choices"][0].get("finish_reason")
        for c in chunks
        if c.get("choices")
    ]
    assert any(f for f in finish_reasons), f"no finish_reason in any chunk: {chunks}"


def test_streaming_respects_max_tokens_ceiling(stack):
    body = {
        "model": stack["PHASE0_MODEL_NAME"],
        "messages": [{"role": "user", "content": "Count to 100."}],
        "stream": True,
        "max_tokens": 16,
    }
    with requests.post(
        f"{stack['PHASE0_FRONTEND_URL']}/v1/chat/completions",
        json=body,
        stream=True,
        timeout=120,
    ) as r:
        r.raise_for_status()
        chunks = _iter_stream(r)

    text = "".join(
        c["choices"][0].get("delta", {}).get("content", "")
        for c in chunks
        if c.get("choices")
    )
    word_count = len(text.split())
    assert word_count <= 25, f"output exceeded max_tokens ceiling: {word_count} words: {text!r}"
