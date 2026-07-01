#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark async NIXL KV pulls through the Dynamo frontend.

Runs inside the Dynamo container against an already-started disaggregated stack.
It reads /tmp/phase3.env unless the same values are supplied in the environment.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests


REQUIRED_VARS = (
    "PHASE3_FRONTEND_URL",
    "PHASE3_MODEL_NAME",
    "PHASE3_DECODE_LOG",
)

PULL_SUBMIT_RE = re.compile(
    r"DISAGG_DECODE_PULL_SUBMIT\s+request_id=(\d+)\s+prompt_tokens=(\d+)\s+blocks=(\d+)\s+pending_imports=(\d+)"
)
IMPORT_RE = re.compile(
    r"DISAGG_DECODE_IMPORT\s+request_id=(\d+)\s+prompt_tokens=(\d+)\s+imported_blocks=(\d+)\s+hashes_registered=(\d+)"
)


def _load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    env_path = Path(os.environ.get("PHASE3_ENV_FILE", "/tmp/phase3.env"))
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :]
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip().strip('"').strip("'")
    for key in REQUIRED_VARS:
        if key in os.environ:
            env[key] = os.environ[key]
    missing = [key for key in REQUIRED_VARS if key not in env]
    if missing:
        raise RuntimeError(f"Missing required environment values: {missing}")
    return env


def _parse_bursts(value: str) -> list[int]:
    bursts = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not bursts or any(count <= 0 for count in bursts):
        raise ValueError("--bursts must contain positive integers")
    return bursts


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def _build_prompt(index: int, prompt_words: int) -> str:
    token = f"async-pull-benchmark-{index}"
    body = " ".join([token] * prompt_words)
    return f"{body}\nReturn one concise sentence."


def _stream_request(
    *,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
    index: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
    }
    started = time.perf_counter()
    first_chunk = None
    chunks = 0
    output_chars = 0
    finish_reason = None
    error = None
    status_code = None
    pending_payload = ""
    try:
        with requests.post(
            f"{url}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=timeout,
        ) as response:
            status_code = response.status_code
            if response.status_code >= 400:
                body = response.text[:1000]
                error = f"HTTP {response.status_code}: {body}"
            else:
                for raw in response.iter_lines(decode_unicode=True):
                    if not raw:
                        continue
                    if not raw.startswith("data:"):
                        if pending_payload:
                            pending_payload += raw.strip()
                        continue
                    if pending_payload:
                        pending_payload += raw[len("data:") :].strip()
                    else:
                        pending_payload = raw[len("data:") :].strip()
                    if pending_payload == "[DONE]":
                        pending_payload = ""
                        break
                    done_after_json = False
                    if pending_payload.endswith("[DONE]"):
                        pending_payload = pending_payload[: -len("[DONE]")].strip()
                        done_after_json = True
                    try:
                        chunk = json.loads(pending_payload)
                    except json.JSONDecodeError:
                        continue
                    pending_payload = ""
                    if first_chunk is None:
                        first_chunk = time.perf_counter()
                    chunks += 1
                    choice = chunk.get("choices", [{}])[0]
                    finish_reason = choice.get("finish_reason") or finish_reason
                    output_chars += len(choice.get("delta", {}).get("content") or "")
                    if done_after_json:
                        break
                if pending_payload:
                    error = f"truncated SSE JSON: {pending_payload[:500]}"
                if chunks == 0:
                    error = error or "stream completed without chunks"
    except Exception as exc:  # noqa: BLE001 - report benchmark failures in summary
        error = repr(exc)
    finished = time.perf_counter()
    return {
        "index": index,
        "ok": error is None,
        "error": error,
        "status_code": status_code,
        "ttft_s": (first_chunk - started) if first_chunk is not None else None,
        "latency_s": finished - started,
        "chunks": chunks,
        "output_chars": output_chars,
        "finish_reason": finish_reason,
    }


def _read_decode_markers(path: str) -> dict[str, Any]:
    try:
        text = Path(path).read_text()
    except OSError:
        text = ""
    submits = PULL_SUBMIT_RE.findall(text)
    imports = IMPORT_RE.findall(text)
    pending_depths = [int(row[3]) for row in submits]
    return {
        "pull_submit_count": len(submits),
        "import_count": len(imports),
        "max_pending_imports": max(pending_depths) if pending_depths else 0,
        "submitted_blocks": sum(int(row[2]) for row in submits),
        "imported_blocks": sum(int(row[2]) for row in imports),
        "registered_hashes": sum(int(row[3]) for row in imports),
    }


def _check_model_ready(url: str, model: str) -> None:
    response = requests.get(f"{url}/v1/models", timeout=30)
    response.raise_for_status()
    body = response.json()
    model_ids = {item.get("id") for item in body.get("data", [])}
    if model not in model_ids:
        raise RuntimeError(f"model {model!r} not listed by /v1/models: {sorted(model_ids)}")


def _tail(path: str, lines: int = 80) -> list[str]:
    try:
        content = Path(path).read_text(errors="replace").splitlines()
    except OSError:
        return []
    return content[-lines:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bursts", default=os.environ.get("PHASE3_BENCH_BURSTS", "2,4,3"))
    parser.add_argument(
        "--burst-gaps",
        default=os.environ.get("PHASE3_BENCH_BURST_GAPS", "1,3"),
        help="Comma-separated seconds between bursts.",
    )
    parser.add_argument(
        "--prompt-words",
        type=int,
        default=int(os.environ.get("PHASE3_BENCH_PROMPT_WORDS", "240")),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("PHASE3_BENCH_MAX_TOKENS", "16")),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("PHASE3_BENCH_TIMEOUT", "300")),
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=int(os.environ.get("PHASE3_BENCH_WARMUP", "1")),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=int(os.environ.get("PHASE3_BENCH_MAX_WORKERS", "0")),
        help="Thread pool size. Default is the largest burst.",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        default=os.environ.get("PHASE3_BENCH_ALLOW_FAILURES", "0") == "1",
    )
    parser.add_argument("--output", default=os.environ.get("PHASE3_BENCH_OUTPUT", ""))
    args = parser.parse_args()

    env = _load_env()
    bursts = _parse_bursts(args.bursts)
    gaps = [float(part.strip()) for part in args.burst_gaps.split(",") if part.strip()]
    if len(gaps) < max(0, len(bursts) - 1):
        gaps.extend([0.0] * (len(bursts) - 1 - len(gaps)))

    url = env["PHASE3_FRONTEND_URL"]
    model = env["PHASE3_MODEL_NAME"]
    decode_log = env["PHASE3_DECODE_LOG"]

    _check_model_ready(url, model)

    warmup_failures = []
    for warmup_idx in range(args.warmup):
        warmup_result = _stream_request(
            url=url,
            model=model,
            prompt=_build_prompt(-(warmup_idx + 1), args.prompt_words),
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            index=-(warmup_idx + 1),
        )
        if not warmup_result["ok"]:
            warmup_failures.append(warmup_result)
    if warmup_failures:
        print("[bench] warmup failed")
        print(json.dumps(warmup_failures, indent=2, sort_keys=True))
        return 10

    total_requests = sum(bursts)
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    max_workers = args.max_workers or max(bursts)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        request_index = 0
        for burst_idx, count in enumerate(bursts):
            print(f"[bench] submitting burst {burst_idx + 1}/{len(bursts)}: {count} requests")
            for _ in range(count):
                futures.append(
                    pool.submit(
                        _stream_request,
                        url=url,
                        model=model,
                        prompt=_build_prompt(request_index, args.prompt_words),
                        max_tokens=args.max_tokens,
                        timeout=args.timeout,
                        index=request_index,
                    )
                )
                request_index += 1
            if burst_idx < len(bursts) - 1 and gaps[burst_idx] > 0:
                time.sleep(gaps[burst_idx])

        for future in as_completed(futures, timeout=args.timeout * max(1, total_requests)):
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - keep benchmark alive
                result = {
                    "index": -1,
                    "ok": False,
                    "error": repr(exc),
                    "status_code": None,
                    "ttft_s": None,
                    "latency_s": 0.0,
                    "chunks": 0,
                    "output_chars": 0,
                    "finish_reason": None,
                }
            results.append(result)
            if result["ok"]:
                print(
                    "[bench] request {index} done latency={latency_s:.3f}s ttft={ttft_s:.3f}s chunks={chunks}".format(
                        **result
                    )
                )
            else:
                print(
                    f"[bench] request {result['index']} failed "
                    f"latency={result['latency_s']:.3f}s error={result['error']}"
                )

    elapsed = time.perf_counter() - started
    successes = [row for row in results if row["ok"]]
    failures = [row for row in results if not row["ok"]]
    latencies = [float(row["latency_s"]) for row in successes]
    ttfts = [float(row["ttft_s"]) for row in successes if row["ttft_s"] is not None]
    markers = _read_decode_markers(decode_log)
    log_dir = str(Path(decode_log).parent)
    summary = {
        "requests": len(results),
        "succeeded": len(successes),
        "failed": len(failures),
        "bursts": bursts,
        "burst_gaps_s": gaps[: max(0, len(bursts) - 1)],
        "max_workers": max_workers,
        "prompt_words": args.prompt_words,
        "max_tokens": args.max_tokens,
        "elapsed_s": elapsed,
        "request_rate": len(successes) / elapsed if elapsed > 0 else 0.0,
        "latency_s": {
            "avg": statistics.mean(latencies) if latencies else 0.0,
            "p50": _percentile(latencies, 50),
            "p90": _percentile(latencies, 90),
            "p99": _percentile(latencies, 99),
            "max": max(latencies) if latencies else 0.0,
        },
        "ttft_s": {
            "avg": statistics.mean(ttfts) if ttfts else 0.0,
            "p50": _percentile(ttfts, 50),
            "p90": _percentile(ttfts, 90),
            "p99": _percentile(ttfts, 99),
            "max": max(ttfts) if ttfts else 0.0,
        },
        "decode_log": markers,
        "failures": failures[:20],
    }

    print("[bench] summary")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    if failures and not args.allow_failures:
        print("[bench] ERROR: one or more requests failed", file=sys.stderr)
        for name in (
            "frontend.log",
            "worker-prefill.log",
            "worker-decode.log",
            "coordinator-prefill.log",
            "coordinator-decode.log",
        ):
            path = str(Path(log_dir) / name)
            tail = _tail(path, lines=60)
            if tail:
                print(f"\n[bench] tail {path}", file=sys.stderr)
                print("\n".join(tail), file=sys.stderr)
        return 20
    if markers["pull_submit_count"] < len(successes):
        print("[bench] ERROR: fewer pull submissions than benchmark requests", file=sys.stderr)
        return 2
    if markers["import_count"] < len(successes):
        print("[bench] ERROR: fewer completed imports than benchmark requests", file=sys.stderr)
        return 3
    if markers["max_pending_imports"] < 2 and len(successes) > 1:
        print("[bench] ERROR: async imports did not overlap", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
