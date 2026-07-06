#!/usr/bin/env python3

"""Exercise every worker in an eight-GPU Nano v3 2P2D deployment."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


MAMBA_BLOCKS_RE = re.compile(r"mamba_blocks=(\d+)")


def completion(url: str, model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
            "nvext": {"extra_fields": ["worker_id", "timing"]},
        }
    ).encode()
    request = urllib.request.Request(
        f"{url}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error

    choices = payload.get("choices") or []
    text = "".join(choice.get("text", "") for choice in choices)
    usage = payload.get("usage") or {}
    worker = (payload.get("nvext") or {}).get("worker_id") or {}
    if not choices or int(usage.get("completion_tokens", 0)) < 1 or not text.strip():
        raise RuntimeError(f"degenerate completion: {payload}")
    if worker.get("prefill_worker_id") is None or worker.get("decode_worker_id") is None:
        raise RuntimeError(f"response did not expose both disaggregated worker IDs: {payload}")
    return {
        "prefill_worker_id": int(worker["prefill_worker_id"]),
        "decode_worker_id": int(worker["decode_worker_id"]),
        "prefill_dp_rank": worker.get("prefill_dp_rank"),
        "decode_dp_rank": worker.get("decode_dp_rank"),
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "elapsed_ms": (time.monotonic() - started) * 1000,
        "text": text,
    }


def load_identities(ready_dir: Path) -> dict[str, set[int]]:
    identities: dict[str, set[int]] = {"prefill": set(), "decode": set()}
    for path in sorted(ready_dir.glob("worker-*")):
        identity = json.loads(path.read_text())
        role = identity.get("role")
        if role not in identities:
            raise RuntimeError(f"unexpected worker role in {path}: {role}")
        identities[role].add(int(identity["worker_id"]))
    return identities


def validate_decode_logs(log_dir: Path, expected: int) -> dict[str, int]:
    logs = sorted(log_dir.glob("node-*-worker-*-decode.log"))
    if len(logs) != expected:
        raise RuntimeError(f"expected {expected} decode logs, found {len(logs)}: {logs}")
    imported: dict[str, int] = {}
    for path in logs:
        text = path.read_text(errors="replace")
        if "DISAGG_DECODE_IMPORT" not in text:
            raise RuntimeError(f"no attention-KV import marker in {path}")
        mamba_lines = [
            line for line in text.splitlines() if "DISAGG_DECODE_MAMBA_IMPORT" in line
        ]
        matches = [
            int(match.group(1))
            for line in mamba_lines
            if (match := MAMBA_BLOCKS_RE.search(line)) is not None
        ]
        blocks = max(matches, default=0)
        if not mamba_lines or blocks < 1:
            raise RuntimeError(f"no committed Mamba state import in {path}")
        imported[path.name] = blocks
    return imported


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-prefill", type=int, default=2)
    parser.add_argument("--expected-decode", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-requests", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--prompt-repeat", type=int, default=1024)
    parser.add_argument("--log-settle-seconds", type=float, default=3)
    args = parser.parse_args()

    if args.concurrency < 1 or args.max_requests < args.concurrency:
        parser.error("max-requests must be at least concurrency, and both must be positive")

    identities = load_identities(args.run_dir / "ready")
    if len(identities["prefill"]) != args.expected_prefill:
        raise RuntimeError(f"prefill identities mismatch: {sorted(identities['prefill'])}")
    if len(identities["decode"]) != args.expected_decode:
        raise RuntimeError(f"decode identities mismatch: {sorted(identities['decode'])}")

    records: list[dict[str, Any]] = []
    observed_prefill: set[int] = set()
    observed_decode: set[int] = set()
    corpus = " evidence" * args.prompt_repeat

    while len(records) < args.max_requests:
        count = min(args.concurrency, args.max_requests - len(records))
        first = len(records)
        prompts = [
            f"Nano 2P2D request {first + index}.{corpus}\nSummarize this evidence in one sentence."
            for index in range(count)
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=count) as executor:
            futures = [
                executor.submit(completion, args.url, args.model, prompt, args.max_tokens)
                for prompt in prompts
            ]
            for future in futures:
                record = future.result()
                records.append(record)
                observed_prefill.add(record["prefill_worker_id"])
                observed_decode.add(record["decode_worker_id"])

        print(
            f"requests={len(records)} "
            f"prefill={sorted(observed_prefill)} decode={sorted(observed_decode)}",
            flush=True,
        )
        if observed_prefill == identities["prefill"] and observed_decode == identities["decode"]:
            break

    if observed_prefill != identities["prefill"]:
        raise RuntimeError(
            f"not every prefill worker served traffic: expected={sorted(identities['prefill'])}, "
            f"observed={sorted(observed_prefill)}"
        )
    if observed_decode != identities["decode"]:
        raise RuntimeError(
            f"not every decode worker served traffic: expected={sorted(identities['decode'])}, "
            f"observed={sorted(observed_decode)}"
        )

    time.sleep(args.log_settle_seconds)
    mamba_imports = validate_decode_logs(args.run_dir / "logs", args.expected_decode)
    summary = {
        "topology": {"prefill_workers": 2, "decode_workers": 2, "ep_size": 2, "gpus": 8},
        "registered": {role: sorted(workers) for role, workers in identities.items()},
        "observed": {
            "prefill": sorted(observed_prefill),
            "decode": sorted(observed_decode),
        },
        "request_count": len(records),
        "prefill_distribution": dict(Counter(str(r["prefill_worker_id"]) for r in records)),
        "decode_distribution": dict(Counter(str(r["decode_worker_id"]) for r in records)),
        "mamba_blocks_imported": mamba_imports,
        "requests": records,
    }
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("PASS: all 2P2D workers served traffic and both decode workers imported KV + Mamba state")


if __name__ == "__main__":
    main()
