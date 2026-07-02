#!/usr/bin/env python3

"""Compare completed Nano v3 KV-routing and round-robin summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_summary(path: Path, expected_mode: str) -> dict[str, Any]:
    """Load a successful summary and validate its mode."""
    payload = json.loads(path.read_text())
    if payload.get("mode") != expected_mode:
        raise ValueError(f"{path} has mode={payload.get('mode')!r}, expected {expected_mode!r}")
    if payload.get("result") != "PASS":
        raise ValueError(f"{path} does not contain a passing result")
    return payload


def ratio(numerator: float, denominator: float) -> float | None:
    """Return a ratio unless its denominator is zero."""
    return numerator / denominator if denominator else None


def main() -> None:
    """Print a compact routing-versus-baseline comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("routing", type=Path)
    parser.add_argument("baseline", type=Path)
    args = parser.parse_args()

    routing = load_summary(args.routing, "routing")
    baseline = load_summary(args.baseline, "baseline")
    comparable_fields = (
        "dataset_sha256",
        "kv_block_size",
        "expected_workers",
        "requests",
        "concurrency",
        "max_tokens",
        "event_settle_seconds",
        "turn_settle_seconds",
    )
    for field in comparable_fields:
        if routing[field] != baseline[field]:
            raise ValueError(
                f"routing and baseline differ in {field}: {routing[field]!r} != {baseline[field]!r}"
            )
    fields = (
        "actual_request_hit_rate",
        "actual_block_hit_rate",
        "actual_cache_hits",
        "actual_cache_blocks_matched",
    )
    comparison: dict[str, Any] = {
        "routing_summary": str(args.routing),
        "baseline_summary": str(args.baseline),
    }
    for field in fields:
        routing_value = float(routing[field])
        baseline_value = float(baseline[field])
        comparison[field] = {
            "routing": routing_value,
            "baseline": baseline_value,
            "routing_over_baseline": ratio(routing_value, baseline_value),
        }

    routing_p95 = float(routing["latency"]["p95_ms"])
    baseline_p95 = float(baseline["latency"]["p95_ms"])
    comparison["latency_p95_ms"] = {
        "routing": routing_p95,
        "baseline": baseline_p95,
        "baseline_over_routing": ratio(baseline_p95, routing_p95),
    }
    routing_ttft = routing["ttft"]["p95_ms"]
    baseline_ttft = baseline["ttft"]["p95_ms"]
    if routing_ttft is not None and baseline_ttft is not None:
        routing_ttft = float(routing_ttft)
        baseline_ttft = float(baseline_ttft)
        comparison["ttft_p95_ms"] = {
            "routing": routing_ttft,
            "baseline": baseline_ttft,
            "baseline_over_routing": ratio(baseline_ttft, routing_ttft),
        }
    comparison["routing_post_warmup_affinity"] = routing["post_warmup_affinity"]
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
