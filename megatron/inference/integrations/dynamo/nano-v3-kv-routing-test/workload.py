#!/usr/bin/env python3

"""Drive Nano v3 KV-event, KV-routing, and round-robin baseline tests."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import random
import re
import statistics
import time
import traceback
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BAD_EVENT_STATUSES = {"parent_block_not_found", "block_not_found", "invalid_block"}
CACHE_COUNTER_RE = re.compile(
    r"prefix cache \(cumul\):\s*(?P<hits>[0-9]+) hits,\s*(?P<blocks>[0-9]+) blocks matched"
)


def http_json(url: str, body: dict[str, Any] | None = None, timeout: float = 900) -> dict[str, Any]:
    """Fetch JSON from an HTTP endpoint."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {detail}") from error


def http_text(url: str, timeout: float = 30) -> str:
    """Fetch text from an HTTP endpoint."""
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def prometheus_samples(text: str, suffix: str) -> list[tuple[dict[str, str], float]]:
    """Return label/value samples for metrics whose name ends with suffix."""
    samples: list[tuple[dict[str, str], float]] = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        metric, _, value_text = line.rpartition(" ")
        name, _, labels_text = metric.partition("{")
        if not name.endswith(suffix):
            continue
        labels: dict[str, str] = {}
        if labels_text:
            labels_text = labels_text.rstrip("}")
            for key, value in re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)="([^"]*)"', labels_text):
                labels[key] = value
        try:
            samples.append((labels, float(value_text)))
        except ValueError:
            continue
    return samples


def event_counters(metrics: str) -> dict[tuple[str, str], float]:
    """Collect event indexer counters by event type and status."""
    counters: dict[tuple[str, str], float] = defaultdict(float)
    for labels, value in prometheus_samples(metrics, "kv_cache_events_applied"):
        counters[(labels.get("event_type", ""), labels.get("status", ""))] += value
    return dict(counters)


def event_warning_counters(metrics: str) -> dict[str, float]:
    """Collect suspicious-event counters by warning kind."""
    counters: dict[str, float] = defaultdict(float)
    for labels, value in prometheus_samples(metrics, "kv_cache_event_warnings"):
        counters[labels.get("warning_kind", "")] += value
    return dict(counters)


def serialize_event_counters(counters: dict[tuple[str, str], float]) -> dict[str, float]:
    """Convert tuple-keyed event counters into stable JSON object keys."""
    return {
        f"{event_type}:{status}": value
        for (event_type, status), value in sorted(counters.items())
    }


def wait_for_workers(worker_dir: Path, expected: int, timeout: float = 180) -> list[int]:
    """Load the structured identities emitted by ready backend adapters."""
    deadline = time.monotonic() + timeout
    last_workers: list[int] = []
    last_error: str | None = None
    while time.monotonic() < deadline:
        identities = sorted(worker_dir.glob("worker-*"))
        try:
            last_workers = sorted(
                {int(json.loads(path.read_text())["worker_id"]) for path in identities}
            )
            last_error = None
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            last_error = str(error)
        if len(identities) == expected and len(last_workers) == expected:
            return last_workers
        time.sleep(2)
    raise RuntimeError(
        f"expected {expected} structured worker identities in {worker_dir}, "
        f"found {len(last_workers)}: {last_workers}; last parse error: {last_error}"
    )


def wait_for_stored_events(
    url: str,
    initial_count: float,
    minimum_delta: int,
    timeout: float = 120,
) -> tuple[dict[tuple[str, str], float], float]:
    """Wait for the stored-event count to reach its minimum and become stable."""
    deadline = time.monotonic() + timeout
    previous: float | None = None
    stable_polls = 0
    latest: dict[tuple[str, str], float] = {}
    while time.monotonic() < deadline:
        latest = event_counters(http_text(f"{url}/metrics"))
        current = latest.get(("stored", "ok"), 0)
        if current - initial_count >= minimum_delta:
            stable_polls = stable_polls + 1 if current == previous else 1
            if stable_polls >= 3:
                return latest, current - initial_count
        previous = current
        time.sleep(1)
    current = latest.get(("stored", "ok"), 0)
    raise RuntimeError(
        f"stored events did not quiesce: initial={initial_count}, current={current}, "
        f"required_delta={minimum_delta}"
    )


def completion(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    worker_id: int | None = None,
) -> dict[str, Any]:
    """Issue one deterministic completion and normalize its test metadata."""
    nvext: dict[str, Any] = {"extra_fields": ["worker_id", "timing"]}
    if worker_id is not None:
        nvext["backend_instance_id"] = worker_id
    started = time.monotonic()
    response = http_json(
        f"{url}/v1/completions",
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
            "nvext": nvext,
        },
    )
    elapsed_ms = (time.monotonic() - started) * 1000
    extension = response.get("nvext") or {}
    worker = extension.get("worker_id") or {}
    timing = extension.get("timing") or {}
    usage = response.get("usage") or {}
    completion_tokens = int(usage.get("completion_tokens", 0))
    if not response.get("choices") or completion_tokens < 1:
        raise RuntimeError(f"model returned a degenerate completion: {response}")
    selected = worker.get("decode_worker_id")
    if selected is None:
        selected = worker.get("prefill_worker_id")
    dp_rank = worker.get("decode_dp_rank")
    if dp_rank is None:
        dp_rank = worker.get("prefill_dp_rank")
    text = "".join(choice.get("text", "") for choice in response.get("choices", []))
    return {
        "worker_id": int(selected) if selected is not None else None,
        "dp_rank": int(dp_rank) if dp_rank is not None else None,
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": completion_tokens,
        "predicted_kv_hit_rate": timing.get("kv_hit_rate"),
        "ttft_ms": timing.get("ttft_ms"),
        "elapsed_ms": elapsed_ms,
        "text": text,
    }


def percentile(values: list[float], quantile: float) -> float | None:
    """Return a nearest-rank percentile."""
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def latency_summary(records: list[dict[str, Any]]) -> dict[str, float | None]:
    """Summarize client-observed request latency."""
    return field_summary(records, "elapsed_ms")


def field_summary(records: list[dict[str, Any]], field: str) -> dict[str, float | None]:
    """Summarize a numeric per-request field."""
    values = [float(record[field]) for record in records if record.get(field) is not None]
    return {
        "mean_ms": statistics.fmean(values) if values else None,
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
    }


def write_json(path: Path, payload: Any) -> None:
    """Write a stable, human-readable JSON artifact."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write request records as JSONL."""
    with path.open("w") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")


def file_sha256(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_counters(log_dir: Path) -> dict[str, Any]:
    """Read one cumulative cache-counter maximum per logical worker log."""
    per_worker: dict[str, dict[str, int]] = {}
    for path in sorted(log_dir.glob("node-*-worker-*.log")):
        matches = list(CACHE_COUNTER_RE.finditer(path.read_text(errors="replace")))
        if not matches:
            per_worker[path.name] = {"hits": 0, "blocks_matched": 0}
            continue
        per_worker[path.name] = {
            "hits": max(int(match.group("hits")) for match in matches),
            "blocks_matched": max(int(match.group("blocks")) for match in matches),
        }
    return {
        "per_worker": per_worker,
        "hits": sum(item["hits"] for item in per_worker.values()),
        "blocks_matched": sum(item["blocks_matched"] for item in per_worker.values()),
    }


def cache_counter_deltas(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Compute per-worker cache-counter changes between two log snapshots."""
    before_workers = before["per_worker"]
    after_workers = after["per_worker"]
    workers = sorted(set(before_workers) | set(after_workers))
    per_worker = {
        worker: {
            "hits": after_workers.get(worker, {}).get("hits", 0)
            - before_workers.get(worker, {}).get("hits", 0),
            "blocks_matched": after_workers.get(worker, {}).get("blocks_matched", 0)
            - before_workers.get(worker, {}).get("blocks_matched", 0),
        }
        for worker in workers
    }
    return {
        "per_worker": per_worker,
        "hits": sum(item["hits"] for item in per_worker.values()),
        "blocks_matched": sum(item["blocks_matched"] for item in per_worker.values()),
    }


def make_event_prompt(worker_index: int, prefix_repeat: int) -> str:
    """Build a worker-specific prompt containing many reusable full blocks."""
    marker = f"This is the private cache-validation corpus for logical worker {worker_index}. "
    body = (" evidence" * prefix_repeat).strip()
    return f"{marker}{body}\nSummarize the corpus in one sentence."


def run_events(args: argparse.Namespace) -> None:
    """Validate event ingestion and repeated-prompt cache reuse on every worker."""
    workers = wait_for_workers(args.worker_dir, args.expected_workers)
    before = event_counters(http_text(f"{args.url}/metrics"))
    records: list[dict[str, Any]] = []
    prompts: dict[int, str] = {}

    for index, worker in enumerate(workers):
        # Keep this below the configured prefill window so each first request
        # produces exactly one multi-block stored event.
        prompt = make_event_prompt(index, min(1024, max(512, args.prefix_repeat)))
        prompts[worker] = prompt
        record = completion(args.url, args.model, prompt, args.max_tokens, worker)
        record.update({"phase": "store", "requested_worker_id": worker})
        if record["worker_id"] != worker or record["dp_rank"] != 0:
            raise AssertionError(f"directed request for {worker} returned {record}")
        records.append(record)

    time.sleep(args.event_settle_seconds)
    initial_stored = before.get(("stored", "ok"), 0)
    after_store, stored_ok_delta = wait_for_stored_events(
        args.url, initial_stored, args.expected_workers
    )
    removed_after_store = after_store.get(("removed", "ok"), 0) - before.get(
        ("removed", "ok"), 0
    )
    cleared_after_store = after_store.get(("cleared", "ok"), 0) - before.get(
        ("cleared", "ok"), 0
    )
    cache_before_reuse = cache_counters(args.log_dir)
    diagnostics = {
        "workers": workers,
        "event_counters_before": serialize_event_counters(before),
        "event_counters_after_store": serialize_event_counters(after_store),
        "stored_events_from_initial_prompts": stored_ok_delta,
        "removed_events_after_initial_prompts": removed_after_store,
        "cleared_events_after_initial_prompts": cleared_after_store,
        "cache_counters_before_reuse": cache_before_reuse,
    }
    append_jsonl(args.run_dir / "records.jsonl", records)
    write_json(args.run_dir / "event-diagnostics.json", diagnostics)
    if stored_ok_delta != args.expected_workers:
        raise AssertionError(
            f"expected exactly one stored event from each of {args.expected_workers} workers, "
            f"got {stored_ok_delta}; model-parallel ranks may be publishing duplicates"
        )
    if removed_after_store != 0:
        raise AssertionError(
            f"initial prompts unexpectedly produced {removed_after_store} removed events; "
            "sequential reuse requires the lru prefix-cache eviction policy"
        )
    if cleared_after_store != 0:
        raise AssertionError(
            f"initial prompts unexpectedly produced {cleared_after_store} cleared events; "
            "an idle reset discarded the prefix cache before reuse"
        )
    if len(cache_before_reuse["per_worker"]) != args.expected_workers:
        raise AssertionError(
            f"found {len(cache_before_reuse['per_worker'])} worker logs before reuse, "
            f"expected {args.expected_workers}"
        )

    for worker in workers:
        record = completion(args.url, args.model, prompts[worker], args.max_tokens, worker)
        record.update({"phase": "reuse", "requested_worker_id": worker})
        if record["worker_id"] != worker or record["dp_rank"] != 0:
            raise AssertionError(f"repeat request for {worker} returned {record}")
        records.append(record)

    time.sleep(args.event_settle_seconds)
    final_metrics = http_text(f"{args.url}/metrics")
    final_events = event_counters(final_metrics)
    stored_after_reuse = final_events.get(("stored", "ok"), 0) - after_store.get(
        ("stored", "ok"), 0
    )
    removed_after_reuse = final_events.get(("removed", "ok"), 0) - after_store.get(
        ("removed", "ok"), 0
    )
    cleared_after_reuse = final_events.get(("cleared", "ok"), 0) - after_store.get(
        ("cleared", "ok"), 0
    )
    time.sleep(args.log_settle_seconds)
    counters = cache_counters(args.log_dir)
    counter_deltas = cache_counter_deltas(cache_before_reuse, counters)
    warnings = {key: value for key, value in event_warning_counters(final_metrics).items() if value}
    diagnostics.update(
        {
            "event_counters_after_reuse": serialize_event_counters(final_events),
            "stored_events_from_repeated_prompts": stored_after_reuse,
            "removed_events_from_repeated_prompts": removed_after_reuse,
            "cleared_events_from_repeated_prompts": cleared_after_reuse,
            "cache_counters_after_reuse": counters,
            "cache_counter_deltas_from_repeated_prompts": counter_deltas,
            "event_warning_counters": warnings,
        }
    )
    append_jsonl(args.run_dir / "records.jsonl", records)
    write_json(args.run_dir / "event-diagnostics.json", diagnostics)
    if cleared_after_reuse != 0:
        raise AssertionError(
            f"identical repeated prompts unexpectedly produced {cleared_after_reuse} cleared events"
        )
    if removed_after_reuse != 0:
        raise AssertionError(
            f"identical repeated prompts unexpectedly produced {removed_after_reuse} removed events"
        )
    if stored_after_reuse != 0:
        raise AssertionError(
            f"identical repeated prompts unexpectedly produced {stored_after_reuse} new stored events"
        )
    bad = {
        f"{event_type}:{status}": value
        for (event_type, status), value in final_events.items()
        if status in BAD_EVENT_STATUSES and value > 0
    }
    if bad:
        raise AssertionError(f"router indexer reported invalid events: {bad}")
    if warnings:
        raise AssertionError(f"router indexer reported suspicious events: {warnings}")

    if len(counters["per_worker"]) != args.expected_workers:
        raise AssertionError(
            f"found {len(counters['per_worker'])} worker logs, expected {args.expected_workers}"
        )
    invalid_hit_deltas = {
        name: item["hits"]
        for name, item in counter_deltas["per_worker"].items()
        if item["hits"] != 1
    }
    if invalid_hit_deltas:
        raise AssertionError(
            "expected exactly one repeated-prompt cache hit per worker, got "
            f"{invalid_hit_deltas}"
        )

    summary = {
        "mode": "events",
        "workers": workers,
        "stored_ok_event_delta": stored_ok_delta,
        "removed_events_after_initial_prompts": removed_after_store,
        "cleared_events_after_initial_prompts": cleared_after_store,
        "stored_events_from_repeated_prompts": stored_after_reuse,
        "removed_events_from_repeated_prompts": removed_after_reuse,
        "cleared_events_from_repeated_prompts": cleared_after_reuse,
        "event_counters_before": serialize_event_counters(before),
        "event_counters_after_store": serialize_event_counters(after_store),
        "event_counters_after": serialize_event_counters(final_events),
        "event_warning_counters": warnings,
        "cache_counters_before_reuse": cache_before_reuse,
        "cache_counters": counters,
        "cache_counter_deltas_from_repeated_prompts": counter_deltas,
        "latency": latency_summary(records),
        "ttft": field_summary(records, "ttft_ms"),
        "result": "PASS",
    }
    write_json(args.run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def generate_dataset(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Generate a deterministic growing-conversation shared-prefix workload."""
    rng = random.Random(args.seed)
    families: dict[int, tuple[str, list[str]]] = {}
    for family in range(args.families):
        marker = f"Knowledge base {family:04d}; all following facts belong only to this family. "
        shared = marker + (" evidence" * args.prefix_repeat)
        families[family] = (shared, [])

    records: list[dict[str, Any]] = []
    request_index = 0
    for turn in range(args.turns):
        order = list(range(args.families))
        rng.shuffle(order)
        for family in order:
            shared, history = families[family]
            history.append(
                f"\nUser turn {turn}: analyze family {family:04d}, subproblem {turn:02d}."
                f"\nAssistant turn {turn}: retained observation {family * 1000 + turn}."
            )
            records.append(
                {
                    "request_index": request_index,
                    "family": family,
                    "turn": turn,
                    "prompt": shared + "".join(history) + "\nAssistant:",
                }
            )
            request_index += 1
    append_jsonl(args.dataset, records)
    return records


def load_dataset(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL workload."""
    with path.open() as source:
        return [json.loads(line) for line in source if line.strip()]


def execute_records(args: argparse.Namespace, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Execute records concurrently and preserve deterministic result ordering."""

    def execute(record: dict[str, Any]) -> dict[str, Any]:
        result = completion(args.url, args.model, record["prompt"], args.max_tokens)
        result.update({key: record[key] for key in ("request_index", "family", "turn")})
        return result

    completed: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(execute, record): record for record in records}
        for future in concurrent.futures.as_completed(futures):
            completed.append(future.result())
    completed.sort(key=lambda record: record["request_index"])
    return completed


def execute_turns(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    first_turn_settle: float,
) -> list[dict[str, Any]]:
    """Execute growing conversations one turn at a time."""
    completed: list[dict[str, Any]] = []
    turns = sorted({int(record["turn"]) for record in records})
    for index, turn in enumerate(turns):
        turn_records = [record for record in records if int(record["turn"]) == turn]
        completed.extend(execute_records(args, turn_records))
        if index + 1 < len(turns):
            delay = first_turn_settle if index == 0 else args.turn_settle_seconds
            time.sleep(delay)
    completed.sort(key=lambda record: record["request_index"])
    return completed


def workload_summary(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    event_metrics: str,
) -> dict[str, Any]:
    """Build shared routing/baseline metrics from completed records and worker logs."""
    counters = cache_counters(args.log_dir)
    if len(counters["per_worker"]) != args.expected_workers:
        raise AssertionError(
            f"found {len(counters['per_worker'])} worker logs, expected {args.expected_workers}"
        )
    eligible_blocks = sum(int(record["prompt_tokens"]) // args.block_size for record in records)
    predicted = [
        float(record["predicted_kv_hit_rate"])
        for record in records
        if record.get("predicted_kv_hit_rate") is not None
    ]
    worker_distribution = Counter(str(record.get("worker_id")) for record in records)
    return {
        "requests": len(records),
        "dataset_path": str(args.dataset),
        "dataset_sha256": file_sha256(args.dataset),
        "kv_block_size": args.block_size,
        "expected_workers": args.expected_workers,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "event_settle_seconds": args.event_settle_seconds,
        "turn_settle_seconds": args.turn_settle_seconds,
        "worker_distribution": dict(sorted(worker_distribution.items())),
        "latency": latency_summary(records),
        "ttft": field_summary(records, "ttft_ms"),
        "mean_predicted_kv_hit_rate": statistics.fmean(predicted) if predicted else None,
        "eligible_prompt_blocks": eligible_blocks,
        "actual_cache_hits": counters["hits"],
        "actual_cache_blocks_matched": counters["blocks_matched"],
        "actual_request_hit_rate": counters["hits"] / len(records) if records else 0,
        "actual_block_hit_rate": counters["blocks_matched"] / eligible_blocks if eligible_blocks else 0,
        "cache_counters": counters,
        "event_counters": {
            f"{key[0]}:{key[1]}": value for key, value in event_counters(event_metrics).items()
        },
        "event_warning_counters": event_warning_counters(event_metrics),
    }


def routing_affinity(
    warm_records: list[dict[str, Any]], measured_records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Measure routing to the warm worker and to any previously caching worker."""
    warm_worker = {int(record["family"]): record["worker_id"] for record in warm_records}
    if any(worker is None for worker in warm_worker.values()):
        raise AssertionError("routing warm-up responses did not include worker IDs")

    known_workers = {family: {worker} for family, worker in warm_worker.items()}
    warm_matches = 0
    known_cache_matches = 0
    cold_spills: list[dict[str, Any]] = []
    for record in sorted(
        measured_records, key=lambda item: (int(item["turn"]), int(item["request_index"]))
    ):
        family = int(record["family"])
        worker = record["worker_id"]
        if worker is None:
            raise AssertionError(f"routing response did not include a worker ID: {record}")
        warm_matches += worker == warm_worker[family]
        if worker in known_workers[family]:
            known_cache_matches += 1
        else:
            cold_spills.append(
                {
                    "request_index": record["request_index"],
                    "family": family,
                    "turn": record["turn"],
                    "worker_id": worker,
                }
            )
        known_workers[family].add(worker)

    count = len(measured_records)
    return {
        "warm_worker_affinity": warm_matches / count if count else 1.0,
        "known_cache_affinity": known_cache_matches / count if count else 1.0,
        "cold_spills": cold_spills,
        "known_workers_by_family": {
            str(family): sorted(workers) for family, workers in sorted(known_workers.items())
        },
    }


def run_routing(args: argparse.Namespace) -> None:
    """Warm each family, then verify event-driven routing to cached workers."""
    workers = wait_for_workers(args.worker_dir, args.expected_workers)
    dataset = generate_dataset(args)
    warm_inputs = [record for record in dataset if int(record["turn"]) == 0]
    measured_inputs = [record for record in dataset if int(record["turn"]) > 0]
    warm_records = execute_records(args, warm_inputs)
    time.sleep(args.event_settle_seconds)
    cache_after_warmup = cache_counters(args.log_dir)
    measured_records = execute_turns(args, measured_inputs, args.turn_settle_seconds)
    all_records = sorted(warm_records + measured_records, key=lambda record: record["request_index"])

    time.sleep(args.log_settle_seconds)
    metrics = http_text(f"{args.url}/metrics")
    final_cache = cache_counters(args.log_dir)
    measured_cache_delta = cache_counter_deltas(cache_after_warmup, final_cache)
    measured_hit_rate = (
        measured_cache_delta["hits"] / len(measured_records) if measured_records else 1.0
    )
    affinity = routing_affinity(warm_records, measured_records)
    bad = {
        f"{event_type}:{status}": value
        for (event_type, status), value in event_counters(metrics).items()
        if status in BAD_EVENT_STATUSES and value > 0
    }
    warnings = {key: value for key, value in event_warning_counters(metrics).items() if value}

    append_jsonl(args.run_dir / "records.jsonl", all_records)
    diagnostics = {
        "workers": workers,
        "warm_requests": len(warm_records),
        "measured_requests": len(measured_records),
        "minimum_affinity": args.min_affinity,
        **affinity,
        "cache_counters_after_warmup": cache_after_warmup,
        "cache_counters_final": final_cache,
        "measured_cache_counter_delta": measured_cache_delta,
        "measured_actual_request_hit_rate": measured_hit_rate,
        "event_counters": serialize_event_counters(event_counters(metrics)),
        "event_warning_counters": warnings,
        "invalid_event_counters": bad,
    }
    write_json(args.run_dir / "routing-diagnostics.json", diagnostics)

    if bad:
        raise AssertionError(f"router indexer reported invalid events: {bad}")
    if warnings:
        raise AssertionError(f"router indexer reported suspicious events: {warnings}")
    if affinity["known_cache_affinity"] < args.min_affinity:
        raise AssertionError(
            f"known-cache affinity {affinity['known_cache_affinity']:.4f} is below "
            f"{args.min_affinity:.4f}; cold spills={len(affinity['cold_spills'])}"
        )
    if measured_hit_rate < args.min_affinity:
        raise AssertionError(
            f"actual post-warmup cache-hit rate {measured_hit_rate:.4f} is below "
            f"{args.min_affinity:.4f}"
        )

    summary = workload_summary(args, all_records, metrics)
    summary.update(
        {
            "mode": "routing",
            "workers": workers,
            "families": args.families,
            "turns": args.turns,
            "warm_worker_affinity": affinity["warm_worker_affinity"],
            "known_cache_affinity": affinity["known_cache_affinity"],
            "cold_spills": len(affinity["cold_spills"]),
            "cache_counters_after_warmup": cache_after_warmup,
            "measured_cache_counter_delta": measured_cache_delta,
            "measured_actual_request_hit_rate": measured_hit_rate,
            "minimum_affinity": args.min_affinity,
            "result": "PASS",
        }
    )
    write_json(args.run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_baseline(args: argparse.Namespace) -> None:
    """Replay the routing dataset through the round-robin frontend."""
    if not args.dataset.is_file():
        raise FileNotFoundError(
            f"routing dataset does not exist: {args.dataset}; run launch-routing.sh first"
        )
    dataset = load_dataset(args.dataset)
    records = execute_turns(args, dataset, args.event_settle_seconds)
    workers = sorted({int(record["worker_id"]) for record in records if record["worker_id"] is not None})
    if len(workers) != args.expected_workers:
        raise AssertionError(
            f"round-robin responses used {len(workers)} workers, expected {args.expected_workers}: {workers}"
        )
    time.sleep(args.log_settle_seconds)
    metrics = http_text(f"{args.url}/metrics")
    append_jsonl(args.run_dir / "records.jsonl", records)
    summary = workload_summary(args, records, metrics)
    summary.update({"mode": "baseline", "workers": workers, "result": "PASS"})
    write_json(args.run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    """Parse the common test-driver CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("events", "routing", "baseline"))
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--worker-dir", type=Path, required=True)
    parser.add_argument("--expected-workers", type=int, required=True)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--families", type=int, default=32)
    parser.add_argument("--turns", type=int, default=8)
    parser.add_argument("--prefix-repeat", type=int, default=3072)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--event-settle-seconds", type=float, default=5)
    parser.add_argument("--turn-settle-seconds", type=float, default=1)
    parser.add_argument("--log-settle-seconds", type=float, default=2)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--min-affinity", type=float, default=0.95)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.expected_workers < 1 or args.block_size < 1 or args.concurrency < 1:
        parser.error("worker count, block size, and concurrency must be positive")
    if args.families < 1 or args.turns < 1 or args.prefix_repeat < 1:
        parser.error("families, turns, and prefix repeat must be positive")
    if args.max_tokens < 1:
        parser.error("max tokens must be positive")
    if min(args.event_settle_seconds, args.turn_settle_seconds, args.log_settle_seconds) < 0:
        parser.error("settle delays cannot be negative")
    if not 0 <= args.min_affinity <= 1:
        parser.error("minimum affinity must be between zero and one")
    return args


def main() -> None:
    """Run the selected integration test."""
    args = parse_args()
    try:
        if args.mode == "events":
            run_events(args)
        elif args.mode == "routing":
            run_routing(args)
        else:
            run_baseline(args)
    except Exception as error:
        failure = {
            "mode": args.mode,
            "result": "FAIL",
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(args.run_dir / "failure.json", failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        raise


if __name__ == "__main__":
    main()
