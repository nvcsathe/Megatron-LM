# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
import logging

import pytest

from megatron.core.inference.streaming_profiler import (
    StreamingProfiler,
    get_streaming_poll_interval_seconds,
)


def test_streaming_profiler_reports_aggregated_stage_data(caplog):
    caplog.set_level(logging.INFO, logger="megatron.core.inference.streaming_profiler")
    profiler = StreamingProfiler("test-component", enabled=True, report_interval=2)

    profiler.record_duration("pack", 1_000_000)
    profiler.increment("tokens", 2)
    profiler.event()
    profiler.record_duration("pack", 3_000_000)
    profiler.increment("tokens", 1)
    profiler.event()

    records = [
        record.message
        for record in caplog.records
        if record.message.startswith("MCORE_STREAM_PROFILE ")
    ]
    assert len(records) == 1
    report = json.loads(records[0].removeprefix("MCORE_STREAM_PROFILE "))
    assert report["component"] == "test-component"
    assert report["events"] == 2
    assert report["counters"] == {"tokens": 3}
    assert report["stages"]["pack"] == {
        "count": 2,
        "avg_ms": 2.0,
        "p50_ms": 1.0,
        "p95_ms": 1.0,
        "p99_ms": 1.0,
        "max_ms": 3.0,
    }


def test_streaming_profiler_is_noop_when_disabled(caplog):
    caplog.set_level(logging.INFO, logger="megatron.core.inference.streaming_profiler")
    profiler = StreamingProfiler("test-component", enabled=False, report_interval=1)

    assert profiler.now_ns() == 0
    profiler.record_duration("pack", 1_000_000)
    profiler.increment("tokens")
    profiler.event()
    profiler.flush()

    assert not [
        record for record in caplog.records if record.message.startswith("MCORE_STREAM_PROFILE ")
    ]


@pytest.mark.parametrize(("milliseconds", "seconds"), [(0, 0.0), (0.5, 0.0005), (5, 0.005)])
def test_streaming_poll_interval_conversion(monkeypatch, milliseconds, seconds):
    monkeypatch.setenv("MCORE_STREAMING_POLL_INTERVAL_MS", str(milliseconds))

    assert get_streaming_poll_interval_seconds() == seconds


def test_streaming_poll_interval_rejects_negative_values(monkeypatch):
    monkeypatch.setenv("MCORE_STREAMING_POLL_INTERVAL_MS", "-1")

    with pytest.raises(ValueError, match="must be non-negative"):
        get_streaming_poll_interval_seconds()
