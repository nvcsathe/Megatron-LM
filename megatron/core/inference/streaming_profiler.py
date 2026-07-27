# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Low-overhead profiling helpers for token streaming.

Profiling is opt-in and configured through environment variables so the setting
is inherited by the engine, coordinator, and text-generation frontend
processes. Samples are aggregated in memory and emitted periodically as one-line
JSON log records prefixed with ``MCORE_STREAM_PROFILE``.
"""

import json
import logging
import os
import socket
import time
from collections import defaultdict

logger = logging.getLogger(__name__)

_PROFILE_ENV = "MCORE_STREAMING_PROFILE"
_REPORT_INTERVAL_ENV = "MCORE_STREAMING_PROFILE_REPORT_INTERVAL"
_POLL_INTERVAL_ENV = "MCORE_STREAMING_POLL_INTERVAL_MS"
_DEFAULT_REPORT_INTERVAL = 1000
_DEFAULT_POLL_INTERVAL_MS = 5.0


def configure_streaming_profiling(
    enabled: bool, report_interval: int, poll_interval_ms: float
) -> None:
    """Configure streaming profiling for this process and its children.

    Args:
        enabled: Whether streaming profiling is enabled.
        report_interval: Number of component events between reports.
        poll_interval_ms: Frontend ZeroMQ empty-poll sleep in milliseconds.
    """
    if report_interval <= 0:
        raise ValueError("streaming profile report interval must be positive")
    if poll_interval_ms < 0:
        raise ValueError("streaming poll interval must be non-negative")
    os.environ[_PROFILE_ENV] = "1" if enabled else "0"
    os.environ[_REPORT_INTERVAL_ENV] = str(report_interval)
    os.environ[_POLL_INTERVAL_ENV] = str(poll_interval_ms)
    _PROFILERS.clear()


def get_streaming_poll_interval_seconds() -> float:
    """Return the configured frontend polling interval in seconds."""
    poll_interval_ms = float(os.getenv(_POLL_INTERVAL_ENV, _DEFAULT_POLL_INTERVAL_MS))
    if poll_interval_ms < 0:
        raise ValueError(f"{_POLL_INTERVAL_ENV} must be non-negative")
    return poll_interval_ms / 1000.0


class StreamingProfiler:
    """Aggregate timing samples and counters for one streaming component."""

    def __init__(
        self, component: str, *, enabled: bool | None = None, report_interval: int | None = None
    ) -> None:
        """Initialize a streaming profiler.

        Args:
            component: Process/component name included in reports.
            enabled: Explicit enablement override, primarily for tests.
            report_interval: Explicit report interval override.
        """
        self.component = component
        self.enabled = (
            os.getenv(_PROFILE_ENV, "0").lower() in {"1", "true", "yes", "on"}
            if enabled is None
            else enabled
        )
        self.report_interval = (
            int(os.getenv(_REPORT_INTERVAL_ENV, _DEFAULT_REPORT_INTERVAL))
            if report_interval is None
            else report_interval
        )
        if self.report_interval <= 0:
            raise ValueError("streaming profile report interval must be positive")
        self._pid = os.getpid()
        self.hostname = socket.gethostname()
        self._events = 0
        self._stages: dict[str, list[int]] = defaultdict(list)
        self._counters: dict[str, int] = defaultdict(int)

    def now_ns(self) -> int:
        """Return a profiling timestamp, or zero when profiling is disabled."""
        return time.perf_counter_ns() if self.enabled else 0

    def record_elapsed(self, stage: str, start_ns: int) -> None:
        """Record elapsed wall time since ``start_ns`` for a named stage."""
        if self.enabled and start_ns:
            self.record_duration(stage, time.perf_counter_ns() - start_ns)

    def record_duration(self, stage: str, duration_ns: int) -> None:
        """Record an already-computed duration for a named stage."""
        if not self.enabled:
            return
        self._ensure_process()
        self._stages[stage].append(max(0, duration_ns))

    def increment(self, counter: str, value: int = 1) -> None:
        """Increment a named counter."""
        if not self.enabled:
            return
        self._ensure_process()
        self._counters[counter] += value

    def maximum(self, counter: str, value: int) -> None:
        """Update a named maximum-value counter."""
        if not self.enabled:
            return
        self._ensure_process()
        self._counters[counter] = max(self._counters[counter], value)

    def event(self) -> None:
        """Record one component event and report when the interval is reached."""
        if not self.enabled:
            return
        self._ensure_process()
        self._events += 1
        if self._events >= self.report_interval:
            self.flush()

    def flush(self) -> None:
        """Emit and reset the current aggregate window."""
        if not self.enabled or self._events == 0:
            return
        self._ensure_process()
        payload = {
            "component": self.component,
            "hostname": self.hostname,
            "pid": self._pid,
            "events": self._events,
            "counters": dict(sorted(self._counters.items())),
            "stages": {
                stage: self._summarize(samples)
                for stage, samples in sorted(self._stages.items())
                if samples
            },
        }
        logger.info("MCORE_STREAM_PROFILE %s", json.dumps(payload, sort_keys=True))
        self._events = 0
        self._stages.clear()
        self._counters.clear()

    def _ensure_process(self) -> None:
        """Discard inherited samples after a multiprocessing fork."""
        pid = os.getpid()
        if pid == self._pid:
            return
        self._pid = pid
        self._events = 0
        self._stages.clear()
        self._counters.clear()

    @staticmethod
    def _summarize(samples: list[int]) -> dict[str, float | int]:
        ordered = sorted(samples)

        def percentile(fraction: float) -> float:
            index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
            return ordered[index] / 1e6

        return {
            "count": len(ordered),
            "avg_ms": sum(ordered) / len(ordered) / 1e6,
            "p50_ms": percentile(0.50),
            "p95_ms": percentile(0.95),
            "p99_ms": percentile(0.99),
            "max_ms": ordered[-1] / 1e6,
        }


_PROFILERS: dict[str, StreamingProfiler] = {}


def get_streaming_profiler(component: str) -> StreamingProfiler:
    """Return the process-local profiler for ``component``."""
    profiler = _PROFILERS.get(component)
    if profiler is None:
        profiler = StreamingProfiler(component)
        _PROFILERS[component] = profiler
    return profiler


def flush_streaming_profilers() -> None:
    """Flush all streaming profilers created in the current process."""
    for profiler in _PROFILERS.values():
        profiler.flush()
