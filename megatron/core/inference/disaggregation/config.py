# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Runtime configuration for coordinator-native disaggregated inference."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class DisaggregationConfig:
    """Settings resolved for one prefill or decode engine instance."""

    role: Literal["prefill", "decode"]
    identity: str
    spawn_coordinator: bool
    router: str
    kv_transport_backend: str

    def __post_init__(self) -> None:
        if self.role not in ("prefill", "decode"):
            raise ValueError(f"Unsupported disaggregation role: {self.role!r}")
        if not self.identity:
            raise ValueError("A disaggregated engine requires a non-empty identity")
