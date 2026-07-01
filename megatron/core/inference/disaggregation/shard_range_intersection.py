# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Range intersection used by KV-cache and Mamba-state reshard planners."""

from __future__ import annotations

from typing import Optional, Tuple


def intersect(a: Tuple[int, int], b: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """Return the overlap of two half-open ranges, or ``None`` if disjoint."""

    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    return (lo, hi) if lo < hi else None
