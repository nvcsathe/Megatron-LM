# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Direct KV transfer backend interface and active-backend factory."""

from __future__ import annotations

import os
from typing import Any, Optional

KVTransportBackend = Any


_backend: Optional[KVTransportBackend] = None


def _construct_backend(name: str) -> KVTransportBackend:
    normalized = name.lower().replace("_", "-")
    if normalized == "nixl":
        from .nixl import NixlTransferBackend

        return NixlTransferBackend
    raise ValueError("Unsupported KV transfer backend %r; expected 'nixl'." % name)


def get_kv_transport_backend() -> KVTransportBackend:
    """Return the active backend, constructing it from env on first use."""

    global _backend
    if _backend is None:
        _backend = _construct_backend(os.getenv("MEGATRON_KV_TRANSFER_BACKEND", "nixl"))
    return _backend


def set_kv_transport_backend(backend: Optional[KVTransportBackend]) -> None:
    """Override the active backend. ``None`` resets to env/default on next use."""

    global _backend
    _backend = backend
