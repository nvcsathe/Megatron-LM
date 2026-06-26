# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility wrapper for the direct NIXL KV transfer backend."""

from megatron.core.inference.disaggregation.transfer_backends.nixl import (
    KvTransferAgent,
    NixlTransferBackend,
    have_nixl,
    make_agent,
)

__all__ = [
    "KvTransferAgent",
    "NixlTransferBackend",
    "have_nixl",
    "make_agent",
]

