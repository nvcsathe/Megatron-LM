# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Commands for launching one node-local Megatron rank agent."""

from __future__ import annotations

import sys

from megatron.inference.integrations.dynamo.args import Config


def build_engine_command(config: Config, parent_event_address: str | None = None) -> list[str]:
    """Build a node-local torchrun command for one Megatron replica."""

    command = [sys.executable, "-m", "torch.distributed.run"]
    if config.launcher == "local":
        command.extend(["--standalone", f"--nproc-per-node={config.nproc_per_node}"])
    elif config.launcher == "external":
        command.extend(
            [
                f"--nnodes={config.nnodes}",
                f"--nproc-per-node={config.nproc_per_node}",
                f"--node-rank={config.node_rank}",
                f"--master-addr={config.master_addr}",
                f"--master-port={config.master_port}",
            ]
        )
    else:
        raise ValueError(f"Unsupported engine launcher: {config.launcher}")
    command.extend(["--module", "megatron.inference.integrations.dynamo.engine_service"])
    if parent_event_address is not None:
        command.extend(["--dynamo-parent-event-address", parent_event_address])
    command.extend(["--role", config.role])
    if config.coordinator_host is not None:
        command.extend(["--coordinator-host", config.coordinator_host])
    if config.coordinator_port is not None:
        command.extend(["--coordinator-port", str(config.coordinator_port)])
    command.extend(config.megatron_argv)
    return command
