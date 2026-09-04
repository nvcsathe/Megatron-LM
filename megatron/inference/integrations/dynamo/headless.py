# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Start a non-leader rank agent without registering a Dynamo worker."""

from __future__ import annotations

import os

from megatron.inference.integrations.dynamo.args import Config, parse_args
from megatron.inference.integrations.dynamo.launcher import build_engine_command


def main(argv: list[str] | None = None) -> None:
    """Replace this process with one externally placed node-local rank agent."""

    config: Config = parse_args(argv)
    if config.launcher != "external":
        raise ValueError("The headless entrypoint requires --launcher external")
    if config.node_rank == 0:
        raise ValueError("The headless entrypoint requires --node-rank greater than zero")
    command = build_engine_command(config)
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
