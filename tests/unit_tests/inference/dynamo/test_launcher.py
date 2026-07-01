from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from megatron.inference.integrations.dynamo.args import parse_args
from megatron.inference.integrations.dynamo.llm_engine import MegatronLLMEngine
from megatron.inference.integrations.dynamo.main import main


def _argv():
    return [
        "--role",
        "aggregated",
        "--model",
        "model-meta",
        "--nproc-per-node",
        "2",
        "--",
        "--load",
        "/checkpoints/model",
        "--tensor-model-parallel-size",
        "2",
    ]


def test_parse_args_splits_dynamo_and_megatron_arguments():
    config = parse_args(_argv())
    assert config.component == "backend"
    assert config.nproc_per_node == 2
    assert config.megatron_argv == [
        "--load",
        "/checkpoints/model",
        "--tensor-model-parallel-size",
        "2",
    ]


def test_disaggregated_role_requires_transfer_and_coordinator_addresses():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--role",
                "prefill",
                "--model",
                "model-meta",
                "--nproc-per-node",
                "1",
                "--",
                "--load",
                "/checkpoint",
            ]
        )


def test_public_entrypoint_uses_common_runner():
    with patch("megatron.inference.integrations.dynamo.main.run") as run:
        main()
    run.assert_called_once_with(MegatronLLMEngine)


def test_owned_engine_command_targets_megatron_only_service():
    config = parse_args(_argv())
    engine = MegatronLLMEngine(config)
    command = engine._engine_command(Path("/tmp/ready.json"))

    assert command[1:4] == ["-m", "torch.distributed.run", "--standalone"]
    assert "--nproc-per-node=2" in command
    assert "megatron.inference.integrations.dynamo.engine_service" in command
    assert "dynamo.megatron" not in command
    assert command[-4:] == [
        "--load",
        "/checkpoints/model",
        "--tensor-model-parallel-size",
        "2",
    ]


@pytest.mark.asyncio
async def test_from_args_is_side_effect_free(monkeypatch):
    async def fail_create_subprocess(*args, **kwargs):
        raise AssertionError("from_args must not start a process")

    monkeypatch.setattr("asyncio.create_subprocess_exec", fail_create_subprocess)
    engine, worker = await MegatronLLMEngine.from_args(_argv())

    assert engine._process is None
    assert engine.client is None
    assert worker.component == "backend"
    assert worker.model_name == "model-meta"


@pytest.mark.asyncio
async def test_readiness_reports_early_child_failure(tmp_path):
    engine = MegatronLLMEngine(parse_args(_argv()))
    engine._process = SimpleNamespace(returncode=17)
    with pytest.raises(RuntimeError, match="exited before readiness.*17"):
        await engine._wait_for_readiness(tmp_path / "missing.json")


@pytest.mark.asyncio
async def test_readiness_descriptor_is_loaded(tmp_path):
    engine = MegatronLLMEngine(parse_args(_argv()))
    path = tmp_path / "ready.json"
    expected = {"coordinator_address": "tcp://127.0.0.1:5000"}
    path.write_text(json.dumps(expected))
    assert await engine._wait_for_readiness(path) == expected
