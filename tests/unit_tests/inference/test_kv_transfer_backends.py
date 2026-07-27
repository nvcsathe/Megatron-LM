# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.inference.disaggregation.transfer_backends import base


class _FakeCudaBuffer:
    is_cuda = True


def test_nixl_rejects_cuda_major_variant_mismatch(monkeypatch):
    from megatron.core.inference.disaggregation.transfer_backends import nixl as nixl_mod

    monkeypatch.setattr(nixl_mod, "_NIXL_VARIANT", "nixl_cu12")
    monkeypatch.setattr(nixl_mod.torch.version, "cuda", "13.2")

    with pytest.raises(RuntimeError, match="NIXL selected nixl_cu12"):
        nixl_mod._validate_nixl_cuda_support(object(), _FakeCudaBuffer())


def test_nixl_rejects_ucx_without_vram_support(monkeypatch):
    from megatron.core.inference.disaggregation.transfer_backends import nixl as nixl_mod

    class FakeAgent:
        def get_backend_mem_types(self, backend):
            assert backend == "UCX"
            return ["DRAM"]

    monkeypatch.setattr(nixl_mod, "_NIXL_VARIANT", "nixl_cu13")
    monkeypatch.setattr(nixl_mod.torch.version, "cuda", "13.2")

    with pytest.raises(RuntimeError, match="does not report CUDA/VRAM"):
        nixl_mod._validate_nixl_cuda_support(FakeAgent(), _FakeCudaBuffer())


def test_nixl_accepts_matching_variant_with_vram_support(monkeypatch):
    from megatron.core.inference.disaggregation.transfer_backends import nixl as nixl_mod

    class FakeAgent:
        def get_backend_mem_types(self, backend):
            assert backend == "UCX"
            return ["DRAM", "VRAM"]

    monkeypatch.setattr(nixl_mod, "_NIXL_VARIANT", "nixl_cu13")
    monkeypatch.setattr(nixl_mod.torch.version, "cuda", "13.2")

    nixl_mod._validate_nixl_cuda_support(FakeAgent(), _FakeCudaBuffer())


def test_backend_registry_selects_by_explicit_name():
    assert base.construct_kv_transfer_backend_class("nixl").name == "nixl"

    try:
        base.construct_kv_transfer_backend_class("unsupported")
    except ValueError as exc:
        assert "expected 'nixl'" in str(exc)
    else:
        raise AssertionError("unsupported backend should raise")


def test_nixl_direct_backend_exports_metadata_with_fake_agent(monkeypatch):
    from megatron.core.inference.disaggregation.transfer_backends import nixl as nixl_mod

    class FakeAgentConfig:
        def __init__(self, *, enable_prog_thread):
            self.enable_prog_thread = enable_prog_thread

    class FakeAgent:
        def __init__(self, name, config):
            self.name = name
            self.config = config

        def get_agent_metadata(self):
            return b"agent-meta"

        def register_memory(self, tensor):
            return ("reg", tuple(tensor.shape))

    monkeypatch.setattr(nixl_mod, "_HAVE_NIXL", True)
    monkeypatch.setattr(nixl_mod, "nixl_agent", FakeAgent)
    monkeypatch.setattr(nixl_mod, "nixl_agent_config", FakeAgentConfig)

    backend = nixl_mod.NixlTransferBackend(
        "prefill", torch.zeros(2, 3, 5, dtype=torch.float32), expected_num_blocks=3
    )
    metadata = backend.export_meta()

    assert metadata["agent_name"] == "prefill"
    assert metadata["bytes_per_slice"] == 20
    assert metadata["num_outer"] == 2
    assert metadata["num_blocks"] == 3
    assert metadata["blocks_axis"] == 1
    assert backend._agent.config.enable_prog_thread is True


def test_nixl_registered_buffers_share_agent_and_peer_cache(monkeypatch):
    from megatron.core.inference.disaggregation.transfer_backends import nixl as nixl_mod

    agents = []

    class FakeAgent:
        def __init__(self, name, config):
            self.registrations = []
            agents.append(self)

        def get_agent_metadata(self):
            return f"registrations={len(self.registrations)}".encode()

        def register_memory(self, tensor):
            self.registrations.append(tensor)
            return ("reg", len(self.registrations))

    monkeypatch.setattr(nixl_mod, "_HAVE_NIXL", True)
    monkeypatch.setattr(nixl_mod, "nixl_agent", FakeAgent)
    monkeypatch.setattr(nixl_mod, "nixl_agent_config", lambda **_: object())

    backend = nixl_mod.NixlTransferBackend(
        "prefill", torch.zeros(2, 3, 5, dtype=torch.float32), expected_num_blocks=3
    )
    sibling = backend.new_registered_buffer(
        agent_name="prefill-mamba-conv",
        memory_buffer=torch.zeros(2, 4, 5, dtype=torch.float32),
        expected_num_blocks=4,
    )

    assert len(agents) == 1
    assert sibling._agent is backend._agent
    assert sibling._known_peers is backend._known_peers
    assert sibling.export_meta()["agent_name"] == "prefill"
    assert backend.export_meta()["agent_metadata_b64"] == "cmVnaXN0cmF0aW9ucz0y"


def test_nixl_empty_pipeline_pull_is_a_completed_noop(monkeypatch):
    from megatron.core.inference.disaggregation.transfer_backends import nixl as nixl_mod

    class FakeAgent:
        def __init__(self, name, config):
            pass

        def register_memory(self, tensor):
            return "reg"

    monkeypatch.setattr(nixl_mod, "_HAVE_NIXL", True)
    monkeypatch.setattr(nixl_mod, "nixl_agent", FakeAgent)
    monkeypatch.setattr(nixl_mod, "nixl_agent_config", lambda **_: object())

    backend = nixl_mod.NixlTransferBackend(
        "decode", torch.zeros(2, 3, 5, dtype=torch.float32), expected_num_blocks=3
    )
    handle = backend.begin_pull_blocks({"pp_metas": [{"block_ids": []}]}, [], [])

    assert handle.poll()


def test_nixl_begin_pull_blocks_uses_remote_metadata_with_fake_agent(monkeypatch):
    from megatron.core.inference.disaggregation.transfer_backends import nixl as nixl_mod

    class FakeAgent:
        def __init__(self, name, config):
            self.name = name
            self.transferred = False

        def get_agent_metadata(self):
            return b"local"

        def register_memory(self, tensor):
            return ("reg", tuple(tensor.shape))

        def add_remote_agent(self, metadata):
            assert metadata == b"remote"
            return "peer"

        def get_xfer_descs(self, tuples, mem_type):
            assert mem_type == "VRAM"
            return tuples

        def initialize_xfer(self, op, local_desc, remote_desc, peer_id):
            assert op == "READ"
            assert peer_id == "peer"
            return (local_desc, remote_desc)

        def transfer(self, xfer):
            self.transferred = True

        def check_xfer_state(self, xfer):
            assert self.transferred
            return "DONE"

    monkeypatch.setattr(nixl_mod, "_HAVE_NIXL", True)
    monkeypatch.setattr(nixl_mod, "nixl_agent", FakeAgent)
    monkeypatch.setattr(nixl_mod, "nixl_agent_config", lambda **_: object())

    backend = nixl_mod.NixlTransferBackend(
        "decode", torch.zeros(2, 3, 5, dtype=torch.float32), expected_num_blocks=3
    )
    peer_meta = {
        "agent_name": "prefill",
        "agent_metadata_b64": "cmVtb3Rl",
        "base_addr": 1234,
        "bytes_per_slice": 20,
        "num_outer": 2,
        "outer_stride_bytes": 60,
        "num_blocks": 3,
        "device_id": 0,
        "blocks_axis": 1,
    }
    backend.begin_pull_blocks(peer_meta, [1], [2]).wait()

    assert backend._agent.transferred is True


def test_nixl_begin_pull_blocks_returns_pollable_handle(monkeypatch):
    from megatron.core.inference.disaggregation.transfer_backends import nixl as nixl_mod

    class FakeAgent:
        def __init__(self, name, config):
            self.name = name
            self.transfers = 0
            self.polls = 0

        def get_agent_metadata(self):
            return b"local"

        def register_memory(self, tensor):
            return ("reg", tuple(tensor.shape))

        def add_remote_agent(self, metadata):
            assert metadata == b"remote"
            return "peer"

        def get_xfer_descs(self, tuples, mem_type):
            assert mem_type == "VRAM"
            return tuples

        def initialize_xfer(self, op, local_desc, remote_desc, peer_id):
            assert op == "READ"
            assert peer_id == "peer"
            return {"local": local_desc, "remote": remote_desc}

        def transfer(self, xfer):
            self.transfers += 1

        def check_xfer_state(self, xfer):
            self.polls += 1
            return "DONE" if self.polls >= 2 else "PENDING"

    monkeypatch.setattr(nixl_mod, "_HAVE_NIXL", True)
    monkeypatch.setattr(nixl_mod, "nixl_agent", FakeAgent)
    monkeypatch.setattr(nixl_mod, "nixl_agent_config", lambda **_: object())

    backend = nixl_mod.NixlTransferBackend(
        "decode", torch.zeros(2, 3, 5, dtype=torch.float32), expected_num_blocks=3
    )
    peer_meta = {
        "agent_name": "prefill",
        "agent_metadata_b64": "cmVtb3Rl",
        "base_addr": 1234,
        "bytes_per_slice": 20,
        "num_outer": 2,
        "outer_stride_bytes": 60,
        "num_blocks": 3,
        "device_id": 0,
        "blocks_axis": 1,
    }

    handle = backend.begin_pull_blocks(peer_meta, [1], [2])

    assert backend._agent.transfers == 1
    assert backend._agent.polls == 0
    assert handle.poll() is False
    assert handle.poll() is True
