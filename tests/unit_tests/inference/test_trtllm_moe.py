# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for TensorRT-LLM fused-MoE integration helpers."""

import pytest
import torch

from megatron.core.inference.moe.trtllm import HAVE_TRITON, pack_topk_ids


@pytest.mark.skipif(
    not HAVE_TRITON or not torch.cuda.is_available(), reason="Triton and CUDA are required"
)
def test_pack_topk_ids_matches_flashinfer_packed_routing_abi():
    routing_map = torch.tensor([[0, 1, 127], [4, 16, 255]], dtype=torch.int64, device="cuda")
    probs = torch.tensor([[0.0, 0.25, 0.5], [0.75, 0.9, 1.0]], dtype=torch.float32, device="cuda")
    expected = (routing_map.to(torch.int32) << 16) | (
        probs.to(torch.bfloat16).view(torch.int16).to(torch.int32)
    )

    actual = pack_topk_ids(routing_map, probs)

    torch.testing.assert_close(actual, expected)
