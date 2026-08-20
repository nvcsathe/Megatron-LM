# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.inference.moe.trtllm_finalize import HAVE_TRITON, padding_aware_finalize

requires_triton_cuda = pytest.mark.skipif(
    not HAVE_TRITON or not torch.cuda.is_available(),
    reason="Padding-aware TRT-LLM finalization requires Triton and CUDA.",
)


def _reference_finalize(
    gemm2_output, expert_weights, inverse_map, output, valid_tokens, round_to_bf16
):
    reference = output.clone()
    for token in range(valid_tokens):
        accumulator = torch.zeros(output.shape[1], dtype=torch.float32, device=output.device)
        for top_k_idx in range(inverse_map.shape[1]):
            mapped_row = int(inverse_map[token, top_k_idx].item())
            if 0 <= mapped_row < gemm2_output.shape[0]:
                accumulator += (
                    gemm2_output[mapped_row].float() * expert_weights[token, top_k_idx].float()
                )
        if round_to_bf16:
            accumulator = accumulator.bfloat16().float()
        reference[token] = accumulator
    return reference


@pytest.mark.internal
@requires_triton_cuda
@pytest.mark.parametrize("round_to_bf16", [False, True])
def test_padding_aware_finalize_skips_invalid_pairs_and_trailing_capacity(round_to_bf16):
    torch.manual_seed(123)
    max_tokens = 7
    valid_token_count = 5
    hidden_size = 257
    top_k = 4
    gemm2_rows = 11

    gemm2_output = torch.randn(gemm2_rows, hidden_size, dtype=torch.bfloat16, device="cuda")
    expert_weights = torch.randn(max_tokens, top_k, dtype=torch.float32, device="cuda")
    inverse_map = torch.tensor(
        [
            [0, 1, -1, -1],
            [-1, -1, -1, -1],
            [2, -1, 3, -1],
            [4, 5, 6, 7],
            [-1, -1, -1, -1],
            [8, 9, 10, 0],
            [1, 2, 3, 4],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    valid_tokens = torch.tensor([valid_token_count], dtype=torch.int32, device="cuda")
    output = torch.full((max_tokens, hidden_size), 123.0, dtype=torch.float32, device="cuda")
    reference = _reference_finalize(
        gemm2_output, expert_weights, inverse_map, output, valid_token_count, round_to_bf16
    )

    padding_aware_finalize(
        gemm2_output,
        expert_weights,
        inverse_map,
        output,
        valid_tokens,
        top_k,
        round_to_bf16=round_to_bf16,
    )

    torch.testing.assert_close(output, reference, atol=1e-5, rtol=1e-5)
    assert torch.count_nonzero(output[1]) == 0
    assert torch.count_nonzero(output[4]) == 0
    torch.testing.assert_close(output[valid_token_count:], reference[valid_token_count:])


@pytest.mark.internal
@requires_triton_cuda
def test_padding_aware_finalize_cuda_graph_overwrites_stale_invalid_rows():
    max_tokens = 4
    hidden_size = 128
    top_k = 2
    gemm2_output = torch.ones(4, hidden_size, dtype=torch.bfloat16, device="cuda")
    expert_weights = torch.ones(max_tokens, top_k, dtype=torch.float32, device="cuda")
    inverse_map = torch.tensor(
        [[0, 1], [2, -1], [3, -1], [-1, -1]], dtype=torch.int32, device="cuda"
    )
    valid_tokens = torch.tensor([max_tokens], dtype=torch.int32, device="cuda")
    output = torch.empty(max_tokens, hidden_size, dtype=torch.float32, device="cuda")

    # Compile before capture so graph construction does not include Triton JIT work.
    padding_aware_finalize(gemm2_output, expert_weights, inverse_map, output, valid_tokens, top_k)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        padding_aware_finalize(
            gemm2_output, expert_weights, inverse_map, output, valid_tokens, top_k
        )

    inverse_map.fill_(-1)
    output.fill_(123.0)
    graph.replay()
    torch.cuda.synchronize()

    assert torch.count_nonzero(output) == 0
