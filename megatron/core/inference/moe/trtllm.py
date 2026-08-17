# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""TensorRT-LLM fused-MoE integration helpers."""

from unittest.mock import MagicMock

import torch

from megatron.core.utils import null_decorator

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

if not HAVE_TRITON:
    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()


@triton.jit
def _pack_topk_ids_kernel(
    routing_map_ptr, probs_ptr, packed_topk_ids_ptr, numel, BLOCK_SIZE: tl.constexpr
):
    """Pack expert IDs and BF16 routing-weight bits into TRTLLM's int32 routing ABI."""
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    expert_ids = tl.load(routing_map_ptr + offsets, mask=mask).to(tl.int32)
    probs = tl.load(probs_ptr + offsets, mask=mask)
    probs_bf16 = tl.cast(probs, tl.bfloat16, fp_downcast_rounding="rtne")
    prob_bits = tl.cast(probs_bf16, tl.uint16, bitcast=True).to(tl.int32)
    packed_topk_ids = (expert_ids << 16) | prob_bits

    tl.store(packed_topk_ids_ptr + offsets, packed_topk_ids, mask=mask)


def pack_topk_ids(routing_map: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """Pack precomputed routing into the legacy FlashInfer TRTLLM tensor ABI.

    This is equivalent to ``(routing_map.int() << 16) | bf16(probs).bits()`` but
    performs the conversion and packing in one GPU kernel instead of several eager
    PyTorch kernels.

    Args:
        routing_map: Contiguous ``[num_tokens, top_k]`` expert IDs.
        probs: Contiguous ``[num_tokens, top_k]`` FP32 routing weights.

    Returns:
        An int32 tensor with the expert ID in the upper 16 bits and the BF16
        routing weight in the lower 16 bits.
    """
    assert routing_map.shape == probs.shape
    assert routing_map.device == probs.device
    assert routing_map.is_contiguous()
    assert probs.is_contiguous()

    if not HAVE_TRITON:
        return (routing_map.to(torch.int32) << 16) | (
            probs.to(torch.bfloat16).view(torch.int16).to(torch.int32)
        )

    packed_topk_ids = torch.empty_like(routing_map, dtype=torch.int32)
    numel = routing_map.numel()
    if numel == 0:
        return packed_topk_ids

    block_size = 256
    _pack_topk_ids_kernel[(triton.cdiv(numel, block_size),)](
        routing_map, probs, packed_topk_ids, numel, BLOCK_SIZE=block_size
    )
    return packed_topk_ids
