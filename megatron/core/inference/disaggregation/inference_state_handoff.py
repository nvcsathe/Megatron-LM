# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Engine-side lifecycle for disaggregated prefill/decode state handoff."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Any, Dict, Optional

import torch

from megatron.core.inference.disaggregation.pending_handoff_imports import (
    PendingKvImport,
    PendingMambaImport,
)
from megatron.core.utils import get_pg_rank, get_pg_size


class InferenceStateHandoffMixin:
    """Optional KV/Mamba handoff behavior composed into the dynamic engine."""

    def _initialize_disaggregation_state(self) -> None:
        """Initialize state without importing or constructing a transfer backend."""

        self._pinned_handoff_blocks: Dict[int, list] = {}
        self._kv_transfer_agent = None
        self._kv_peer_metas = None
        self._mamba_conv_agent = None
        self._mamba_ssm_agent = None
        self._pending_kv_imports = deque()
        self.role = "aggregated"

    @property
    def pinned_handoff_count(self) -> int:
        """Number of completed requests whose source blocks remain pinned."""

        return len(self._pinned_handoff_blocks)

    @property
    def pending_kv_import_count(self) -> int:
        """Number of decode requests waiting for state transfer completion."""

        return len(self._pending_kv_imports)

    @property
    def has_pending_kv_imports(self) -> bool:
        """Whether any decode request is waiting for imported state."""

        return bool(self._pending_kv_imports)

    def _reset_pending_kv_imports(self) -> None:
        """Clear pending import bookkeeping during an engine reset."""

        self._pending_kv_imports = deque()

    def setup_kv_transfer(self, role: str, listen_addr: Optional[str]) -> None:
        """Bring up the NIXL transfer agent for this engine, if configured.

        Args:
            role: "prefill" or "decode"; used to name the local NIXL agent.
            listen_addr: ``host:port`` for the NIXL agent. ``None`` disables
                KV transfer.
        """
        self.role = role
        if not listen_addr:
            return
        from megatron.core.inference.disaggregation.transfer_backends.nixl import make_agent

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        # TP topology, so a peer at a different TP can re-shard our KV heads.
        # KV heads under GQA == num_query_groups (falls back to attention heads).
        model_config = self.controller.inference_wrapped_model.model.config
        num_kv_heads_global = (
            model_config.num_query_groups or model_config.num_attention_heads
        )
        tp_size = get_pg_size(self.pg_collection.tp)
        tp_rank = get_pg_rank(self.pg_collection.tp)

        # Compute this PP rank's global attention-layer range.
        pp_size = get_pg_size(self.pg_collection.pp)
        pp_rank = get_pg_rank(self.pg_collection.pp)
        local_num_layers = self.context.num_attention_layers

        if pp_size > 1 and torch.distributed.is_initialized():
            layer_counts: list = [None] * pp_size
            torch.distributed.all_gather_object(
                layer_counts, local_num_layers, group=self.pg_collection.pp
            )
            layer_start = sum(layer_counts[:pp_rank])
        else:
            layer_start = 0
        layer_end = layer_start + local_num_layers

        self._kv_transfer_agent = make_agent(
            role=role,
            rank=rank,
            listen_addr=listen_addr,
            memory_buffer=self.context.memory_buffer,
            expected_num_blocks=self.context.kv_block_allocator.total_count,
            tp_size=tp_size,
            tp_rank=tp_rank,
            num_kv_heads_global=num_kv_heads_global,
            heads_per_partition=self.context.num_attention_heads_per_partition,
            head_dim=self.context.hidden_size_per_attention_head,
            tokens_per_block=self.context.block_size_tokens,
            pp_rank=pp_rank,
            layer_start=layer_start,
            layer_end=layer_end,
        )

        # Cache peer metadata for cross-TP pulls.
        self._kv_peer_metas = None
        if self._kv_transfer_agent is not None and torch.distributed.is_initialized() and tp_size > 1:
            local_meta = self._kv_transfer_agent.export_meta()
            gathered: list = [None] * tp_size
            torch.distributed.all_gather_object(
                gathered, local_meta, group=self.pg_collection.tp
            )
            self._kv_peer_metas = gathered

        # Mamba state transfer currently requires a matched layout.
        self._mamba_conv_agent = None
        self._mamba_ssm_agent = None
        if getattr(self.context, "is_hybrid_model", False):
            if tp_size > 1 or pp_size > 1:
                raise NotImplementedError(
                    "Disaggregated KV handoff for hybrid (Mamba) models "
                    f"currently supports only TP=1/PP=1 (got tp={tp_size}, "
                    f"pp={pp_size}). Mamba conv/ssm state re-sharding across "
                    "TP/PP is not implemented. Run prefill and decode at "
                    "TP=1 PP=1, or disable KV transfer for this model."
                )
            msa = getattr(self.context, "mamba_slot_allocator", None)
            if msa is None:
                raise RuntimeError(
                    "Hybrid model KV handoff requires the Mamba state cache. "
                    "Pass --inference-dynamic-batching-prefix-caching and "
                    "--inference-dynamic-batching-prefix-caching-mamba-gb <GB> "
                    "so the decode engine can restore transferred Mamba state."
                )
            self._mamba_conv_agent = make_agent(
                role=f"{role}-mamba-conv",
                rank=rank,
                listen_addr=listen_addr,
                memory_buffer=msa.conv_states,
                expected_num_blocks=msa.max_slots,
            )
            self._mamba_ssm_agent = make_agent(
                role=f"{role}-mamba-ssm",
                rank=rank,
                listen_addr=listen_addr,
                memory_buffer=msa.ssm_states,
                expected_num_blocks=msa.max_slots,
            )

    def _capture_handoff_meta(
        self, request: "DynamicInferenceRequest", block_ids: list
    ) -> None:
        """Attach transfer metadata and retain the request's pinned blocks."""
        rid = request.request_id
        if not block_ids:
            logging.warning(
                "DISAGG_PREFILL_HANDOFF request_id=%d had no snapshot blocks "
                "(controller missed the slot?); decode peer will receive empty handoff",
                rid,
            )
            return

        self._pinned_handoff_blocks[rid] = list(block_ids)

        local_kv: Any = {}
        if self._kv_peer_metas is not None:
            local_kv = self._kv_peer_metas
        elif self._kv_transfer_agent is not None:
            local_kv = self._kv_transfer_agent.export_meta()

        pp_size = get_pg_size(self.pg_collection.pp)
        if pp_size > 1 and torch.distributed.is_initialized():
            local_entry = {"kv_meta": local_kv, "block_ids": list(block_ids)}
            gathered: list = [None] * pp_size
            torch.distributed.all_gather_object(
                gathered, local_entry, group=self.pg_collection.pp
            )
            kv_meta: Any = {
                "pp_metas": [
                    {"tp_metas": e["kv_meta"], "block_ids": e["block_ids"]}
                    for e in gathered
                ]
            }
            top_block_ids: Any = gathered[0]["block_ids"]
        else:
            kv_meta = local_kv
            top_block_ids = block_ids

        if self._mamba_conv_agent is not None and isinstance(kv_meta, dict):
            msa = self.context.mamba_slot_allocator
            mamba_blocks = []
            for pos, b in enumerate(block_ids):
                slot = msa.get_slot(int(b))
                if slot >= 0:
                    mamba_blocks.append([pos, int(slot)])
            kv_meta["mamba"] = {
                "conv": self._mamba_conv_agent.export_meta(),
                "ssm": self._mamba_ssm_agent.export_meta(),
                "blocks": mamba_blocks,
            }

        request.disaggregated_params = {
            "request_id": rid,
            "block_ids": top_block_ids,
            "kv_meta": kv_meta,
        }
        logging.info(
            "DISAGG_PREFILL_HANDOFF request_id=%d pinned_blocks=%d mamba_blocks=%d",
            rid,
            len(block_ids),
            len(kv_meta.get("mamba", {}).get("blocks", [])) if isinstance(kv_meta, dict) else 0,
        )

    def release_handoff_blocks(self, request_id: int) -> None:
        """Release blocks pinned by a previous do_kv_handoff completion."""
        block_ids = self._pinned_handoff_blocks.pop(request_id, None)
        if not block_ids:
            return
        released = self._release_pinned_handoff_blocks(block_ids)
        logging.info(
            "DISAGG_PREFILL_RELEASE request_id=%d released_blocks=%d",
            request_id,
            released,
        )

    def _release_pinned_handoff_blocks(self, block_ids: list) -> int:
        """Release this request's ownership of its pinned handoff blocks."""
        allocator = self.context.kv_block_allocator
        return allocator.release_pinned_memory_blocks(block_ids)

    def add_request_with_kv_handoff(
        self,
        request_id: int,
        prompt: list,
        sampling_params: "SamplingParams",
        kv_meta: dict,
        src_block_ids: list,
    ) -> "asyncio.Future[DynamicInferenceRequest]":
        """Submit an async NIXL pull, then admit the request when KV is local."""
        from megatron.core.inference.inference_request import compute_block_hashes_batched

        allocator = self.context.kv_block_allocator
        if not allocator.enable_prefix_caching:
            raise RuntimeError(
                "add_request_with_kv_handoff requires --enable-prefix-caching on the "
                "decode engine; the prefill-skip path uses the prefix-cache match logic."
            )

        # PP entries all have the same sequence-block count.
        if isinstance(kv_meta, dict) and "pp_metas" in kv_meta:
            pp_metas = kv_meta["pp_metas"]
            num_blocks = len(pp_metas[0]["block_ids"]) if pp_metas else 0
        else:
            num_blocks = len(src_block_ids)
        local_blocks_tensor = allocator.allocate_memory_blocks(num_blocks)
        if local_blocks_tensor is None:
            raise RuntimeError(
                f"add_request_with_kv_handoff: OOM allocating {num_blocks} blocks"
            )
        local_blocks = [int(b) for b in local_blocks_tensor.tolist()]

        if self._kv_transfer_agent is not None and kv_meta:
            handle = self._kv_transfer_agent.begin_pull_blocks(
                kv_meta, src_block_ids, local_blocks
            )
        else:
            handle = None

        # Register hashes only after transfer; include short partial blocks.
        prompt_tensor = torch.tensor(prompt, dtype=torch.int64)
        hashes = compute_block_hashes_batched(
            prompt_tensor, self.context.block_size_tokens, include_partial=True
        )
        hashes_to_register = min(num_blocks, len(hashes))

        mamba_import = None
        mamba_meta = kv_meta.get("mamba") if isinstance(kv_meta, dict) else None
        if mamba_meta:
            mamba_import = self._begin_mamba_handoff_import(
                request_id, mamba_meta, local_blocks, hashes
            )

        future = self._loop.create_future()
        pending = PendingKvImport(
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling_params,
            local_blocks=local_blocks,
            hashes=hashes,
            hashes_to_register=hashes_to_register,
            handle=handle,
            future=future,
            mamba=mamba_import,
        )
        self._pending_kv_imports.append(pending)
        logging.info(
            "DISAGG_DECODE_PULL_SUBMIT request_id=%d prompt_tokens=%d blocks=%d pending_imports=%d",
            request_id,
            len(prompt),
            num_blocks,
            len(self._pending_kv_imports),
        )
        self._loop.call_soon_threadsafe(
            self._loop.create_task, self._notify_cond_for_new_request()
        )
        return future

    def _kv_import_done(self, pending: PendingKvImport) -> bool:
        if pending.handle is not None and not pending.handle.poll():
            return False
        if pending.mamba is not None:
            if not pending.mamba.conv_handle.poll():
                return False
            if not pending.mamba.ssm_handle.poll():
                return False
        return True

    def _finalize_kv_handoff_import(self, pending: PendingKvImport) -> None:
        allocator = self.context.kv_block_allocator
        n = pending.hashes_to_register
        local_blocks = pending.local_blocks

        if n > 0:
            allocator.register_kv_block_hashes(local_blocks[:n], pending.hashes[:n])

        local_blocks_idx = torch.tensor(local_blocks, dtype=torch.int64)
        allocator.block_ref_counts[local_blocks_idx] -= 1

        if pending.mamba is not None:
            self._complete_mamba_handoff_import(pending.request_id, pending.mamba, pending.hashes)

        logging.info(
            "DISAGG_DECODE_IMPORT request_id=%d prompt_tokens=%d imported_blocks=%d hashes_registered=%d pending_imports=%d",
            pending.request_id,
            len(pending.prompt),
            len(local_blocks),
            n,
            len(self._pending_kv_imports),
        )
        request_future = self.add_request(
            pending.request_id,
            pending.prompt,
            pending.sampling_params,
            precomputed_block_hashes=pending.hashes[:n] if n > 0 else None,
        )

        def _relay_result(src: asyncio.Future) -> None:
            if pending.future.done():
                return
            if src.cancelled():
                pending.future.cancel()
                return
            exc = src.exception()
            if exc is not None:
                pending.future.set_exception(exc)
            else:
                pending.future.set_result(src.result())

        request_future.add_done_callback(_relay_result)

    def _release_pending_kv_import(self, pending: PendingKvImport) -> None:
        if pending.local_blocks:
            block_tensor = torch.tensor(pending.local_blocks, dtype=torch.int32, device='cpu')
            self.context.kv_block_allocator.release_memory_blocks(block_tensor)
        if pending.mamba is not None:
            msa = self.context.mamba_slot_allocator
            if msa is not None:
                for block_id in pending.mamba.target_blocks:
                    msa.invalidate_block(int(block_id))

    def _poll_pending_kv_imports(self) -> int:
        if not self._pending_kv_imports:
            return 0
        ready = 0
        remaining = deque()
        while self._pending_kv_imports:
            pending = self._pending_kv_imports.popleft()
            try:
                if self._kv_import_done(pending):
                    self._finalize_kv_handoff_import(pending)
                    ready += 1
                else:
                    remaining.append(pending)
            except Exception as exc:
                self._release_pending_kv_import(pending)
                if not pending.future.done():
                    pending.future.set_exception(exc)
                logging.exception(
                    "DISAGG_DECODE_PULL_FAILED request_id=%d", pending.request_id
                )
                raise
        self._pending_kv_imports = remaining
        if ready:
            self._loop.call_soon_threadsafe(
                self._loop.create_task, self._notify_cond_for_new_request()
            )
        return ready

    def _begin_mamba_handoff_import(
        self,
        request_id: int,
        mamba_meta: dict,
        local_blocks: list,
        hashes: list,
    ) -> Optional[PendingMambaImport]:
        pairs = mamba_meta.get("blocks") or []
        if not pairs:
            return None
        if self._mamba_conv_agent is None or self._mamba_ssm_agent is None:
            raise RuntimeError(
                "Received Mamba handoff state but this decode engine has no "
                "Mamba transfer agents. Ensure it runs a hybrid model with "
                "--inference-dynamic-batching-prefix-caching-mamba-gb set and "
                "KV transfer enabled (matched TP=1/PP=1)."
            )
        msa = self.context.mamba_slot_allocator
        if msa is None:
            raise RuntimeError(
                "Mamba handoff requires the decode engine's Mamba state cache; "
                "pass --inference-dynamic-batching-prefix-caching-mamba-gb."
            )

        positions = [int(p) for p, _ in pairs]
        src_slots = [int(s) for _, s in pairs]
        target_blocks = [int(local_blocks[p]) for p in positions]
        local_slots = msa.allocate_slots_batch(target_blocks)
        conv_handle = self._mamba_conv_agent.begin_pull_blocks(
            mamba_meta["conv"], src_slots, local_slots
        )
        ssm_handle = self._mamba_ssm_agent.begin_pull_blocks(
            mamba_meta["ssm"], src_slots, local_slots
        )
        logging.info(
            "DISAGG_DECODE_MAMBA_IMPORT_SUBMIT request_id=%d mamba_blocks=%d",
            request_id,
            len(target_blocks),
        )
        return PendingMambaImport(
            conv_handle=conv_handle,
            ssm_handle=ssm_handle,
            target_blocks=target_blocks,
            positions=positions,
        )

    def _complete_mamba_handoff_import(
        self,
        request_id: int,
        pending: PendingMambaImport,
        hashes: list,
    ) -> None:
        msa = self.context.mamba_slot_allocator
        if msa is None:
            raise RuntimeError("Mamba handoff completed but the decode cache is unavailable.")
        msa.register_block_hashes_batch(
            pending.target_blocks, [hashes[p] for p in pending.positions]
        )
        logging.info(
            "DISAGG_DECODE_MAMBA_IMPORT request_id=%d mamba_blocks=%d",
            request_id,
            len(pending.target_blocks),
        )

    def _import_mamba_handoff(
        self,
        request_id: int,
        mamba_meta: dict,
        local_blocks: list,
        hashes: list,
    ) -> None:
        """Synchronously pull the prefill peer's Mamba state."""
        pending = self._begin_mamba_handoff_import(request_id, mamba_meta, local_blocks, hashes)
        if pending is None:
            return
        pending.conv_handle.wait()
        pending.ssm_handle.wait()
        self._complete_mamba_handoff_import(request_id, pending, hashes)
