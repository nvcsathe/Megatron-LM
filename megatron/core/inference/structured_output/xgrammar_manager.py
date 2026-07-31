# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from typing import Any, Dict, Iterable, Optional, Sequence

import torch
from torch import Tensor


class XGrammarConstraintManager:
    """Compile and advance per-request Qwen tool-call grammars."""

    def __init__(self, tokenizer: Any, vocab_size: int, max_requests: int):
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.max_requests = max_requests
        self._xgrammar = None
        self._compiler = None
        self._token_bitmask = None
        self._compiled_grammars: Dict[str, Any] = {}
        self._matchers: Dict[int, Any] = {}

    @property
    def has_requests(self) -> bool:
        return bool(self._matchers)

    def _initialize_backend(self) -> None:
        if self._compiler is not None:
            return
        try:
            import xgrammar as xgr
        except ImportError as error:
            raise RuntimeError(
                "Schema-constrained tool calling requires xgrammar>=0.2.5. "
                "Install the Megatron Core inference dependencies with XGrammar enabled."
            ) from error

        tokenizer = getattr(self.tokenizer, "_tokenizer", self.tokenizer)
        tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        try:
            tokenizer_info = xgr.TokenizerInfo.from_huggingface(
                tokenizer, vocab_size=self.vocab_size
            )
        except Exception as error:
            raise ValueError(
                "Schema-constrained tool calling requires a Hugging Face compatible tokenizer"
            ) from error

        self._xgrammar = xgr
        self._compiler = xgr.GrammarCompiler(tokenizer_info, cache_enabled=True)
        self._token_bitmask = xgr.allocate_token_bitmask(self.max_requests, self.vocab_size)

    @staticmethod
    def _cache_key(spec: Dict[str, Any]) -> str:
        return json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def initialize_request(
        self, request_id: int, spec: Optional[Dict[str, Any]], num_speculative_tokens: int = 0
    ) -> None:
        if spec is None:
            return
        if not isinstance(spec, dict):
            raise ValueError("structured_output must be a dictionary")
        if spec.get("backend") != "xgrammar":
            raise ValueError(f"unsupported structured-output backend: {spec.get('backend')}")
        missing_fields = {
            "model",
            "tools",
            "tool_choice",
            "reasoning",
        } - spec.keys()
        if missing_fields:
            raise ValueError(
                "structured_output is missing required fields: "
                + ", ".join(sorted(missing_fields))
            )
        if num_speculative_tokens:
            raise ValueError(
                "Schema-constrained tool calling does not yet support speculative/MTP decoding"
            )

        self._initialize_backend()
        cache_key = self._cache_key(spec)
        compiled_grammar = self._compiled_grammars.get(cache_key)
        if compiled_grammar is None:
            try:
                structural_tag = self._xgrammar.get_model_structural_tag(
                    spec["model"],
                    tools=spec["tools"],
                    tool_choice=spec["tool_choice"],
                    reasoning=spec["reasoning"],
                    # XGrammar's any_order mode permits required properties to be
                    # omitted. Keep schema order so required keys and uniqueness
                    # are both enforced.
                    any_order=False,
                )
                compiled_grammar = self._compiler.compile_structural_tag(structural_tag)
            except Exception as error:
                raise ValueError(
                    f"Failed to compile structured-output grammar: {error}"
                ) from error
            self._compiled_grammars[cache_key] = compiled_grammar

        self._matchers[int(request_id)] = self._xgrammar.GrammarMatcher(compiled_grammar)

    @staticmethod
    def _as_int_list(values: Iterable[int]) -> Sequence[int]:
        if isinstance(values, Tensor):
            return values.tolist()
        return [int(value) for value in values]

    def apply_logits_mask(self, logits: Tensor, request_ids: Iterable[int]) -> None:
        """Mask invalid next tokens in request-aligned 2D logits."""
        if not self._matchers:
            return
        request_ids = self._as_int_list(request_ids)
        if logits.ndim != 2 or logits.shape[0] != len(request_ids):
            raise ValueError("structured-output logits must have one row per request")

        constrained_rows = []
        for row, request_id in enumerate(request_ids):
            matcher = self._matchers.get(request_id)
            if matcher is None:
                continue
            matcher.fill_next_token_bitmask(self._token_bitmask, row)
            constrained_rows.append(row)

        if not constrained_rows:
            return
        rows = torch.tensor(constrained_rows, dtype=torch.long)
        selected_mask = self._token_bitmask.index_select(0, rows).to(logits.device)
        selected_logits = logits.index_select(0, rows.to(logits.device))
        self._xgrammar.apply_token_bitmask_inplace(
            selected_logits, selected_mask, vocab_size=self.vocab_size
        )
        logits.index_copy_(0, rows.to(logits.device), selected_logits)

    def accept_tokens(self, request_ids: Iterable[int], token_ids: Iterable[int]) -> None:
        for request_id, token_id in zip(
            self._as_int_list(request_ids), self._as_int_list(token_ids)
        ):
            matcher = self._matchers.get(request_id)
            if matcher is not None and not matcher.accept_token(int(token_id)):
                raise RuntimeError(
                    f"XGrammar rejected sampled token {token_id} for request {request_id}"
                )

    def finish_requests(self, request_ids: Iterable[int]) -> None:
        for request_id in self._as_int_list(request_ids):
            self._matchers.pop(request_id, None)

    def reset(self) -> None:
        self._matchers.clear()
