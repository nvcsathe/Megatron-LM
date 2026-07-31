# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import copy
import sys
from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.structured_output.tool_constraints import build_tool_constraint
from megatron.core.inference.structured_output.xgrammar_manager import XGrammarConstraintManager


@pytest.fixture
def tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run a shell command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["command", "description"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "todowrite",
                "description": "Replace the todo list",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "todos": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "content": {"type": "string"},
                                    "status": {"type": "string"},
                                },
                                "required": ["content", "status"],
                            },
                        }
                    },
                    "required": ["todos"],
                },
            },
        },
    ]


def test_required_tool_constraint_preserves_complete_schemas(tools):
    original_tools = copy.deepcopy(tools)

    constraint = build_tool_constraint(tools, "required")

    assert constraint["model"] == "qwen_3_coder"
    assert constraint["tool_choice"] == "required"
    assert constraint["tools"][0]["function"]["parameters"]["required"] == [
        "command",
        "description",
    ]
    assert (
        constraint["tools"][1]["function"]["parameters"]["properties"]["todos"]["type"]
        == "array"
    )
    assert all(tool["function"]["strict"] for tool in constraint["tools"])
    assert tools == original_tools


def test_named_tool_constraint_preserves_choice(tools):
    tool_choice = {"type": "function", "function": {"name": "bash"}}

    constraint = build_tool_constraint(
        tools, tool_choice, reasoning=False, parallel_tool_calls=False
    )

    assert constraint["tool_choice"] == tool_choice
    assert constraint["reasoning"] is False
    assert constraint["parallel_tool_calls"] is False
    assert [tool["function"]["name"] for tool in constraint["tools"]] == [
        "bash",
        "todowrite",
    ]


@pytest.mark.parametrize("tool_choice", [None, "none", "auto"])
def test_unforced_tool_choice_does_not_enable_constraints(tools, tool_choice):
    assert build_tool_constraint(tools, tool_choice) is None


def test_named_tool_constraint_rejects_unknown_function(tools):
    with pytest.raises(ValueError, match="unknown function"):
        build_tool_constraint(
            tools, {"type": "function", "function": {"name": "missing"}}
        )


def test_required_tool_constraint_rejects_missing_parameter_schema():
    tools = [{"type": "function", "function": {"name": "bash"}}]

    with pytest.raises(ValueError, match="parameters must be a JSON Schema object"):
        build_tool_constraint(tools, "required")


def test_xgrammar_manager_caches_grammar_and_uses_strict_property_order(monkeypatch, tools):
    structural_tag_calls = []

    class FakeTokenizerInfo:
        @staticmethod
        def from_huggingface(tokenizer, vocab_size):
            return (tokenizer, vocab_size)

    class FakeGrammarCompiler:
        compile_count = 0

        def __init__(self, tokenizer_info, cache_enabled):
            self.tokenizer_info = tokenizer_info
            self.cache_enabled = cache_enabled

        def compile_structural_tag(self, structural_tag):
            FakeGrammarCompiler.compile_count += 1
            return structural_tag

    class FakeGrammarMatcher:
        def __init__(self, compiled_grammar):
            self.compiled_grammar = compiled_grammar

    def get_model_structural_tag(model, **kwargs):
        structural_tag_calls.append((model, kwargs))
        return (model, kwargs)

    fake_xgrammar = SimpleNamespace(
        TokenizerInfo=FakeTokenizerInfo,
        GrammarCompiler=FakeGrammarCompiler,
        GrammarMatcher=FakeGrammarMatcher,
        allocate_token_bitmask=lambda batch_size, vocab_size: torch.empty(
            (batch_size, (vocab_size + 31) // 32), dtype=torch.int32
        ),
        get_model_structural_tag=get_model_structural_tag,
    )
    monkeypatch.setitem(sys.modules, "xgrammar", fake_xgrammar)

    constraint = build_tool_constraint(tools, "required")
    manager = XGrammarConstraintManager(SimpleNamespace(), vocab_size=64, max_requests=4)
    manager.initialize_request(10, constraint)
    manager.initialize_request(11, constraint)

    assert FakeGrammarCompiler.compile_count == 1
    assert structural_tag_calls[0][0] == "qwen_3_coder"
    assert structural_tag_calls[0][1]["tool_choice"] == "required"
    assert structural_tag_calls[0][1]["any_order"] is False
    assert manager.has_requests

    manager.finish_requests([10, 11])
    assert not manager.has_requests


def test_xgrammar_manager_rejects_speculative_decoding_before_backend_import(tools):
    constraint = build_tool_constraint(tools, "required")
    manager = XGrammarConstraintManager(SimpleNamespace(), vocab_size=64, max_requests=4)

    with pytest.raises(ValueError, match="speculative/MTP"):
        manager.initialize_request(10, constraint, num_speculative_tokens=1)
