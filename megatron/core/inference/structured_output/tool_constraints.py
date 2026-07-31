# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import copy
from typing import Any, Dict, List, Optional


def _validate_tools(tools: Any) -> List[Dict[str, Any]]:
    """Validate and copy OpenAI function tools without weakening their schemas."""
    if not isinstance(tools, list) or not tools:
        raise ValueError("tool_choice 'required' or a named tool requires a non-empty tools list")

    normalized_tools = []
    function_names = set()
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ValueError(f"tools[{index}] must be an OpenAI function tool")
        function = tool.get("function")
        if not isinstance(function, dict):
            raise ValueError(f"tools[{index}].function must be an object")
        function_name = function.get("name")
        if not isinstance(function_name, str) or not function_name:
            raise ValueError(f"tools[{index}].function.name must be a non-empty string")
        if function_name in function_names:
            raise ValueError(f"duplicate tool function name: {function_name}")
        function_names.add(function_name)

        parameters = function.get("parameters")
        if not isinstance(parameters, dict):
            raise ValueError(
                f"tools[{index}].function.parameters must be a JSON Schema object"
            )

        normalized_tool = copy.deepcopy(tool)
        # vLLM guarantees schema-conforming output for required and named tool
        # choices. XGrammar uses the strict flag to select that behavior.
        normalized_tool["function"]["strict"] = True
        normalized_tools.append(normalized_tool)

    return normalized_tools


def build_tool_constraint(
    tools: Any,
    tool_choice: Any,
    *,
    reasoning: bool = True,
    parallel_tool_calls: bool = True,
) -> Optional[Dict[str, Any]]:
    """Build a serializable Qwen3-Coder constraint for required/named tool use.

    ``auto`` remains parser-driven in this initial implementation. Required and
    named choices are constrained to the complete tool schemas, matching the
    modes for which vLLM guarantees schema-conforming tool arguments.
    """
    is_named = isinstance(tool_choice, dict) and tool_choice.get("type") == "function"
    if tool_choice != "required" and not is_named:
        return None

    normalized_tools = _validate_tools(tools)
    if is_named:
        function = tool_choice.get("function")
        function_name = function.get("name") if isinstance(function, dict) else None
        if not isinstance(function_name, str) or not function_name:
            raise ValueError("named tool_choice.function.name must be a non-empty string")
        available_names = {tool["function"]["name"] for tool in normalized_tools}
        if function_name not in available_names:
            raise ValueError(f"named tool_choice references unknown function: {function_name}")

    return {
        "backend": "xgrammar",
        "model": "qwen_3_coder",
        "tools": normalized_tools,
        "tool_choice": copy.deepcopy(tool_choice),
        "reasoning": bool(reasoning),
        "parallel_tool_calls": bool(parallel_tool_calls),
    }
