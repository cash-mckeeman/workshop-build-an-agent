"""
Simple Tools for Agent Tutorial

This module provides basic tools that the agent can use to perform calculations
and demonstrate function calling capabilities.
"""

import json
from typing import Dict, List, Any


def add(a, b) -> int:
    """Add two integers."""
    return int(a) + int(b)


def multiply(a, b) -> int:
    """Multiply two integers."""
    return int(a) * int(b)


def subtract(a, b) -> int:
    """Subtract b from a."""
    return int(a) - int(b)


def get_tools_schema() -> List[Dict[str, Any]]:
    """
    Get the OpenAI-compatible tool schema for all available tools.

    Returns:
        List of tool schemas in OpenAI format
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two integers together",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer", "description": "First integer"},
                        "b": {"type": "integer", "description": "Second integer"},
                    },
                    "required": ["a", "b"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "multiply",
                "description": "Multiply two integers together",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer", "description": "First integer"},
                        "b": {"type": "integer", "description": "Second integer"},
                    },
                    "required": ["a", "b"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "subtract",
                "description": "Subtract the second integer from the first",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer", "description": "First integer (minuend)"},
                        "b": {"type": "integer", "description": "Second integer (subtrahend)"},
                    },
                    "required": ["a", "b"],
                },
            },
        },
    ]


def execute_tool(tool_name: str, tool_args: Dict[str, Any]) -> Any:
    """
    Execute a tool by name with the given arguments.

    Args:
        tool_name: Name of the tool to execute
        tool_args: Dictionary of arguments to pass to the tool

    Returns:
        Result of the tool execution

    Raises:
        ValueError: If tool name is not recognized
    """
    tools_map = {
        "add": add,
        "multiply": multiply,
        "subtract": subtract,
    }

    if tool_name not in tools_map:
        raise ValueError(f"Unknown tool: {tool_name}")

    return tools_map[tool_name](**tool_args)


def parse_tool_call(llm_response: Dict[str, Any]) -> tuple:
    """
    Parse tool call information from LLM response.

    Args:
        llm_response: Response from the LLM containing tool calls

    Returns:
        Tuple of (tool_name, tool_args, tool_id)

    Raises:
        ValueError: If no tool calls found or invalid format
    """
    if "tool_calls" not in llm_response or not llm_response["tool_calls"]:
        raise ValueError("No tool calls found in response")

    tool_call = llm_response["tool_calls"][0]
    tool_name = tool_call["function"]["name"]
    tool_id = tool_call["id"]

    # Parse arguments (might be string or dict)
    tool_args = tool_call["function"]["arguments"]
    if isinstance(tool_args, str):
        try:
            # Try JSON first (standard format)
            tool_args = json.loads(tool_args)
        except json.JSONDecodeError:
            try:
                # Fallback to eval for Python dict format
                tool_args = eval(tool_args)
            except:
                raise ValueError(f"Could not parse arguments: {tool_args}")

    return tool_name, tool_args, tool_id


def create_tool_result_message(tool_id: str, tool_name: str, tool_output: Any) -> Dict[str, Any]:
    """
    Create a properly formatted tool result message.

    Args:
        tool_id: ID of the tool call
        tool_name: Name of the tool that was executed
        tool_output: Result from tool execution

    Returns:
        Formatted message for adding to conversation history
    """
    return {
        "role": "tool",
        "tool_call_id": tool_id,
        "name": tool_name,
        "content": str(tool_output)
    }