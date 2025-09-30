"""
Helper utilities for creating PydanticAI agent tools

This module provides decorators and utilities to make it easier to convert
regular functions into PydanticAI agent tools, handling the RunContext requirement
automatically.
"""

import functools
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic_ai import RunContext


def with_run_context(func: Callable) -> Callable:
    """Decorator that adds RunContext as the first parameter to a function.

    This decorator allows you to convert regular functions into PydanticAI agent tools
    by automatically adding the required RunContext parameter.

    Args:
        func: The original function to wrap

    Returns:
        A wrapped function that accepts RunContext as its first parameter

    Example:
        @with_run_context
        def add(a: int, b: int) -> int:
            return a + b

        # Can now be used as:
        # @agent.tool
        # def my_add_tool(ctx: RunContext, a: int, b: int) -> int:
        #     return add(a, b)
    """

    @functools.wraps(func)
    def wrapper(ctx: RunContext, *args, **kwargs):
        # Call the original function without the RunContext
        return func(*args, **kwargs)

    # Create new annotations that include ctx: RunContext
    new_annotations = {"ctx": RunContext}
    if hasattr(func, "__annotations__"):
        new_annotations.update(func.__annotations__)
    wrapper.__annotations__ = new_annotations

    # Update the signature to include RunContext as the first parameter
    sig = inspect.signature(func)
    new_params = [
        inspect.Parameter(
            "ctx", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=RunContext
        )
    ]
    new_params.extend(sig.parameters.values())

    new_sig = sig.replace(parameters=new_params)
    wrapper.__signature__ = new_sig

    return wrapper


def register_tool_with_context(agent, func: Callable, name: str = None):
    """Register a tool with an agent, automatically adding RunContext.

    This is a convenience function that applies the with_run_context decorator
    and registers the tool with the agent in one step.

    Args:
        agent: The PydanticAI agent to register the tool with
        func: The function to register as a tool
        name: Optional name for the tool (defaults to function name)

    Returns:
        The wrapped and registered tool function
    """
    wrapped_func = with_run_context(func)

    # Register with agent using the decorator pattern
    tool_func = agent.tool(wrapped_func, name=name)

    return tool_func


def create_tool_wrapper(func: Callable) -> Callable:
    """Create a tool wrapper function that can be used directly with @agent.tool.

    This is similar to with_run_context but creates a new function that can be
    assigned to a variable and then used with @agent.tool.

    Args:
        func: The original function to wrap

    Returns:
        A wrapped function ready for use with @agent.tool

    Example:
        from src.tools.demo_tools import word_count
        from src.tools.agent_helpers import create_tool_wrapper

        count_words = create_tool_wrapper(word_count)

        @agent.tool
        def my_word_count_tool(ctx: RunContext, text: str) -> int:
            return count_words(ctx, text)
    """
    return with_run_context(func)


# Pre-wrapped tool functions for common use cases
def make_agent_tools(*functions):
    """Convert a list of functions into agent-ready tool functions.

    Args:
        *functions: Functions to convert

    Returns:
        Dictionary mapping function names to wrapped functions

    Example:
        from src.tools.demo_tools import word_count, reverse_string
        from src.tools.math_tools import add, multiply

        tools = make_agent_tools(word_count, reverse_string, add, multiply)

        @agent.tool
        def count_words_tool(ctx: RunContext, text: str) -> int:
            return tools['word_count'](ctx, text)
    """
    tool_dict = {}
    for func in functions:
        wrapped = with_run_context(func)
        tool_dict[func.__name__] = wrapped
    return tool_dict


# Convenience function for bulk tool registration
def register_multiple_tools(agent, tool_functions: dict, prefix: str = ""):
    """Register multiple tools with an agent at once.

    Args:
        agent: The PydanticAI agent
        tool_functions: Dict of {name: function} to register
        prefix: Optional prefix for tool names

    Example:
        from src.tools.demo_tools import word_count, reverse_string
        from src.tools.math_tools import add, multiply

        tools = {
            'word_count': word_count,
            'reverse_text': reverse_string,
            'add': add,
            'multiply': multiply
        }

        register_multiple_tools(agent, tools)
    """
    for name, func in tool_functions.items():
        tool_name = f"{prefix}{name}" if prefix else name
        register_tool_with_context(agent, func, tool_name)


# Agent Diagnostics and Tool Call Tracking


@dataclass
class ToolCallInfo:
    """Information about a tool call execution."""

    tool_name: str | None = None
    arguments: dict[str, Any] | None = None
    result: str | None = None
    call_id: str | None = None
    message_index: int | None = None
    has_matching_return: bool = False
    call_part_index: int | None = None
    return_part_index: int | None = None


@dataclass
class AgentDiagnostics:
    """Comprehensive diagnostics for an agent run."""

    tool_calls: list[ToolCallInfo]
    total_messages: int
    has_tools_executed: bool
    execution_summary: str
    confidence_indicators: dict[str, Any]


def extract_tool_calls_from_messages(
    messages: list[Any], debug: bool = False
) -> list[ToolCallInfo]:
    """Extract tool call information from PydanticAI message history.

    Properly pairs ToolCallPart and ToolReturnPart objects to count each tool
    execution as a single operation rather than two separate events.

    Args:
        messages: List of messages from result.all_messages()
        debug: If True, print detailed parsing information

    Returns:
        List of ToolCallInfo objects with tool execution details
    """
    tool_calls = []
    call_parts = {}  # Map call_id -> ToolCallInfo for matching with returns

    if debug:
        print(f"\n=== DEBUG: Parsing {len(messages)} messages ===")

    for i, msg in enumerate(messages):
        if debug:
            print(f"\nMessage {i}:")
            print(f"  Type: {type(msg)}")
            if hasattr(msg, "__dict__"):
                print(f"  Attributes: {list(msg.__dict__.keys())}")
            elif isinstance(msg, dict):
                print(f"  Keys: {list(msg.keys())}")
            print(f"  Content: {str(msg)[:150]}...")

        try:
            # Handle different message formats that PydanticAI might use

            # 0. Check for PydanticAI parts structure first (most common)
            if hasattr(msg, "parts") and msg.parts:
                for part_idx, part in enumerate(msg.parts):
                    # Check for ToolCallPart
                    if (
                        hasattr(part, "tool_name")
                        and hasattr(part, "tool_call_id")
                        and not hasattr(part, "content")
                    ):
                        call_id = getattr(part, "tool_call_id", None)
                        tool_info = ToolCallInfo(
                            tool_name=getattr(part, "tool_name", None),
                            arguments=getattr(part, "args", None),
                            call_id=call_id,
                            message_index=i,
                            call_part_index=part_idx,
                        )

                        # Store this call for later matching with return
                        if call_id:
                            call_parts[call_id] = tool_info

                        tool_calls.append(tool_info)
                        if debug:
                            print(
                                f"  ✓ Found ToolCallPart: {tool_info.tool_name} (call_id: {call_id})"
                            )

                    # Check for ToolReturnPart
                    elif (
                        hasattr(part, "tool_name")
                        and hasattr(part, "content")
                        and hasattr(part, "tool_call_id")
                    ):
                        call_id = getattr(part, "tool_call_id", None)
                        tool_name = getattr(part, "tool_name", None)
                        result = str(getattr(part, "content", ""))

                        # Try to match with existing call
                        if call_id and call_id in call_parts:
                            # Update the existing call info with the result
                            existing_call = call_parts[call_id]
                            existing_call.result = result
                            existing_call.has_matching_return = True
                            existing_call.return_part_index = part_idx
                            if debug:
                                print(
                                    f"  ✓ Matched ToolReturnPart to existing call: {tool_name} -> {result[:50]}..."
                                )
                        else:
                            # Create new entry for orphaned return (shouldn't happen normally)
                            tool_info = ToolCallInfo(
                                tool_name=tool_name,
                                result=result,
                                call_id=call_id,
                                message_index=i,
                                return_part_index=part_idx,
                            )
                            tool_calls.append(tool_info)
                            if debug:
                                print(
                                    f"  ✓ Found orphaned ToolReturnPart: {tool_name} -> {result[:50]}..."
                                )

            # 1. Check for tool role messages first
            if hasattr(msg, "role") and msg.role == "tool":
                tool_info = ToolCallInfo(
                    tool_name=getattr(msg, "name", None),
                    result=str(msg.content) if hasattr(msg, "content") else str(msg),
                    call_id=getattr(msg, "tool_call_id", None),
                    message_index=i,
                )
                tool_calls.append(tool_info)
                if debug:
                    print(f"  ✓ Found tool message: {tool_info.tool_name}")

            elif isinstance(msg, dict):
                # Dictionary-based message
                if msg.get("role") == "tool":
                    tool_info = ToolCallInfo(
                        tool_name=msg.get("name"),
                        result=str(msg.get("content", "")),
                        call_id=msg.get("tool_call_id"),
                        message_index=i,
                    )
                    tool_calls.append(tool_info)
                    if debug:
                        print(f"  ✓ Found dict tool message: {tool_info.tool_name}")

                # Check for tool_calls in assistant messages
                elif msg.get("role") == "assistant" and "tool_calls" in msg:
                    for j, tool_call in enumerate(msg.get("tool_calls", [])):
                        if isinstance(tool_call, dict):
                            tool_info = ToolCallInfo(
                                tool_name=tool_call.get("function", {}).get("name"),
                                arguments=tool_call.get("function", {}).get(
                                    "arguments"
                                ),
                                call_id=tool_call.get("id"),
                                message_index=i,
                            )
                            tool_calls.append(tool_info)
                            if debug:
                                print(
                                    f"  ✓ Found assistant tool call {j}: {tool_info.tool_name}"
                                )

            # 2. Check for PydanticAI specific message structures
            if hasattr(msg, "content") and hasattr(msg.content, "tool_calls"):
                # Some PydanticAI versions might store tool calls in content
                for tool_call in msg.content.tool_calls:
                    tool_info = ToolCallInfo(
                        tool_name=getattr(tool_call, "name", None),
                        arguments=getattr(tool_call, "arguments", None),
                        call_id=getattr(tool_call, "id", None),
                        message_index=i,
                    )
                    tool_calls.append(tool_info)
                    if debug:
                        print(f"  ✓ Found content tool call: {tool_info.tool_name}")

            # 3. Check for tool calls as direct attributes
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_info = ToolCallInfo(
                        tool_name=getattr(tool_call, "name", None)
                        or getattr(tool_call, "function", {}).get("name"),
                        arguments=getattr(tool_call, "arguments", None)
                        or getattr(tool_call, "function", {}).get("arguments"),
                        call_id=getattr(tool_call, "id", None),
                        message_index=i,
                    )
                    tool_calls.append(tool_info)
                    if debug:
                        print(f"  ✓ Found attribute tool call: {tool_info.tool_name}")

            # 4. Check for function_call attribute (older format)
            if hasattr(msg, "function_call") or (
                isinstance(msg, dict) and "function_call" in msg
            ):
                func_call = (
                    msg.function_call
                    if hasattr(msg, "function_call")
                    else msg["function_call"]
                )
                tool_info = ToolCallInfo(
                    tool_name=func_call.get("name")
                    if isinstance(func_call, dict)
                    else str(func_call),
                    arguments=func_call.get("arguments")
                    if isinstance(func_call, dict)
                    else None,
                    message_index=i,
                )
                tool_calls.append(tool_info)
                if debug:
                    print(f"  ✓ Found function call: {tool_info.tool_name}")

            # 5. Look for tool execution patterns in message content
            if hasattr(msg, "content") and isinstance(msg.content, str):
                # Look for patterns like "calling tool_name" or "using get_current_time"
                content_lower = msg.content.lower()
                tool_patterns = [
                    (r"calling (\w+)", "calling"),
                    (r"using (\w+)", "using"),
                    (r"executed (\w+)", "executed"),
                    (r"tool (\w+) returned", "returned"),
                    (r"get_current_time\(\)", "get_current_time"),
                    (r"current_time\(\)", "current_time"),
                    (r"time_tool\(\)", "time_tool"),
                ]

                import re

                for pattern, indicator in tool_patterns:
                    matches = re.findall(pattern, content_lower)
                    for match in matches:
                        tool_info = ToolCallInfo(
                            tool_name=match if isinstance(match, str) else indicator,
                            result=f"Inferred from content: {msg.content[:100]}...",
                            message_index=i,
                        )
                        tool_calls.append(tool_info)
                        if debug:
                            print(
                                f"  ✓ Inferred tool from content: {tool_info.tool_name}"
                            )

        except Exception as e:
            if debug:
                print(f"  ✗ Error parsing message: {e}")
            continue

    if debug:
        print(f"\n=== Total tool calls found: {len(tool_calls)} ===")
        executed_tools = [
            tc for tc in tool_calls if tc.has_matching_return or tc.result
        ]
        print(f"=== Tools with complete execution: {len(executed_tools)} ===")
        for tc in tool_calls:
            status = (
                "✓ Complete" if (tc.has_matching_return or tc.result) else "⚠ Call only"
            )
            print(f"  - {tc.tool_name} (msg {tc.message_index}) - {status}")

    return tool_calls


def analyze_agent_execution(result, agent_type: str, question: str) -> AgentDiagnostics:
    """Analyze an agent execution result for tool usage and reliability.

    Args:
        result: PydanticAI RunResult object
        agent_type: Type of agent (basic, math, demo, tools)
        question: The question that was asked

    Returns:
        AgentDiagnostics object with comprehensive analysis
    """
    try:
        messages = result.all_messages() if hasattr(result, "all_messages") else []
        tool_calls = extract_tool_calls_from_messages(messages)

        # Analyze confidence indicators
        confidence_indicators = _analyze_response_confidence(
            result.output, agent_type, question, tool_calls
        )

        # Count only complete tool executions for diagnostics
        complete_executions = [
            tc for tc in tool_calls if tc.has_matching_return or tc.result
        ]
        has_tools_executed = len(complete_executions) > 0

        # Generate execution summary
        summary = _generate_execution_summary(
            agent_type, question, tool_calls, confidence_indicators
        )

        return AgentDiagnostics(
            tool_calls=tool_calls,
            total_messages=len(messages),
            has_tools_executed=has_tools_executed,
            execution_summary=summary,
            confidence_indicators=confidence_indicators,
        )

    except Exception as e:
        return AgentDiagnostics(
            tool_calls=[],
            total_messages=0,
            has_tools_executed=False,
            execution_summary=f"Error analyzing execution: {str(e)}",
            confidence_indicators={},
        )


def _analyze_response_confidence(
    output: str, agent_type: str, question: str, tool_calls: list[ToolCallInfo]
) -> dict[str, Any]:
    """Analyze response confidence based on various indicators."""
    # Count only complete tool executions (with matching call and return)
    complete_executions = [
        tc for tc in tool_calls if tc.has_matching_return or tc.result
    ]

    indicators = {
        "tools_executed": len(complete_executions) > 0,
        "response_length": len(output.split()),
        "contains_specific_data": False,
        "expected_tool_usage": False,
        "potential_hallucination": False,
        "agent_has_required_tools": False,
    }

    # Check what tools this agent type should have
    agent_tool_capabilities = {
        "basic": [],
        "math": ["add", "multiply"],
        "demo": ["time", "facts", "text_manipulation"],
        "tools": ["add", "multiply", "time", "facts", "text_manipulation"],
    }

    agent_tools = agent_tool_capabilities.get(agent_type, [])

    # Check for specific data patterns (be more precise)
    output_lower = output.lower()

    # More specific time pattern detection
    actual_time_patterns = [
        r"\d{1,2}:\d{2}",  # 3:47, 15:47
        r"\d{1,2}:\d{2}:\d{2}",  # 15:47:14
        r"\d{1,2}\s*(am|pm)",  # 3 PM, 3:47 PM
        r"(current time|time is)\s*[:\d\s]*(am|pm|\d)",  # "current time is 3:47 PM"
    ]

    contains_actual_time = False
    import re

    for pattern in actual_time_patterns:
        if re.search(pattern, output_lower):
            contains_actual_time = True
            break

    if contains_actual_time:
        indicators["contains_specific_data"] = True
        indicators["data_type"] = "time"

    # Check for calculation results
    numeric_indicators = [
        r"=\s*\d+",
        r"equals?\s+\d+",
        r"result:\s*\d+",
        r"answer:\s*\d+",
    ]
    for pattern in numeric_indicators:
        if re.search(pattern, output_lower):
            indicators["contains_specific_data"] = True
            indicators["data_type"] = "calculation"
            break

    # Determine if tools should have been used based on question AND agent capabilities
    question_lower = question.lower()
    tool_keywords = {
        "time": ["time", "what time", "current time", "now"],
        "math": ["add", "plus", "+", "multiply", "times", "*", "calculate", "math"],
        "text": ["reverse", "word count", "palindrome", "length", "words"],
    }

    for category, keywords in tool_keywords.items():
        if any(keyword in question_lower for keyword in keywords):
            # Check if this agent type has the required tools
            if category == "time" and "time" in agent_tools:
                indicators["expected_tool_usage"] = True
                indicators["expected_tool_category"] = category
                indicators["agent_has_required_tools"] = True
                break
            elif category == "math" and any(
                math_tool in agent_tools for math_tool in ["add", "multiply"]
            ):
                indicators["expected_tool_usage"] = True
                indicators["expected_tool_category"] = category
                indicators["agent_has_required_tools"] = True
                break
            elif category == "text" and "text_manipulation" in agent_tools:
                indicators["expected_tool_usage"] = True
                indicators["expected_tool_category"] = category
                indicators["agent_has_required_tools"] = True
                break

    # More nuanced hallucination detection
    if (
        indicators["expected_tool_usage"]
        and indicators["agent_has_required_tools"]
        and not indicators["tools_executed"]
    ):
        # The agent should have tools and should have used them
        if (
            indicators.get("expected_tool_category") == "time"
            and indicators.get("data_type") == "time"
        ):
            # Agent has time tools, question asks for time, response contains time data, but no tool calls detected
            indicators["potential_hallucination"] = False
            indicators["confidence_note"] = (
                "Response contains time data - tools may have been executed but not detected"
            )
        else:
            indicators["potential_hallucination"] = True
            indicators["confidence_note"] = (
                "Expected tool usage but no tools executed - response may be hallucinated"
            )
    elif (
        indicators["expected_tool_usage"] and not indicators["agent_has_required_tools"]
    ):
        # Question requires tools this agent doesn't have - this is expected behavior
        indicators["potential_hallucination"] = False
        indicators["confidence_note"] = (
            f"{agent_type.title()} agent does not have required tools for this question"
        )
    elif not indicators["expected_tool_usage"]:
        # Question doesn't require tools - text-only response is appropriate
        indicators["potential_hallucination"] = False
        indicators["confidence_note"] = (
            "Text-only response appropriate for this question"
        )

    return indicators


def _generate_execution_summary(
    agent_type: str,
    question: str,
    tool_calls: list[ToolCallInfo],
    confidence: dict[str, Any],
) -> str:
    """Generate a human-readable execution summary."""
    if not tool_calls:
        if confidence.get("potential_hallucination"):
            return "⚠️ No tools executed - response may be hallucinated"
        elif confidence.get("confidence_note"):
            return f"🤔 {confidence['confidence_note']}"
        else:
            return "💭 No tools executed - text-only response"

    # Count only tools that have both call and return (complete executions)
    complete_executions = [
        tc for tc in tool_calls if tc.has_matching_return or tc.result
    ]
    tool_names = [tc.tool_name for tc in complete_executions if tc.tool_name]

    if tool_names:
        return f"✅ {len(complete_executions)} tool(s) executed: {', '.join(set(tool_names))}"
    else:
        return f"✅ {len(complete_executions)} tool(s) executed"


def get_agent_capabilities(agent_type: str) -> str:
    """Get a description of agent capabilities for user information."""
    capabilities = {
        "basic": "💡 Basic agent can only respond with text - no tools available",
        "math": "💡 Math agent has access to mathematical tools (add, multiply)",
        "demo": "💡 Demo agent has access to utility tools (time, facts, text manipulation) but no math",
        "tools": "💡 Tool agent has access to ALL tools - both utility and math tools",
    }
    return capabilities.get(agent_type, "💡 Unknown agent type")


def should_expect_tools(agent_type: str, question: str) -> bool:
    """Determine if we should expect tools to be used for this question/agent combo."""
    if agent_type == "basic":
        return False

    question_lower = question.lower()

    # Math-related keywords
    math_keywords = ["add", "plus", "+", "multiply", "times", "*", "calculate", "math"]
    if agent_type == "math" and any(
        keyword in question_lower for keyword in math_keywords
    ):
        return True

    # Demo/utility keywords
    utility_keywords = ["time", "fact", "word", "reverse", "palindrome", "count"]
    if agent_type in ["demo", "tools"] and any(
        keyword in question_lower for keyword in utility_keywords
    ):
        return True

    return False


def format_tool_execution_details(tool_calls: list[ToolCallInfo]) -> str:
    """Format detailed information about tool calls and their results.

    Creates a nicely formatted post-script showing exactly which tools were called,
    their inputs, and their outputs.

    Args:
        tool_calls: List of ToolCallInfo objects from extract_tool_calls_from_messages

    Returns:
        Formatted string with tool execution details
    """
    if not tool_calls:
        return ""

    # Filter to only tools that were actually executed (have results)
    executed_tools = [tc for tc in tool_calls if tc.has_matching_return or tc.result]

    if not executed_tools:
        return ""

    lines = []
    lines.append("\n" + "=" * 50)
    lines.append("TOOL EXECUTION DETAILS")
    lines.append("=" * 50)

    for i, tc in enumerate(executed_tools, 1):
        lines.append(f"\n[{i}] Tool: {tc.tool_name or 'Unknown'}")

        # Format arguments
        if tc.arguments:
            if isinstance(tc.arguments, dict):
                args_str = ", ".join([f"{k}={v}" for k, v in tc.arguments.items()])
            else:
                args_str = str(tc.arguments)
            lines.append(f"    Input: {args_str}")

        # Format result
        if tc.result:
            # Truncate very long results
            result_str = str(tc.result)
            if len(result_str) > 200:
                result_str = result_str[:200] + "..."
            lines.append(f"    Output: {result_str}")

        # Add call ID for debugging if available
        if tc.call_id:
            lines.append(f"    Call ID: {tc.call_id}")

    lines.append("\n" + "=" * 50)

    return "\n".join(lines)


def enhance_agent_response(response_text: str, tool_calls: list[ToolCallInfo]) -> str:
    """Enhance agent response by appending tool execution details.

    Takes the main text response and appends a detailed breakdown of any
    tool calls that were made during the response generation.

    Args:
        response_text: The main text response from the agent
        tool_calls: List of ToolCallInfo objects from extract_tool_calls_from_messages

    Returns:
        Enhanced response with tool details appended
    """
    tool_details = format_tool_execution_details(tool_calls)

    if tool_details:
        return response_text + tool_details
    else:
        return response_text
