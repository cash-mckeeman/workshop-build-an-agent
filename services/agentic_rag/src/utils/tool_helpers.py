"""
Tool parsing helpers for displaying agent tool usage.

Adapted from agent_intro service to show tool calls and returns in the UI.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ToolCallInfo:
    """Information about a tool call execution."""
    tool_name: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    result: Optional[str] = None
    call_id: Optional[str] = None
    message_index: Optional[int] = None
    has_matching_return: bool = False


def extract_tool_calls_from_messages(messages: List[Any], debug: bool = False) -> List[ToolCallInfo]:
    """Extract tool call information from PydanticAI message history.

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
            print(f"\nMessage {i}: {type(msg)}")
            print(f"  Content: {str(msg)[:150]}...")

        try:
            # Check for PydanticAI parts structure
            if hasattr(msg, 'parts') and msg.parts:
                for part_idx, part in enumerate(msg.parts):
                    # Check for ToolCallPart
                    if hasattr(part, 'tool_name') and hasattr(part, 'tool_call_id') and not hasattr(part, 'content'):
                        call_id = getattr(part, 'tool_call_id', None)
                        tool_info = ToolCallInfo(
                            tool_name=getattr(part, 'tool_name', None),
                            arguments=getattr(part, 'args', None),
                            call_id=call_id,
                            message_index=i
                        )

                        # Store this call for later matching with return
                        if call_id:
                            call_parts[call_id] = tool_info

                        tool_calls.append(tool_info)
                        if debug:
                            print(f"  ✓ Found ToolCallPart: {tool_info.tool_name}")

                    # Check for ToolReturnPart
                    elif hasattr(part, 'tool_name') and hasattr(part, 'content') and hasattr(part, 'tool_call_id'):
                        call_id = getattr(part, 'tool_call_id', None)
                        tool_name = getattr(part, 'tool_name', None)
                        result = str(getattr(part, 'content', ''))

                        # Try to match with existing call
                        if call_id and call_id in call_parts:
                            # Update the existing call info with the result
                            existing_call = call_parts[call_id]
                            existing_call.result = result
                            existing_call.has_matching_return = True
                            if debug:
                                print(f"  ✓ Matched ToolReturnPart: {tool_name} -> {result[:50]}...")
                        else:
                            # Create new entry for orphaned return
                            tool_info = ToolCallInfo(
                                tool_name=tool_name,
                                result=result,
                                call_id=call_id,
                                message_index=i
                            )
                            tool_calls.append(tool_info)
                            if debug:
                                print(f"  ✓ Found orphaned ToolReturnPart: {tool_name}")

            # Check for tool role messages
            if hasattr(msg, 'role') and msg.role == 'tool':
                tool_info = ToolCallInfo(
                    tool_name=getattr(msg, 'name', None),
                    result=str(msg.content) if hasattr(msg, 'content') else str(msg),
                    call_id=getattr(msg, 'tool_call_id', None),
                    message_index=i
                )
                tool_calls.append(tool_info)
                if debug:
                    print(f"  ✓ Found tool message: {tool_info.tool_name}")

        except Exception as e:
            if debug:
                print(f"  ⚠️ Error parsing message {i}: {e}")
            continue

    if debug:
        print(f"\n=== DEBUG: Found {len(tool_calls)} total tool calls ===")

    return tool_calls