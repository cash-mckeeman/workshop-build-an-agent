"""
Formatting utilities for the Streamlit app.
"""

import streamlit as st
from typing import Any, Dict, List, Optional


def format_code_block(code: str, language: str = "python") -> str:
    """Format code with syntax highlighting."""
    return f"```{language}\n{code}\n```"


def format_agent_response(response: str, agent_type: str = "assistant") -> str:
    """Format agent response with appropriate styling."""
    return f"**{agent_type.title()}:** {response}"


def format_error_message(error: Exception) -> str:
    """Format error message for display."""
    return f"❌ **Error:** {str(error)}"


def format_success_message(message: str) -> str:
    """Format success message for display."""
    return f"✅ **Success:** {message}"


def format_warning_message(message: str) -> str:
    """Format warning message for display."""
    return f"⚠️ **Warning:** {message}"


def format_info_message(message: str) -> str:
    """Format info message for display."""
    return f"ℹ️ **Info:** {message}"


def format_provider_status(name: str, status: str, message: str = "") -> str:
    """Format provider health status."""
    if status == "healthy":
        return f"✅ **{name.upper()}**: Ready"
    else:
        return f"❌ **{name.upper()}**: {message}"


def format_tutorial_progress(progress: Dict[str, bool]) -> str:
    """Format tutorial progress as a progress bar."""
    total_steps = len(progress)
    completed_steps = sum(progress.values())
    progress_percent = (completed_steps / total_steps) * 100 if total_steps > 0 else 0

    progress_bar = "█" * int(progress_percent // 10) + "░" * (10 - int(progress_percent // 10))
    return f"Progress: {progress_bar} {completed_steps}/{total_steps} ({progress_percent:.0f}%)"


def format_component_explanation(title: str, description: str, examples: List[str] = None) -> str:
    """Format component explanation with examples."""
    explanation = f"### {title}\n\n{description}\n\n"

    if examples:
        explanation += "**Examples:**\n"
        for example in examples:
            explanation += f"- {example}\n"

    return explanation


def format_step_by_step(steps: List[str], current_step: int = -1) -> str:
    """Format step-by-step instructions with current step highlighted."""
    formatted_steps = []

    for i, step in enumerate(steps, 1):
        if i == current_step:
            formatted_steps.append(f"**👉 {i}. {step}**")
        elif i < current_step:
            formatted_steps.append(f"✅ {i}. {step}")
        else:
            formatted_steps.append(f"⏳ {i}. {step}")

    return "\n".join(formatted_steps)


def format_code_with_explanation(code: str, explanation: str, language: str = "python") -> str:
    """Format code block with explanation."""
    return f"{explanation}\n\n{format_code_block(code, language)}"


def format_agent_comparison(basic_result: str, enhanced_result: str) -> str:
    """Format side-by-side agent comparison."""
    return f"""
**Without Tools (Basic Agent):**
{basic_result}

**With Tools (Enhanced Agent):**
{enhanced_result}
"""


def format_tool_call_visualization(tool_name: str, inputs: Dict[str, Any], output: Any) -> str:
    """Format tool call for visualization."""
    formatted_inputs = ", ".join([f"{k}={v}" for k, v in inputs.items()])
    return f"🔧 **Tool Call:** `{tool_name}({formatted_inputs})` → `{output}`"


def format_conversation_history(messages: List[Dict[str, str]], max_messages: int = 10) -> str:
    """Format conversation history for display."""
    if not messages:
        return "No conversation history yet."

    # Show only the last max_messages
    recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages

    formatted = []
    for msg in recent_messages:
        role = "👤" if msg["role"] == "user" else "🤖"
        formatted.append(f"{role} **{msg['role'].title()}:** {msg['content']}")

    if len(messages) > max_messages:
        formatted.insert(0, f"... (showing last {max_messages} of {len(messages)} messages)")

    return "\n\n".join(formatted)


def format_memory_visualization(conversation_turns: int, total_tokens: int = None) -> str:
    """Format memory state visualization."""
    memory_info = f"💾 **Memory State:**\n- Conversation turns: {conversation_turns}"

    if total_tokens:
        memory_info += f"\n- Total tokens: {total_tokens:,}"

    return memory_info


def format_routing_decision(decision: str, reasoning: str = None) -> str:
    """Format routing decision explanation."""
    formatted = f"🚦 **Routing Decision:** {decision}"

    if reasoning:
        formatted += f"\n**Reasoning:** {reasoning}"

    return formatted