"""
Tests for formatting utilities.
"""

import pytest

from streamlit_app.utils.formatters import (
    format_agent_comparison,
    format_agent_response,
    format_code_block,
    format_code_with_explanation,
    format_component_explanation,
    format_conversation_history,
    format_error_message,
    format_info_message,
    format_memory_visualization,
    format_provider_status,
    format_routing_decision,
    format_step_by_step,
    format_success_message,
    format_tool_call_visualization,
    format_tutorial_progress,
    format_warning_message,
)


@pytest.mark.unit
class TestFormatCodeBlock:
    """Test format_code_block function."""

    def test_format_code_block_python(self):
        """Test formatting Python code block."""
        code = "print('hello')"
        result = format_code_block(code, "python")
        expected = "```python\nprint('hello')\n```"
        assert result == expected

    def test_format_code_block_javascript(self):
        """Test formatting JavaScript code block."""
        code = "console.log('hello');"
        result = format_code_block(code, "javascript")
        expected = "```javascript\nconsole.log('hello');\n```"
        assert result == expected

    def test_format_code_block_default_language(self):
        """Test formatting code block with default language."""
        code = "some code"
        result = format_code_block(code)
        expected = "```python\nsome code\n```"
        assert result == expected


@pytest.mark.unit
class TestFormatAgentResponse:
    """Test format_agent_response function."""

    def test_format_agent_response_default(self):
        """Test formatting agent response with default type."""
        response = "Hello, how can I help?"
        result = format_agent_response(response)
        expected = "**Assistant:** Hello, how can I help?"
        assert result == expected

    def test_format_agent_response_custom_type(self):
        """Test formatting agent response with custom type."""
        response = "Math problem solved!"
        result = format_agent_response(response, "math_agent")
        expected = "**Math_Agent:** Math problem solved!"
        assert result == expected


@pytest.mark.unit
class TestFormatMessages:
    """Test various message formatting functions."""

    def test_format_error_message(self):
        """Test error message formatting."""
        error = ValueError("Something went wrong")
        result = format_error_message(error)
        expected = "❌ **Error:** Something went wrong"
        assert result == expected

    def test_format_success_message(self):
        """Test success message formatting."""
        message = "Operation completed"
        result = format_success_message(message)
        expected = "✅ **Success:** Operation completed"
        assert result == expected

    def test_format_warning_message(self):
        """Test warning message formatting."""
        message = "This might be risky"
        result = format_warning_message(message)
        expected = "⚠️ **Warning:** This might be risky"
        assert result == expected

    def test_format_info_message(self):
        """Test info message formatting."""
        message = "Just so you know"
        result = format_info_message(message)
        expected = "ℹ️ **Info:** Just so you know"
        assert result == expected


@pytest.mark.unit
class TestFormatProviderStatus:
    """Test format_provider_status function."""

    def test_format_provider_status_healthy(self):
        """Test formatting healthy provider status."""
        result = format_provider_status("openai", "healthy")
        expected = "✅ **OPENAI**: Ready"
        assert result == expected

    def test_format_provider_status_unhealthy(self):
        """Test formatting unhealthy provider status."""
        result = format_provider_status("anthropic", "unhealthy", "API key missing")
        expected = "❌ **ANTHROPIC**: API key missing"
        assert result == expected

    def test_format_provider_status_unhealthy_no_message(self):
        """Test formatting unhealthy provider status without message."""
        result = format_provider_status("ollama", "unhealthy")
        expected = "❌ **OLLAMA**: "
        assert result == expected


@pytest.mark.unit
class TestFormatTutorialProgress:
    """Test format_tutorial_progress function."""

    def test_format_tutorial_progress_empty(self):
        """Test formatting progress with no steps."""
        progress = {}
        result = format_tutorial_progress(progress)
        expected = "Progress: ░░░░░░░░░░ 0/0 (0%)"
        assert result == expected

    def test_format_tutorial_progress_partial(self):
        """Test formatting partial progress."""
        progress = {"step1": True, "step2": False, "step3": True, "step4": False}
        result = format_tutorial_progress(progress)
        expected = "Progress: █████░░░░░ 2/4 (50%)"
        assert result == expected

    def test_format_tutorial_progress_complete(self):
        """Test formatting complete progress."""
        progress = {"step1": True, "step2": True}
        result = format_tutorial_progress(progress)
        expected = "Progress: ██████████ 2/2 (100%)"
        assert result == expected

    def test_format_tutorial_progress_none_complete(self):
        """Test formatting progress with no completed steps."""
        progress = {"step1": False, "step2": False, "step3": False}
        result = format_tutorial_progress(progress)
        expected = "Progress: ░░░░░░░░░░ 0/3 (0%)"
        assert result == expected


@pytest.mark.unit
class TestFormatComponentExplanation:
    """Test format_component_explanation function."""

    def test_format_component_explanation_no_examples(self):
        """Test formatting component explanation without examples."""
        title = "Model Component"
        description = "The brain of the agent"
        result = format_component_explanation(title, description)
        expected = "### Model Component\n\nThe brain of the agent\n\n"
        assert result == expected

    def test_format_component_explanation_with_examples(self):
        """Test formatting component explanation with examples."""
        title = "Tools Component"
        description = "Functions the agent can call"
        examples = ["Calculator", "Web Search", "File Operations"]
        result = format_component_explanation(title, description, examples)
        expected = """### Tools Component

Functions the agent can call

**Examples:**
- Calculator
- Web Search
- File Operations
"""
        assert result == expected

    def test_format_component_explanation_empty_examples(self):
        """Test formatting component explanation with empty examples list."""
        title = "Memory Component"
        description = "Stores conversation history"
        result = format_component_explanation(title, description, [])
        expected = "### Memory Component\n\nStores conversation history\n\n"
        assert result == expected


@pytest.mark.unit
class TestFormatStepByStep:
    """Test format_step_by_step function."""

    def test_format_step_by_step_no_current(self):
        """Test formatting steps with no current step."""
        steps = ["Start the agent", "Ask a question", "Get response"]
        result = format_step_by_step(steps)
        expected = "⏳ 1. Start the agent\n⏳ 2. Ask a question\n⏳ 3. Get response"
        assert result == expected

    def test_format_step_by_step_with_current(self):
        """Test formatting steps with current step highlighted."""
        steps = ["Start the agent", "Ask a question", "Get response"]
        result = format_step_by_step(steps, current_step=2)
        expected = "✅ 1. Start the agent\n**👉 2. Ask a question**\n⏳ 3. Get response"
        assert result == expected

    def test_format_step_by_step_all_complete(self):
        """Test formatting steps when all are complete."""
        steps = ["Step 1", "Step 2", "Step 3"]
        result = format_step_by_step(steps, current_step=4)
        expected = "✅ 1. Step 1\n✅ 2. Step 2\n✅ 3. Step 3"
        assert result == expected

    def test_format_step_by_step_empty_steps(self):
        """Test formatting empty steps list."""
        steps = []
        result = format_step_by_step(steps)
        expected = ""
        assert result == expected


@pytest.mark.unit
class TestFormatCodeWithExplanation:
    """Test format_code_with_explanation function."""

    def test_format_code_with_explanation(self):
        """Test formatting code with explanation."""
        code = "x = 5"
        explanation = "This assigns the value 5 to variable x"
        result = format_code_with_explanation(code, explanation)
        expected = "This assigns the value 5 to variable x\n\n```python\nx = 5\n```"
        assert result == expected

    def test_format_code_with_explanation_custom_language(self):
        """Test formatting code with explanation and custom language."""
        code = "let x = 5;"
        explanation = "JavaScript variable assignment"
        result = format_code_with_explanation(code, explanation, "javascript")
        expected = "JavaScript variable assignment\n\n```javascript\nlet x = 5;\n```"
        assert result == expected


@pytest.mark.unit
class TestFormatAgentComparison:
    """Test format_agent_comparison function."""

    def test_format_agent_comparison(self):
        """Test formatting agent comparison."""
        basic_result = "I think the answer is around 15"
        enhanced_result = "Using the add() tool: 3 + 12 = 15"
        result = format_agent_comparison(basic_result, enhanced_result)
        expected = """
**Without Tools (Basic Agent):**
I think the answer is around 15

**With Tools (Enhanced Agent):**
Using the add() tool: 3 + 12 = 15
"""
        assert result == expected


@pytest.mark.unit
class TestFormatToolCallVisualization:
    """Test format_tool_call_visualization function."""

    def test_format_tool_call_visualization(self):
        """Test formatting tool call visualization."""
        tool_name = "add"
        inputs = {"a": 3, "b": 12}
        output = 15
        result = format_tool_call_visualization(tool_name, inputs, output)
        expected = "🔧 **Tool Call:** `add(a=3, b=12)` → `15`"
        assert result == expected

    def test_format_tool_call_visualization_no_inputs(self):
        """Test formatting tool call with no inputs."""
        tool_name = "get_time"
        inputs = {}
        output = "2:30 PM"
        result = format_tool_call_visualization(tool_name, inputs, output)
        expected = "🔧 **Tool Call:** `get_time()` → `2:30 PM`"
        assert result == expected

    def test_format_tool_call_visualization_multiple_inputs(self):
        """Test formatting tool call with multiple inputs."""
        tool_name = "calculate"
        inputs = {"operation": "multiply", "x": 5, "y": 7}
        output = 35
        result = format_tool_call_visualization(tool_name, inputs, output)
        # Note: dict order might vary, so we check the components
        assert "🔧 **Tool Call:**" in result
        assert "calculate(" in result
        assert "operation=multiply" in result
        assert "x=5" in result
        assert "y=7" in result
        assert "→ `35`" in result


@pytest.mark.unit
class TestFormatConversationHistory:
    """Test format_conversation_history function."""

    def test_format_conversation_history_empty(self):
        """Test formatting empty conversation history."""
        messages = []
        result = format_conversation_history(messages)
        expected = "No conversation history yet."
        assert result == expected

    def test_format_conversation_history_basic(self):
        """Test formatting basic conversation history."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = format_conversation_history(messages)
        expected = "👤 **User:** Hello\n\n🤖 **Assistant:** Hi there!"
        assert result == expected

    def test_format_conversation_history_max_limit(self):
        """Test formatting conversation history with max limit."""
        messages = [{"role": "user", "content": f"Message {i}"} for i in range(15)]
        result = format_conversation_history(messages, max_messages=5)

        assert "... (showing last 5 of 15 messages)" in result
        assert "Message 10" in result  # Should show last 5 messages
        assert "Message 14" in result
        assert "Message 0" not in result  # Should not show early messages

    def test_format_conversation_history_under_limit(self):
        """Test formatting conversation history under max limit."""
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        result = format_conversation_history(messages, max_messages=10)

        assert "... (showing last" not in result
        assert "👤 **User:** Hi" in result
        assert "🤖 **Assistant:** Hello" in result


@pytest.mark.unit
class TestFormatMemoryVisualization:
    """Test format_memory_visualization function."""

    def test_format_memory_visualization_basic(self):
        """Test formatting memory visualization without tokens."""
        turns = 5
        result = format_memory_visualization(turns)
        expected = "💾 **Memory State:**\n- Conversation turns: 5"
        assert result == expected

    def test_format_memory_visualization_with_tokens(self):
        """Test formatting memory visualization with tokens."""
        turns = 3
        tokens = 1234
        result = format_memory_visualization(turns, tokens)
        expected = (
            "💾 **Memory State:**\n- Conversation turns: 3\n- Total tokens: 1,234"
        )
        assert result == expected

    def test_format_memory_visualization_zero_turns(self):
        """Test formatting memory visualization with zero turns."""
        turns = 0
        result = format_memory_visualization(turns)
        expected = "💾 **Memory State:**\n- Conversation turns: 0"
        assert result == expected


@pytest.mark.unit
class TestFormatRoutingDecision:
    """Test format_routing_decision function."""

    def test_format_routing_decision_basic(self):
        """Test formatting routing decision without reasoning."""
        decision = "Use calculator tool"
        result = format_routing_decision(decision)
        expected = "🚦 **Routing Decision:** Use calculator tool"
        assert result == expected

    def test_format_routing_decision_with_reasoning(self):
        """Test formatting routing decision with reasoning."""
        decision = "Use web search"
        reasoning = "User asked for current weather information"
        result = format_routing_decision(decision, reasoning)
        expected = "🚦 **Routing Decision:** Use web search\n**Reasoning:** User asked for current weather information"
        assert result == expected

    def test_format_routing_decision_empty_reasoning(self):
        """Test formatting routing decision with empty reasoning."""
        decision = "Direct response"
        result = format_routing_decision(decision, "")
        expected = "🚦 **Routing Decision:** Direct response"
        assert result == expected


@pytest.mark.unit
class TestFormatterIntegration:
    """Integration tests for formatter functions."""

    def test_code_formatting_chain(self):
        """Test chaining code formatting functions."""
        code = "def add(a, b): return a + b"
        explanation = "Simple addition function"

        # Test individual functions
        code_block = format_code_block(code)
        code_with_explanation = format_code_with_explanation(code, explanation)

        assert code in code_block
        assert explanation in code_with_explanation
        assert code_block.replace("```python\n", "").replace("\n```", "") == code

    def test_message_formatting_consistency(self):
        """Test consistency across message formatting functions."""
        message = "Test message"

        success = format_success_message(message)
        error = format_error_message(Exception(message))
        warning = format_warning_message(message)
        info = format_info_message(message)

        # All should contain the original message
        assert message in success
        assert message in error
        assert message in warning
        assert message in info

        # All should have appropriate icons
        assert "✅" in success
        assert "❌" in error
        assert "⚠️" in warning
        assert "ℹ️" in info

    def test_progress_visualization_accuracy(self):
        """Test that progress visualization accurately represents completion."""
        # Test various completion percentages
        test_cases = [
            ({"a": True}, 1, 1, 100),  # 100%
            ({"a": True, "b": False}, 1, 2, 50),  # 50%
            ({"a": False, "b": False, "c": False}, 0, 3, 0),  # 0%
            ({"a": True, "b": True, "c": False, "d": False}, 2, 4, 50),  # 50%
        ]

        for (
            progress_dict,
            expected_completed,
            expected_total,
            expected_percent,
        ) in test_cases:
            result = format_tutorial_progress(progress_dict)
            assert f"{expected_completed}/{expected_total}" in result
            assert f"({expected_percent}%)" in result
