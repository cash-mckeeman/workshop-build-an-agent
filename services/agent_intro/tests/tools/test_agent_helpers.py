"""Tests for tools.agent_helpers module."""

import pytest
import inspect
from unittest.mock import MagicMock, patch, PropertyMock
from pydantic_ai import RunContext
from tools.agent_helpers import (
    with_run_context,
    register_tool_with_context,
    create_tool_wrapper,
    make_agent_tools,
    register_multiple_tools,
    ToolCallInfo,
    AgentDiagnostics,
    extract_tool_calls_from_messages,
    analyze_agent_execution,
    format_tool_execution_details,
    enhance_agent_response
)


class TestWithRunContext:
    """Test cases for with_run_context decorator."""

    def test_with_run_context_basic(self):
        """Test basic functionality of with_run_context decorator."""
        def simple_func(a: int, b: int) -> int:
            return a + b

        wrapped = with_run_context(simple_func)

        # Mock RunContext
        mock_ctx = MagicMock(spec=RunContext)

        # Test that wrapped function works
        result = wrapped(mock_ctx, 3, 5)
        assert result == 8

    def test_with_run_context_preserves_docstring(self):
        """Test that with_run_context preserves the original function's docstring."""
        def documented_func(x: int) -> int:
            """This is a test function."""
            return x * 2

        wrapped = with_run_context(documented_func)
        assert wrapped.__doc__ == documented_func.__doc__

    def test_with_run_context_preserves_name(self):
        """Test that with_run_context preserves the original function's name."""
        def named_func(x: int) -> int:
            return x

        wrapped = with_run_context(named_func)
        assert wrapped.__name__ == named_func.__name__

    def test_with_run_context_annotations(self):
        """Test that with_run_context adds proper annotations."""
        def annotated_func(a: int, b: str) -> bool:
            return True

        wrapped = with_run_context(annotated_func)

        # Check that RunContext is added as first parameter
        annotations = wrapped.__annotations__
        assert 'ctx' in annotations
        assert annotations['ctx'] == RunContext

        # Original annotations should be preserved
        assert 'a' in annotations
        assert 'b' in annotations
        assert annotations['a'] == int
        assert annotations['b'] == str
        assert annotations['return'] == bool

    def test_with_run_context_signature(self):
        """Test that with_run_context updates the function signature."""
        def original_func(x: int, y: str) -> int:
            return len(y) + x

        wrapped = with_run_context(original_func)

        # Check signature
        sig = inspect.signature(wrapped)
        params = list(sig.parameters.keys())

        assert params[0] == 'ctx'
        assert params[1] == 'x'
        assert params[2] == 'y'

        # Check parameter annotations
        assert sig.parameters['ctx'].annotation == RunContext
        assert sig.parameters['x'].annotation == int
        assert sig.parameters['y'].annotation == str

    def test_with_run_context_kwargs(self):
        """Test with_run_context with keyword arguments."""
        def func_with_kwargs(a: int, b: int = 10) -> int:
            return a + b

        wrapped = with_run_context(func_with_kwargs)
        mock_ctx = MagicMock(spec=RunContext)

        # Test with positional args
        assert wrapped(mock_ctx, 5) == 15

        # Test with keyword args
        assert wrapped(mock_ctx, 5, b=20) == 25


class TestRegisterToolWithContext:
    """Test cases for register_tool_with_context function."""

    def test_register_tool_with_context_basic(self):
        """Test basic functionality of register_tool_with_context."""
        mock_agent = MagicMock()
        mock_agent.tool = MagicMock(return_value="tool_result")

        def test_func(x: int) -> int:
            return x * 2

        result = register_tool_with_context(mock_agent, test_func)

        # Should return the result of agent.tool()
        assert result == "tool_result"

        # Should have called agent.tool with wrapped function
        mock_agent.tool.assert_called_once()
        call_args = mock_agent.tool.call_args
        assert len(call_args[0]) == 1  # One positional argument (the function)

    def test_register_tool_with_context_with_name(self):
        """Test register_tool_with_context with custom name."""
        mock_agent = MagicMock()
        mock_agent.tool = MagicMock(return_value="tool_result")

        def test_func(x: int) -> int:
            return x

        register_tool_with_context(mock_agent, test_func, name="custom_name")

        # Should have called agent.tool with name parameter
        mock_agent.tool.assert_called_once()
        call_kwargs = mock_agent.tool.call_args[1]
        assert call_kwargs.get('name') == "custom_name"


class TestCreateToolWrapper:
    """Test cases for create_tool_wrapper function."""

    def test_create_tool_wrapper(self):
        """Test create_tool_wrapper function."""
        def original_func(a: int, b: int) -> int:
            return a + b

        wrapped = create_tool_wrapper(original_func)

        # Should behave like with_run_context
        mock_ctx = MagicMock(spec=RunContext)
        result = wrapped(mock_ctx, 3, 5)
        assert result == 8

        # Should have proper annotations
        assert 'ctx' in wrapped.__annotations__
        assert wrapped.__annotations__['ctx'] == RunContext


class TestMakeAgentTools:
    """Test cases for make_agent_tools function."""

    def test_make_agent_tools_single_function(self):
        """Test make_agent_tools with a single function."""
        def test_func(x: int) -> int:
            return x

        tools = make_agent_tools(test_func)

        assert isinstance(tools, dict)
        assert 'test_func' in tools
        assert callable(tools['test_func'])

    def test_make_agent_tools_multiple_functions(self):
        """Test make_agent_tools with multiple functions."""
        def func_one(x: int) -> int:
            return x + 1

        def func_two(x: int) -> int:
            return x * 2

        tools = make_agent_tools(func_one, func_two)

        assert isinstance(tools, dict)
        assert 'func_one' in tools
        assert 'func_two' in tools
        assert len(tools) == 2

        # Test that wrapped functions work
        mock_ctx = MagicMock(spec=RunContext)
        assert tools['func_one'](mock_ctx, 5) == 6
        assert tools['func_two'](mock_ctx, 5) == 10

    def test_make_agent_tools_empty(self):
        """Test make_agent_tools with no functions."""
        tools = make_agent_tools()

        assert isinstance(tools, dict)
        assert len(tools) == 0


class TestRegisterMultipleTools:
    """Test cases for register_multiple_tools function."""

    def test_register_multiple_tools_basic(self):
        """Test basic functionality of register_multiple_tools."""
        mock_agent = MagicMock()

        def func_one(x: int) -> int:
            return x

        def func_two(x: int) -> int:
            return x

        tools = {
            'tool_one': func_one,
            'tool_two': func_two
        }

        with patch('tools.agent_helpers.register_tool_with_context') as mock_register:
            register_multiple_tools(mock_agent, tools)

            # Should have called register_tool_with_context for each tool
            assert mock_register.call_count == 2

            # Check the calls
            calls = mock_register.call_args_list

            # First call
            assert calls[0][0][0] is mock_agent
            assert calls[0][0][1] is func_one
            assert calls[0][0][2] == 'tool_one'

            # Second call
            assert calls[1][0][0] is mock_agent
            assert calls[1][0][1] is func_two
            assert calls[1][0][2] == 'tool_two'

    def test_register_multiple_tools_with_prefix(self):
        """Test register_multiple_tools with prefix."""
        mock_agent = MagicMock()

        def test_func(x: int) -> int:
            return x

        tools = {'tool': test_func}

        with patch('tools.agent_helpers.register_tool_with_context') as mock_register:
            register_multiple_tools(mock_agent, tools, prefix="test_")

            # Should have called with prefixed name
            mock_register.assert_called_once()
            call_args = mock_register.call_args[0]
            assert call_args[2] == 'test_tool'  # prefixed name

    def test_register_multiple_tools_empty(self):
        """Test register_multiple_tools with empty tools dict."""
        mock_agent = MagicMock()

        with patch('tools.agent_helpers.register_tool_with_context') as mock_register:
            register_multiple_tools(mock_agent, {})

            # Should not have called register_tool_with_context
            mock_register.assert_not_called()

    def test_integration_workflow(self):
        """Test a complete workflow using the helper functions."""
        # Create some test functions
        def add_func(a: int, b: int) -> int:
            return a + b

        def multiply_func(a: int, b: int) -> int:
            return a * b

        # Create tools dict
        tools = make_agent_tools(add_func, multiply_func)

        # Verify tools were created correctly
        assert 'add_func' in tools
        assert 'multiply_func' in tools

        # Mock agent
        mock_agent = MagicMock()

        # Register tools
        with patch('tools.agent_helpers.register_tool_with_context') as mock_register:
            register_multiple_tools(mock_agent, tools)

            # Should have registered both tools
            assert mock_register.call_count == 2


class TestDataClasses:
    """Test cases for data classes."""

    def test_tool_call_info_creation(self):
        """Test ToolCallInfo creation."""
        info = ToolCallInfo(
            tool_name="test_tool",
            arguments={"x": 1},
            result="test_result",
            call_id="123"
        )

        assert info.tool_name == "test_tool"
        assert info.arguments == {"x": 1}
        assert info.result == "test_result"
        assert info.call_id == "123"
        assert info.has_matching_return == False  # Default value

    def test_agent_diagnostics_creation(self):
        """Test AgentDiagnostics creation."""
        tool_calls = [ToolCallInfo(tool_name="test")]
        diagnostics = AgentDiagnostics(
            tool_calls=tool_calls,
            total_messages=3,
            has_tools_executed=True,
            execution_summary="Test summary",
            confidence_indicators={"test": True}
        )

        assert len(diagnostics.tool_calls) == 1
        assert diagnostics.total_messages == 3
        assert diagnostics.has_tools_executed == True
        assert diagnostics.execution_summary == "Test summary"
        assert diagnostics.confidence_indicators["test"] == True


class TestMessageParsing:
    """Test cases for message parsing functions."""

    def test_extract_tool_calls_empty_messages(self):
        """Test extracting tool calls from empty message list."""
        result = extract_tool_calls_from_messages([])
        assert result == []

    def test_extract_tool_calls_with_parts(self):
        """Test extracting tool calls from messages with parts structure."""
        # Mock message with parts containing tool call
        mock_part = MagicMock()
        mock_part.tool_name = "test_tool"
        mock_part.tool_call_id = "call_123"
        mock_part.args = {"x": 1}
        # Explicitly remove content attribute to prevent ToolReturnPart detection
        del mock_part.content

        mock_message = MagicMock()
        mock_message.parts = [mock_part]
        # Ensure message doesn't match other detection patterns
        del mock_message.role
        del mock_message.function_call

        tool_calls = extract_tool_calls_from_messages([mock_message])

        assert len(tool_calls) == 1
        assert tool_calls[0].tool_name == "test_tool"
        assert tool_calls[0].call_id == "call_123"
        assert tool_calls[0].arguments == {"x": 1}

    def test_extract_tool_calls_with_tool_role(self):
        """Test extracting tool calls from tool role messages."""
        mock_message = MagicMock()
        mock_message.role = 'tool'
        mock_message.name = 'test_tool'
        mock_message.content = 'tool result'
        mock_message.tool_call_id = 'call_123'
        # Ensure this doesn't have parts to avoid double detection
        del mock_message.parts
        del mock_message.function_call

        tool_calls = extract_tool_calls_from_messages([mock_message])

        assert len(tool_calls) == 1
        assert tool_calls[0].tool_name == 'test_tool'
        assert tool_calls[0].result == 'tool result'
        assert tool_calls[0].call_id == 'call_123'

    def test_extract_tool_calls_with_dict_messages(self):
        """Test extracting tool calls from dictionary messages."""
        message = {
            'role': 'tool',
            'name': 'test_tool',
            'content': 'tool result',
            'tool_call_id': 'call_123'
        }

        tool_calls = extract_tool_calls_from_messages([message])

        assert len(tool_calls) == 1
        assert tool_calls[0].tool_name == 'test_tool'
        assert tool_calls[0].result == 'tool result'
        assert tool_calls[0].call_id == 'call_123'


class TestAnalysis:
    """Test cases for analysis functions."""

    def test_analyze_agent_execution_success(self):
        """Test successful agent execution analysis."""
        # Mock result object
        mock_result = MagicMock()
        mock_result.output = "The result is 42"
        mock_result.all_messages.return_value = []

        diagnostics = analyze_agent_execution(mock_result, "math", "What is 2 + 2?")

        assert isinstance(diagnostics, AgentDiagnostics)
        assert diagnostics.total_messages == 0
        assert diagnostics.has_tools_executed == False
        assert isinstance(diagnostics.execution_summary, str)
        assert isinstance(diagnostics.confidence_indicators, dict)

    def test_analyze_agent_execution_error(self):
        """Test agent execution analysis with error."""
        # Mock result that will cause an error
        mock_result = MagicMock()
        mock_result.all_messages.side_effect = Exception("Test error")

        diagnostics = analyze_agent_execution(mock_result, "math", "test question")

        assert isinstance(diagnostics, AgentDiagnostics)
        assert diagnostics.total_messages == 0
        assert diagnostics.has_tools_executed == False
        assert "Error analyzing execution" in diagnostics.execution_summary


class TestHelperFunctions:
    """Test cases for helper functions."""

    def test_format_tool_execution_details_empty(self):
        """Test formatting tool execution details with no tools."""
        result = format_tool_execution_details([])
        assert result == ""

    def test_format_tool_execution_details_with_tools(self):
        """Test formatting tool execution details with tools."""
        tool_call = ToolCallInfo(
            tool_name="test_tool",
            arguments={"x": 1, "y": 2},
            result="42",
            call_id="call_123",
            has_matching_return=True
        )

        result = format_tool_execution_details([tool_call])

        assert "TOOL EXECUTION DETAILS" in result
        assert "test_tool" in result
        assert "x=1, y=2" in result
        assert "42" in result

    def test_enhance_agent_response(self):
        """Test enhancing agent response with tool details."""
        tool_call = ToolCallInfo(
            tool_name="test_tool",
            result="42",
            has_matching_return=True
        )

        response = "The answer is 42"
        enhanced = enhance_agent_response(response, [tool_call])

        assert "The answer is 42" in enhanced
        assert "TOOL EXECUTION DETAILS" in enhanced

    def test_enhance_agent_response_no_tools(self):
        """Test enhancing agent response with no tools."""
        response = "Just a text response"
        enhanced = enhance_agent_response(response, [])

        assert enhanced == response  # Should be unchanged