"""
Tests for code display components.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from streamlit_app.components.code_display import (
    display_interactive_code,
    show_execution_step,
    display_code_comparison,
    display_code_evolution,
    display_runnable_code,
    display_tool_signature,
    display_agent_architecture,
    display_code_playground
)


class TestDisplayInteractiveCode:
    """Test display_interactive_code function."""

    @patch('streamlit.markdown')
    @patch('streamlit.code')
    def test_display_interactive_code_read_only(self, mock_code, mock_markdown):
        """Test displaying code in read-only mode."""
        code = "print('hello')"
        explanation = "This prints hello"

        result = display_interactive_code(code, explanation, allow_edit=False)

        mock_markdown.assert_called_once_with(explanation)
        mock_code.assert_called_once_with(code, language="python")
        assert result == code

    @patch('streamlit.markdown')
    @patch('streamlit.text_area')
    @patch('streamlit.code')
    def test_display_interactive_code_editable(self, mock_code, mock_text_area, mock_markdown):
        """Test displaying code in editable mode."""
        code = "print('hello')"
        explanation = "This prints hello"
        edited_code = "print('hello world')"

        mock_text_area.return_value = edited_code

        result = display_interactive_code(code, explanation, allow_edit=True, key="test")

        mock_markdown.assert_called_once_with(explanation)
        mock_text_area.assert_called_once_with(
            "Edit the code:",
            value=code,
            height=200,
            key="code_edit_test"
        )
        mock_code.assert_called_once_with(edited_code, language="python")
        assert result == edited_code

    @patch('streamlit.markdown')
    @patch('streamlit.code')
    def test_display_interactive_code_no_key_no_edit(self, mock_code, mock_markdown):
        """Test that editing is disabled when no key is provided."""
        code = "print('hello')"
        explanation = "This prints hello"

        result = display_interactive_code(code, explanation, allow_edit=True, key=None)

        mock_markdown.assert_called_once_with(explanation)
        mock_code.assert_called_once_with(code, language="python")
        assert result == code


class TestShowExecutionStep:
    """Test show_execution_step function."""

    @patch('streamlit.expander')
    @patch('streamlit.columns')
    @patch('streamlit.subheader')
    @patch('streamlit.json')
    @patch('streamlit.write')
    def test_show_execution_step_basic(self, mock_write, mock_json, mock_subheader,
                                      mock_columns, mock_expander):
        """Test basic execution step display."""
        # Mock context managers
        mock_expander_context = MagicMock()
        mock_expander.return_value.__enter__.return_value = mock_expander_context
        mock_expander.return_value.__exit__.return_value = None

        mock_col_contexts = [MagicMock(), MagicMock()]
        mock_columns.return_value = mock_col_contexts
        for mock_col in mock_col_contexts:
            mock_col.__enter__.return_value = mock_col
            mock_col.__exit__.return_value = None

        input_data = "test input"
        output_data = "test output"

        show_execution_step("Test Step", input_data, output_data)

        mock_expander.assert_called_once_with("🔍 Test Step", expanded=False)
        mock_columns.assert_called_once_with(2)

    @patch('streamlit.expander')
    @patch('streamlit.columns')
    @patch('streamlit.subheader')
    @patch('streamlit.json')
    @patch('streamlit.code')
    @patch('streamlit.markdown')
    def test_show_execution_step_with_code_and_explanation(self, mock_markdown, mock_code,
                                                          mock_json, mock_subheader,
                                                          mock_columns, mock_expander):
        """Test execution step with code and explanation."""
        # Mock context managers
        mock_expander_context = MagicMock()
        mock_expander.return_value.__enter__.return_value = mock_expander_context
        mock_expander.return_value.__exit__.return_value = None

        mock_col_contexts = [MagicMock(), MagicMock()]
        mock_columns.return_value = mock_col_contexts
        for mock_col in mock_col_contexts:
            mock_col.__enter__.return_value = mock_col
            mock_col.__exit__.return_value = None

        input_data = {"key": "value"}
        output_data = {"result": "success"}
        code = "print('test')"
        explanation = "This is a test"

        show_execution_step("Test Step", input_data, output_data, code, explanation)

        mock_json.assert_called()  # Called for both input and output dicts
        mock_code.assert_called_once_with(code, language="python")
        mock_markdown.assert_called_with(explanation)


class TestDisplayCodeComparison:
    """Test display_code_comparison function."""

    @patch('streamlit.columns')
    @patch('streamlit.subheader')
    @patch('streamlit.code')
    def test_display_code_comparison(self, mock_code, mock_subheader, mock_columns):
        """Test side-by-side code comparison."""
        # Mock columns context managers
        mock_col_contexts = [MagicMock(), MagicMock()]
        mock_columns.return_value = mock_col_contexts
        for mock_col in mock_col_contexts:
            mock_col.__enter__.return_value = mock_col
            mock_col.__exit__.return_value = None

        title1 = "Before"
        code1 = "old code"
        title2 = "After"
        code2 = "new code"

        display_code_comparison(title1, code1, title2, code2, "python")

        mock_columns.assert_called_once_with(2)
        # Subheader should be called twice (once for each column)
        assert mock_subheader.call_count == 2
        # Code should be called twice (once for each code block)
        assert mock_code.call_count == 2


class TestDisplayCodeEvolution:
    """Test display_code_evolution function."""

    @patch('streamlit.subheader')
    @patch('streamlit.markdown')
    @patch('streamlit.expander')
    @patch('streamlit.code')
    def test_display_code_evolution(self, mock_code, mock_expander, mock_markdown, mock_subheader):
        """Test code evolution display."""
        # Mock expander context manager
        mock_expander_context = MagicMock()
        mock_expander.return_value.__enter__.return_value = mock_expander_context
        mock_expander.return_value.__exit__.return_value = None

        steps = [
            {"title": "Step 1", "code": "code1", "explanation": "First step"},
            {"title": "Step 2", "code": "code2"},
            {"title": "Step 3", "code": "code3", "language": "javascript"}
        ]

        display_code_evolution(steps, current_step=1)

        mock_subheader.assert_called_once_with("Code Evolution")
        # Should have markdown calls for each step status
        assert mock_markdown.call_count >= len(steps)
        # Should have expander for each step
        assert mock_expander.call_count == len(steps)


class TestDisplayRunnableCode:
    """Test display_runnable_code function."""

    @patch('streamlit.markdown')
    @patch('streamlit.code')
    @patch('streamlit.button')
    def test_display_runnable_code_with_explanation(self, mock_button, mock_code, mock_markdown):
        """Test runnable code display with explanation."""
        code = "print('hello')"
        explanation = "This code prints hello"
        mock_button.return_value = True

        result = display_runnable_code(code, explanation=explanation, key="test")

        mock_markdown.assert_called_once_with(explanation)
        mock_code.assert_called_once_with(code, language="python")
        mock_button.assert_called_once_with("Run Code", key="test")
        assert result is True

    @patch('streamlit.code')
    @patch('streamlit.button')
    def test_display_runnable_code_no_explanation(self, mock_button, mock_code):
        """Test runnable code display without explanation."""
        code = "print('hello')"
        mock_button.return_value = False

        result = display_runnable_code(code)

        mock_code.assert_called_once_with(code, language="python")
        mock_button.assert_called_once_with("Run Code", key=None)
        assert result is False


class TestDisplayToolSignature:
    """Test display_tool_signature function."""

    @patch('streamlit.expander')
    @patch('streamlit.code')
    @patch('streamlit.markdown')
    def test_display_tool_signature(self, mock_markdown, mock_code, mock_expander):
        """Test tool signature display."""
        # Mock expander context manager
        mock_expander_context = MagicMock()
        mock_expander.return_value.__enter__.return_value = mock_expander_context
        mock_expander.return_value.__exit__.return_value = None

        tool_name = "add"
        parameters = {
            "a": {"type": "int", "description": "First number"},
            "b": {"type": "int", "description": "Second number"}
        }
        return_type = "int"
        description = "Adds two numbers"

        display_tool_signature(tool_name, parameters, return_type, description)

        mock_expander.assert_called_once_with("🔧 add Function", expanded=False)
        mock_code.assert_called_once()  # Called for the signature
        # Should have multiple markdown calls for description and parameters
        assert mock_markdown.call_count >= 3

    @patch('streamlit.expander')
    @patch('streamlit.code')
    @patch('streamlit.markdown')
    def test_display_tool_signature_no_parameters(self, mock_markdown, mock_code, mock_expander):
        """Test tool signature display with no parameters."""
        # Mock expander context manager
        mock_expander_context = MagicMock()
        mock_expander.return_value.__enter__.return_value = mock_expander_context
        mock_expander.return_value.__exit__.return_value = None

        tool_name = "get_time"
        parameters = {}
        return_type = "str"
        description = "Gets current time"

        display_tool_signature(tool_name, parameters, return_type, description)

        mock_expander.assert_called_once_with("🔧 get_time Function", expanded=False)
        mock_code.assert_called_once()
        # Should have description but no parameters section
        mock_markdown.assert_called()


class TestDisplayAgentArchitecture:
    """Test display_agent_architecture function."""

    @patch('streamlit.subheader')
    @patch('streamlit.columns')
    @patch('streamlit.markdown')
    def test_display_agent_architecture(self, mock_markdown, mock_columns, mock_subheader):
        """Test agent architecture display."""
        # Mock columns context managers
        mock_col_contexts = [MagicMock() for _ in range(4)]
        mock_columns.return_value = mock_col_contexts
        for mock_col in mock_col_contexts:
            mock_col.__enter__.return_value = mock_col
            mock_col.__exit__.return_value = None

        display_agent_architecture()

        mock_subheader.assert_called_once_with("🏗️ Agent Architecture")
        mock_columns.assert_called_once_with(4)
        # Should have markdown calls for each component plus the data flow
        assert mock_markdown.call_count >= 5


class TestDisplayCodePlayground:
    """Test display_code_playground function."""

    @patch('streamlit.subheader')
    @patch('streamlit.markdown')
    @patch('streamlit.text_area')
    @patch('streamlit.columns')
    @patch('streamlit.button')
    @patch('streamlit.info')
    def test_display_code_playground(self, mock_info, mock_button, mock_columns,
                                   mock_text_area, mock_markdown, mock_subheader):
        """Test code playground display."""
        # Mock context managers
        mock_col_contexts = [MagicMock(), MagicMock()]
        mock_columns.return_value = mock_col_contexts
        for mock_col in mock_col_contexts:
            mock_col.__enter__.return_value = mock_col
            mock_col.__exit__.return_value = None

        initial_code = "print('hello')"
        edited_code = "print('hello world')"
        mock_text_area.return_value = edited_code
        mock_button.return_value = True

        result = display_code_playground(initial_code, "test")

        mock_subheader.assert_called_once_with("🎮 Code Playground")
        mock_text_area.assert_called_once_with(
            "Python Code:",
            value=initial_code,
            height=300,
            key="playground_test",
            help="Write your Python code here. Click 'Run Code' to execute."
        )
        mock_columns.assert_called_once_with([1, 4])
        mock_button.assert_called_once_with("▶️ Run Code", key="run_test")
        mock_info.assert_called_once()  # Called when run button is clicked
        assert result == edited_code

    @patch('streamlit.subheader')
    @patch('streamlit.markdown')
    @patch('streamlit.text_area')
    @patch('streamlit.columns')
    @patch('streamlit.button')
    def test_display_code_playground_no_run(self, mock_button, mock_columns,
                                          mock_text_area, mock_markdown, mock_subheader):
        """Test code playground when run button is not clicked."""
        # Mock context managers
        mock_col_contexts = [MagicMock(), MagicMock()]
        mock_columns.return_value = mock_col_contexts
        for mock_col in mock_col_contexts:
            mock_col.__enter__.return_value = mock_col
            mock_col.__exit__.return_value = None

        initial_code = ""
        edited_code = "print('test')"
        mock_text_area.return_value = edited_code
        mock_button.return_value = False

        result = display_code_playground(initial_code, "test")

        assert result == edited_code
        # info should not be called when button is not clicked
        with patch('streamlit.info') as mock_info:
            mock_info.assert_not_called()


class TestCodeDisplayIntegration:
    """Integration tests for code display components."""

    @patch('streamlit.markdown')
    @patch('streamlit.code')
    @patch('streamlit.text_area')
    def test_interactive_code_flow(self, mock_text_area, mock_code, mock_markdown):
        """Test the flow of interactive code editing."""
        original_code = "x = 1"
        modified_code = "x = 2"
        explanation = "Test variable assignment"

        mock_text_area.return_value = modified_code

        # Test read-only first
        result1 = display_interactive_code(original_code, explanation, allow_edit=False)
        assert result1 == original_code

        # Test editable
        result2 = display_interactive_code(original_code, explanation, allow_edit=True, key="test")
        assert result2 == modified_code

    @patch('streamlit.expander')
    @patch('streamlit.markdown')
    @patch('streamlit.code')
    def test_code_evolution_progression(self, mock_code, mock_markdown, mock_expander):
        """Test code evolution with different step states."""
        # Mock expander context manager
        mock_expander_context = MagicMock()
        mock_expander.return_value.__enter__.return_value = mock_expander_context
        mock_expander.return_value.__exit__.return_value = None

        steps = [
            {"title": "Init", "code": "x = 0"},
            {"title": "Process", "code": "x += 1"},
            {"title": "Result", "code": "print(x)"}
        ]

        # Test different current steps
        for current in [-1, 0, 1, 2]:
            display_code_evolution(steps, current)

        # Should have called expander for each step in each iteration
        expected_expander_calls = len(steps) * 4  # 4 iterations
        assert mock_expander.call_count == expected_expander_calls