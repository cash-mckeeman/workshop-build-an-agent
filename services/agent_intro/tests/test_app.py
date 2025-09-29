"""
Tests for the Streamlit app module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import streamlit as st

# Import the functions we want to test
from app import (
    main,
    tutorial_section,
    playground_section,
    math_demo_section,
    tools_demo_section
)


class TestMain:
    """Test main function."""

    @patch('streamlit.set_page_config')
    @patch('streamlit.title')
    @patch('streamlit.markdown')
    @patch('streamlit.sidebar')
    @patch('streamlit.tabs')
    @patch('app.get_recommended_setup')
    def test_main_page_setup(self, mock_get_setup, mock_tabs, mock_sidebar, mock_markdown, mock_title, mock_set_config):
        """Test main function sets up page correctly."""
        mock_get_setup.return_value = {"status": "ready", "recommended_provider": "openai", "recommended_model": "gpt-4"}

        # Mock tabs context managers
        mock_tab_contexts = [MagicMock() for _ in range(4)]
        mock_tabs.return_value = mock_tab_contexts

        # Mock sidebar context manager
        mock_sidebar_context = MagicMock()
        mock_sidebar.return_value.__enter__.return_value = mock_sidebar_context

        # Set up the context managers to work properly
        for mock_tab in mock_tab_contexts:
            mock_tab.__enter__.return_value = mock_tab
            mock_tab.__exit__.return_value = None

        with patch('app.tutorial_section') as mock_tutorial:
            with patch('app.playground_section') as mock_playground:
                with patch('app.math_demo_section') as mock_math:
                    with patch('app.tools_demo_section') as mock_tools:
                        main()

        mock_set_config.assert_called_once_with(
            page_title="Agent Introduction Tutorial",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        mock_title.assert_called_once_with("🤖 Introduction to AI Agents")


class TestTutorialSection:
    """Test tutorial_section function."""

    @patch('streamlit.header')
    @patch('streamlit.markdown')
    @patch('streamlit.columns')
    @patch('streamlit.subheader')
    def test_tutorial_section_content(self, mock_subheader, mock_columns, mock_markdown, mock_header):
        """Test tutorial section displays correct content."""
        # Mock columns context managers
        mock_col_contexts = [MagicMock(), MagicMock()]
        mock_columns.return_value = mock_col_contexts

        for mock_col in mock_col_contexts:
            mock_col.__enter__.return_value = mock_col
            mock_col.__exit__.return_value = None

        tutorial_section()

        mock_header.assert_called_once_with("📚 Learn About AI Agents")
        assert mock_markdown.call_count >= 1  # Multiple markdown calls expected
        mock_columns.assert_called_once_with(2)


class TestPlaygroundSection:
    """Test playground_section function."""

    @patch('streamlit.header')
    @patch('streamlit.error')
    @patch('app.get_recommended_setup')
    def test_playground_section_not_ready(self, mock_get_setup, mock_error, mock_header):
        """Test playground section when provider is not ready."""
        mock_get_setup.return_value = {"status": "not_ready"}

        playground_section()

        mock_header.assert_called_once_with("🎮 Agent Playground")
        mock_error.assert_called_once_with("Please configure a provider first (see sidebar)")

    @patch('streamlit.header')
    @patch('streamlit.markdown')
    @patch('streamlit.selectbox')
    @patch('streamlit.chat_input')
    @patch('streamlit.button')
    @patch('streamlit.session_state')
    @patch('app.get_recommended_setup')
    def test_playground_section_ready(self, mock_get_setup, mock_session_state, mock_button,
                                     mock_chat_input, mock_selectbox, mock_markdown, mock_header):
        """Test playground section when provider is ready."""
        mock_get_setup.return_value = {"status": "ready"}
        mock_session_state.messages = []
        mock_chat_input.return_value = None
        mock_button.return_value = False

        playground_section()

        mock_header.assert_called_once_with("🎮 Agent Playground")
        mock_selectbox.assert_called_once()

    @patch('streamlit.header')
    @patch('streamlit.markdown')
    @patch('streamlit.selectbox')
    @patch('streamlit.chat_input')
    @patch('streamlit.chat_message')
    @patch('streamlit.spinner')
    @patch('streamlit.rerun')
    @patch('streamlit.button')
    @patch('streamlit.session_state')
    @patch('app.get_recommended_setup')
    @patch('app.create_tutorial_agent')
    def test_playground_section_with_message(self, mock_create_agent, mock_get_setup, mock_session_state,
                                            mock_button, mock_rerun, mock_spinner, mock_chat_message,
                                            mock_chat_input, mock_selectbox, mock_markdown, mock_header):
        """Test playground section handling user input."""
        mock_get_setup.return_value = {"status": "ready"}
        mock_session_state.messages = []
        mock_chat_input.return_value = "Hello agent"
        mock_selectbox.return_value = "Basic Agent (MODEL only)"
        mock_button.return_value = False

        # Mock agent and response
        mock_agent = Mock()
        mock_result = Mock()
        mock_result.output = "Hello! I'm an AI assistant."
        mock_agent.run_sync.return_value = mock_result
        mock_create_agent.return_value = mock_agent

        # Mock context managers
        mock_chat_message.return_value.__enter__ = Mock()
        mock_chat_message.return_value.__exit__ = Mock(return_value=None)
        mock_spinner.return_value.__enter__ = Mock()
        mock_spinner.return_value.__exit__ = Mock(return_value=None)

        playground_section()

        mock_create_agent.assert_called_with("basic")


class TestMathDemoSection:
    """Test math_demo_section function."""

    @patch('streamlit.header')
    @patch('streamlit.error')
    @patch('app.get_recommended_setup')
    def test_math_demo_section_not_ready(self, mock_get_setup, mock_error, mock_header):
        """Test math demo section when provider is not ready."""
        mock_get_setup.return_value = {"status": "not_ready"}

        math_demo_section()

        mock_header.assert_called_once_with("🧮 Math Agent Demo")
        mock_error.assert_called_once_with("Please configure a provider first (see sidebar)")

    @patch('streamlit.header')
    @patch('streamlit.markdown')
    @patch('streamlit.columns')
    @patch('streamlit.subheader')
    @patch('streamlit.code')
    @patch('streamlit.button')
    @patch('app.get_recommended_setup')
    def test_math_demo_section_ready(self, mock_get_setup, mock_button, mock_code, mock_subheader,
                                    mock_columns, mock_markdown, mock_header):
        """Test math demo section when provider is ready."""
        mock_get_setup.return_value = {"status": "ready"}
        mock_button.return_value = False

        # Mock columns context managers
        mock_col_contexts = [MagicMock(), MagicMock()]
        mock_columns.return_value = mock_col_contexts

        for mock_col in mock_col_contexts:
            mock_col.__enter__.return_value = mock_col
            mock_col.__exit__.return_value = None

        math_demo_section()

        mock_header.assert_called_once_with("🧮 Math Agent Demo")
        mock_columns.assert_called_once_with([1, 1])

    @patch('streamlit.header')
    @patch('streamlit.markdown')
    @patch('streamlit.columns')
    @patch('streamlit.subheader')
    @patch('streamlit.code')
    @patch('streamlit.button')
    @patch('streamlit.spinner')
    @patch('streamlit.write')
    @patch('streamlit.warning')
    @patch('app.get_recommended_setup')
    @patch('app.create_tutorial_agent')
    def test_math_demo_basic_agent_button(self, mock_create_agent, mock_get_setup, mock_warning,
                                         mock_write, mock_spinner, mock_button, mock_code,
                                         mock_subheader, mock_columns, mock_markdown, mock_header):
        """Test math demo section basic agent button."""
        mock_get_setup.return_value = {"status": "ready"}

        # Mock button returns True only for the basic agent button
        def button_side_effect(*args, **kwargs):
            if kwargs.get('key') == 'basic_math':
                return True
            return False

        mock_button.side_effect = button_side_effect

        # Mock agent and response
        mock_agent = Mock()
        mock_result = Mock()
        mock_result.output = "I think it's around 15"
        mock_agent.run_sync.return_value = mock_result
        mock_create_agent.return_value = mock_agent

        # Mock columns context managers
        mock_col_contexts = [MagicMock(), MagicMock()]
        mock_columns.return_value = mock_col_contexts

        for mock_col in mock_col_contexts:
            mock_col.__enter__.return_value = mock_col
            mock_col.__exit__.return_value = None

        # Mock spinner context manager
        mock_spinner.return_value.__enter__ = Mock()
        mock_spinner.return_value.__exit__ = Mock(return_value=None)

        math_demo_section()

        mock_create_agent.assert_called_with("basic")
        mock_agent.run_sync.assert_called_with("What is 3 plus 12?")


class TestToolsDemoSection:
    """Test tools_demo_section function."""

    @patch('streamlit.header')
    @patch('streamlit.error')
    @patch('app.get_recommended_setup')
    def test_tools_demo_section_not_ready(self, mock_get_setup, mock_error, mock_header):
        """Test tools demo section when provider is not ready."""
        mock_get_setup.return_value = {"status": "not_ready"}

        tools_demo_section()

        mock_header.assert_called_once_with("🔧 Tools Demonstration")
        mock_error.assert_called_once_with("Please configure a provider first (see sidebar)")

    @patch('streamlit.header')
    @patch('streamlit.markdown')
    @patch('streamlit.subheader')
    @patch('streamlit.expander')
    @patch('streamlit.button')
    @patch('streamlit.text_input')
    @patch('app.get_recommended_setup')
    def test_tools_demo_section_ready(self, mock_get_setup, mock_text_input, mock_button, mock_expander,
                                     mock_subheader, mock_markdown, mock_header):
        """Test tools demo section when provider is ready."""
        mock_get_setup.return_value = {"status": "ready"}
        mock_button.return_value = False
        mock_text_input.return_value = ""

        # Mock expander context manager
        mock_expander.return_value.__enter__ = Mock()
        mock_expander.return_value.__exit__ = Mock(return_value=None)

        tools_demo_section()

        mock_header.assert_called_once_with("🔧 Tools Demonstration")

    @patch('streamlit.header')
    @patch('streamlit.markdown')
    @patch('streamlit.subheader')
    @patch('streamlit.expander')
    @patch('streamlit.button')
    @patch('streamlit.text_input')
    @patch('streamlit.spinner')
    @patch('streamlit.success')
    @patch('app.get_recommended_setup')
    @patch('app.create_tutorial_agent')
    def test_tools_demo_button_click(self, mock_create_agent, mock_get_setup, mock_success,
                                    mock_spinner, mock_text_input, mock_button, mock_expander,
                                    mock_subheader, mock_markdown, mock_header):
        """Test tools demo section button interactions."""
        mock_get_setup.return_value = {"status": "ready"}
        mock_text_input.return_value = ""

        # Mock button to return True for one specific tool example
        def button_side_effect(*args, **kwargs):
            if "What is 15 multiplied by 8?" in str(kwargs.get('key', '')):
                return True
            return False

        mock_button.side_effect = button_side_effect

        # Mock agent and response
        mock_agent = Mock()
        mock_result = Mock()
        mock_result.output = "15 × 8 = 120"
        mock_agent.run_sync.return_value = mock_result
        mock_create_agent.return_value = mock_agent

        # Mock context managers
        mock_expander.return_value.__enter__ = Mock()
        mock_expander.return_value.__exit__ = Mock(return_value=None)
        mock_spinner.return_value.__enter__ = Mock()
        mock_spinner.return_value.__exit__ = Mock(return_value=None)

        tools_demo_section()

        mock_create_agent.assert_called_with("tools")

    @patch('streamlit.header')
    @patch('streamlit.markdown')
    @patch('streamlit.subheader')
    @patch('streamlit.expander')
    @patch('streamlit.button')
    @patch('streamlit.text_input')
    @patch('streamlit.spinner')
    @patch('streamlit.success')
    @patch('app.get_recommended_setup')
    @patch('app.create_tutorial_agent')
    def test_tools_demo_custom_question(self, mock_create_agent, mock_get_setup, mock_success,
                                       mock_spinner, mock_text_input, mock_button, mock_expander,
                                       mock_subheader, mock_markdown, mock_header):
        """Test tools demo section custom question functionality."""
        mock_get_setup.return_value = {"status": "ready"}
        mock_text_input.return_value = "What time is it?"

        # Mock button to return True for the custom question button
        def button_side_effect(*args, **kwargs):
            if "Ask Tool Agent" in str(args[0]):
                return True
            return False

        mock_button.side_effect = button_side_effect

        # Mock agent and response
        mock_agent = Mock()
        mock_result = Mock()
        mock_result.output = "It's 2:30 PM"
        mock_agent.run_sync.return_value = mock_result
        mock_create_agent.return_value = mock_agent

        # Mock context managers
        mock_expander.return_value.__enter__ = Mock()
        mock_expander.return_value.__exit__ = Mock(return_value=None)
        mock_spinner.return_value.__enter__ = Mock()
        mock_spinner.return_value.__exit__ = Mock(return_value=None)

        tools_demo_section()

        mock_create_agent.assert_called_with("tools")
        mock_agent.run_sync.assert_called_with("What time is it?")


class TestSidebarFunctionality:
    """Test sidebar functionality within main function."""

    @patch('streamlit.set_page_config')
    @patch('streamlit.title')
    @patch('streamlit.markdown')
    @patch('streamlit.tabs')
    @patch('streamlit.sidebar')
    @patch('streamlit.header')
    @patch('streamlit.success')
    @patch('streamlit.info')
    @patch('streamlit.radio')
    @patch('streamlit.button')
    @patch('app.get_recommended_setup')
    def test_main_sidebar_ready_provider(self, mock_get_setup, mock_button, mock_radio, mock_info,
                                        mock_success, mock_header, mock_sidebar, mock_tabs,
                                        mock_markdown, mock_title, mock_set_config):
        """Test sidebar when provider is ready."""
        mock_get_setup.return_value = {
            "status": "ready",
            "recommended_provider": "openai",
            "recommended_model": "gpt-4"
        }
        mock_button.return_value = False
        mock_radio.return_value = "🎓 Beginner"

        # Mock context managers
        mock_sidebar.return_value.__enter__ = Mock()
        mock_sidebar.return_value.__exit__ = Mock(return_value=None)

        mock_tab_contexts = [MagicMock() for _ in range(4)]
        mock_tabs.return_value = mock_tab_contexts
        for mock_tab in mock_tab_contexts:
            mock_tab.__enter__.return_value = mock_tab
            mock_tab.__exit__.return_value = None

        # Mock the section functions
        with patch('app.tutorial_section'):
            with patch('app.playground_section'):
                with patch('app.math_demo_section'):
                    with patch('app.tools_demo_section'):
                        main()

        mock_success.assert_called_once_with("✅ Ready with openai")
        mock_info.assert_called_once_with("Model: gpt-4")

    @patch('streamlit.set_page_config')
    @patch('streamlit.title')
    @patch('streamlit.markdown')
    @patch('streamlit.tabs')
    @patch('streamlit.sidebar')
    @patch('streamlit.header')
    @patch('streamlit.error')
    @patch('streamlit.warning')
    @patch('streamlit.expander')
    @patch('streamlit.radio')
    @patch('streamlit.button')
    @patch('app.get_recommended_setup')
    def test_main_sidebar_not_ready_provider(self, mock_get_setup, mock_button, mock_radio,
                                            mock_expander, mock_warning, mock_error, mock_header,
                                            mock_sidebar, mock_tabs, mock_markdown, mock_title,
                                            mock_set_config):
        """Test sidebar when provider is not ready."""
        mock_get_setup.return_value = {"status": "not_ready"}
        mock_button.return_value = False
        mock_radio.return_value = "🎓 Beginner"

        # Mock context managers
        mock_sidebar.return_value.__enter__ = Mock()
        mock_sidebar.return_value.__exit__ = Mock(return_value=None)
        mock_expander.return_value.__enter__ = Mock()
        mock_expander.return_value.__exit__ = Mock(return_value=None)

        mock_tab_contexts = [MagicMock() for _ in range(4)]
        mock_tabs.return_value = mock_tab_contexts
        for mock_tab in mock_tab_contexts:
            mock_tab.__enter__.return_value = mock_tab
            mock_tab.__exit__.return_value = None

        # Mock the section functions
        with patch('app.tutorial_section'):
            with patch('app.playground_section'):
                with patch('app.math_demo_section'):
                    with patch('app.tools_demo_section'):
                        main()

        mock_error.assert_called_once_with("❌ No providers available")
        mock_warning.assert_called_once_with("Please configure API keys or start Ollama")

    @patch('streamlit.set_page_config')
    @patch('streamlit.title')
    @patch('streamlit.markdown')
    @patch('streamlit.tabs')
    @patch('streamlit.sidebar')
    @patch('streamlit.header')
    @patch('streamlit.success')
    @patch('streamlit.error')
    @patch('streamlit.radio')
    @patch('streamlit.button')
    @patch('streamlit.spinner')
    @patch('streamlit.subheader')
    @patch('app.get_recommended_setup')
    @patch('app.check_all_providers')
    def test_main_sidebar_health_check(self, mock_check_providers, mock_get_setup, mock_subheader,
                                      mock_spinner, mock_button, mock_radio, mock_error,
                                      mock_success, mock_header, mock_sidebar, mock_tabs,
                                      mock_markdown, mock_title, mock_set_config):
        """Test sidebar health check functionality."""
        mock_get_setup.return_value = {
            "status": "ready",
            "recommended_provider": "openai",
            "recommended_model": "gpt-4"
        }

        # Mock button to return True for health check button
        def button_side_effect(*args, **kwargs):
            if "Check Provider Health" in str(args[0]):
                return True
            return False

        mock_button.side_effect = button_side_effect
        mock_radio.return_value = "🎓 Beginner"

        # Mock health check results
        from models.health import HealthCheckResult, HealthStatus
        mock_results = {
            "openai": HealthCheckResult("openai", HealthStatus.HEALTHY, "OK"),
            "anthropic": HealthCheckResult("anthropic", HealthStatus.UNHEALTHY, "Failed")
        }
        mock_check_providers.return_value = mock_results

        # Mock context managers
        mock_sidebar.return_value.__enter__ = Mock()
        mock_sidebar.return_value.__exit__ = Mock(return_value=None)
        mock_spinner.return_value.__enter__ = Mock()
        mock_spinner.return_value.__exit__ = Mock(return_value=None)

        mock_tab_contexts = [MagicMock() for _ in range(4)]
        mock_tabs.return_value = mock_tab_contexts
        for mock_tab in mock_tab_contexts:
            mock_tab.__enter__.return_value = mock_tab
            mock_tab.__exit__.return_value = None

        # Mock the section functions
        with patch('app.tutorial_section'):
            with patch('app.playground_section'):
                with patch('app.math_demo_section'):
                    with patch('app.tools_demo_section'):
                        main()

        mock_check_providers.assert_called_once()


class TestErrorHandling:
    """Test error handling in app sections."""

    @patch('streamlit.header')
    @patch('streamlit.markdown')
    @patch('streamlit.columns')
    @patch('streamlit.subheader')
    @patch('streamlit.code')
    @patch('streamlit.button')
    @patch('streamlit.spinner')
    @patch('streamlit.error')
    @patch('app.get_recommended_setup')
    @patch('app.create_tutorial_agent')
    def test_math_demo_error_handling(self, mock_create_agent, mock_get_setup, mock_error,
                                     mock_spinner, mock_button, mock_code, mock_subheader,
                                     mock_columns, mock_markdown, mock_header):
        """Test error handling in math demo section."""
        mock_get_setup.return_value = {"status": "ready"}

        # Mock button to return True for basic agent button
        def button_side_effect(*args, **kwargs):
            if kwargs.get('key') == 'basic_math':
                return True
            return False

        mock_button.side_effect = button_side_effect
        mock_create_agent.side_effect = Exception("Test error")

        # Mock columns context managers
        mock_col_contexts = [MagicMock(), MagicMock()]
        mock_columns.return_value = mock_col_contexts

        for mock_col in mock_col_contexts:
            mock_col.__enter__.return_value = mock_col
            mock_col.__exit__.return_value = None

        # Mock spinner context manager
        mock_spinner.return_value.__enter__ = Mock()
        mock_spinner.return_value.__exit__ = Mock(return_value=None)

        math_demo_section()

        mock_error.assert_called_with("Error: Test error")

    @patch('streamlit.header')
    @patch('streamlit.markdown')
    @patch('streamlit.subheader')
    @patch('streamlit.expander')
    @patch('streamlit.button')
    @patch('streamlit.text_input')
    @patch('streamlit.spinner')
    @patch('streamlit.error')
    @patch('app.get_recommended_setup')
    @patch('app.create_tutorial_agent')
    def test_tools_demo_error_handling(self, mock_create_agent, mock_get_setup, mock_error,
                                      mock_spinner, mock_text_input, mock_button, mock_expander,
                                      mock_subheader, mock_markdown, mock_header):
        """Test error handling in tools demo section."""
        mock_get_setup.return_value = {"status": "ready"}
        mock_text_input.return_value = "test question"

        # Mock button to return True for custom question button
        def button_side_effect(*args, **kwargs):
            if "Ask Tool Agent" in str(args[0]):
                return True
            return False

        mock_button.side_effect = button_side_effect
        mock_create_agent.side_effect = Exception("Test error")

        # Mock context managers
        mock_expander.return_value.__enter__ = Mock()
        mock_expander.return_value.__exit__ = Mock(return_value=None)
        mock_spinner.return_value.__enter__ = Mock()
        mock_spinner.return_value.__exit__ = Mock(return_value=None)

        tools_demo_section()

        mock_error.assert_called_with("Error: Test error")