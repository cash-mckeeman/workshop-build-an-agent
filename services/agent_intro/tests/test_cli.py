"""
Tests for the CLI module.
"""

import pytest
import argparse
import sys
from unittest.mock import Mock, patch, call
from io import StringIO

from cli import (
    cmd_status,
    cmd_demo,
    cmd_interactive,
    cmd_tutorial,
    main
)


class TestCmdStatus:
    """Test cmd_status function."""

    @patch('cli.print_health_status')
    @patch('cli.get_recommended_setup')
    def test_cmd_status_ready(self, mock_get_setup, mock_print_health, capsys):
        """Test status command when system is ready."""
        mock_get_setup.return_value = {
            'status': 'ready',
            'recommended_provider': 'openai',
            'recommended_model': 'gpt-4'
        }

        cmd_status()

        captured = capsys.readouterr()
        assert "Agent Intro System Status" in captured.out
        assert "Setup Status: ready" in captured.out
        assert "✅ Recommended Provider: openai" in captured.out
        assert "✅ Recommended Model: gpt-4" in captured.out
        mock_print_health.assert_called_once()

    @patch('cli.print_health_status')
    @patch('cli.get_recommended_setup')
    def test_cmd_status_not_ready(self, mock_get_setup, mock_print_health, capsys):
        """Test status command when system is not ready."""
        mock_get_setup.return_value = {
            'status': 'no_providers'
        }

        cmd_status()

        captured = capsys.readouterr()
        assert "Agent Intro System Status" in captured.out
        assert "Setup Status: no_providers" in captured.out
        assert "❌ No providers available" in captured.out
        mock_print_health.assert_called_once()


class TestCmdDemo:
    """Test cmd_demo function."""

    @patch('cli.create_tutorial_agent')
    @patch('cli.get_recommended_setup')
    def test_cmd_demo_success(self, mock_get_setup, mock_create_agent, capsys):
        """Test successful demo execution."""
        mock_get_setup.return_value = {
            'status': 'ready',
            'recommended_provider': 'openai',
            'recommended_model': 'gpt-4'
        }

        mock_agent = Mock()
        mock_result = Mock()
        mock_result.output = "The answer is 15"
        mock_agent.run_sync.return_value = mock_result
        mock_create_agent.return_value = mock_agent

        cmd_demo("math", "What is 3 plus 12?")

        captured = capsys.readouterr()
        assert "🤖 Running math agent demo..." in captured.out
        assert "Provider: openai" in captured.out
        assert "Model: gpt-4" in captured.out
        assert "Question: What is 3 plus 12?" in captured.out
        assert "🤖 Response: The answer is 15" in captured.out

        mock_create_agent.assert_called_once_with("math")
        mock_agent.run_sync.assert_called_once_with("What is 3 plus 12?")

    @patch('cli.get_recommended_setup')
    def test_cmd_demo_not_ready(self, mock_get_setup, capsys):
        """Test demo when system is not ready."""
        mock_get_setup.return_value = {
            'status': 'no_providers'
        }

        cmd_demo("math")

        captured = capsys.readouterr()
        assert "❌ No providers available" in captured.out

    def test_cmd_demo_default_questions(self):
        """Test that default questions are used correctly."""
        with patch('cli.get_recommended_setup') as mock_get_setup:
            mock_get_setup.return_value = {'status': 'no_providers'}

            # Test that we don't call create_agent when not ready
            with patch('cli.create_tutorial_agent') as mock_create_agent:
                cmd_demo("basic")
                mock_create_agent.assert_not_called()

    @patch('cli.create_tutorial_agent')
    @patch('cli.get_recommended_setup')
    def test_cmd_demo_exception_handling(self, mock_get_setup, mock_create_agent, capsys):
        """Test demo exception handling."""
        mock_get_setup.return_value = {
            'status': 'ready',
            'recommended_provider': 'openai',
            'recommended_model': 'gpt-4'
        }

        mock_create_agent.side_effect = Exception("Test error")

        cmd_demo("math")

        captured = capsys.readouterr()
        assert "❌ Error: Test error" in captured.out


class TestCmdInteractive:
    """Test cmd_interactive function."""

    @patch('cli.get_recommended_setup')
    def test_cmd_interactive_not_ready(self, mock_get_setup, capsys):
        """Test interactive mode when system is not ready."""
        mock_get_setup.return_value = {
            'status': 'no_providers'
        }

        cmd_interactive("tools")

        captured = capsys.readouterr()
        assert "❌ No providers available" in captured.out

    @patch('builtins.input')
    @patch('cli.create_tutorial_agent')
    @patch('cli.get_recommended_setup')
    def test_cmd_interactive_quit(self, mock_get_setup, mock_create_agent, mock_input, capsys):
        """Test interactive mode with quit command."""
        mock_get_setup.return_value = {
            'status': 'ready',
            'recommended_provider': 'openai',
            'recommended_model': 'gpt-4'
        }

        mock_agent = Mock()
        mock_create_agent.return_value = mock_agent
        mock_input.return_value = "quit"

        cmd_interactive("tools")

        captured = capsys.readouterr()
        assert "🎮 Interactive tools Agent Session" in captured.out
        assert "👋 Goodbye!" in captured.out

    @patch('builtins.input')
    @patch('cli.create_tutorial_agent')
    @patch('cli.get_recommended_setup')
    def test_cmd_interactive_conversation(self, mock_get_setup, mock_create_agent, mock_input, capsys):
        """Test interactive conversation flow."""
        mock_get_setup.return_value = {
            'status': 'ready',
            'recommended_provider': 'openai',
            'recommended_model': 'gpt-4'
        }

        mock_agent = Mock()
        mock_result = Mock()
        mock_result.output = "Hello there!"
        mock_result.new_messages.return_value = [{"role": "user", "content": "hi"}]
        mock_agent.run_sync.return_value = mock_result
        mock_create_agent.return_value = mock_agent

        mock_input.side_effect = ["hi", "exit"]

        cmd_interactive("tools")

        captured = capsys.readouterr()
        assert "🎮 Interactive tools Agent Session" in captured.out
        assert "🤖: Hello there!" in captured.out
        assert "👋 Goodbye!" in captured.out

        # Check that agent was called with message history
        mock_agent.run_sync.assert_called_with("hi", message_history=[])

    @patch('builtins.input')
    @patch('cli.create_tutorial_agent')
    @patch('cli.get_recommended_setup')
    def test_cmd_interactive_empty_input(self, mock_get_setup, mock_create_agent, mock_input, capsys):
        """Test interactive mode with empty input."""
        mock_get_setup.return_value = {
            'status': 'ready',
            'recommended_provider': 'openai',
            'recommended_model': 'gpt-4'
        }

        mock_agent = Mock()
        mock_create_agent.return_value = mock_agent
        mock_input.side_effect = ["", "  ", "quit"]

        cmd_interactive("tools")

        # Agent should not be called for empty inputs
        mock_agent.run_sync.assert_not_called()

    @patch('builtins.input')
    @patch('cli.create_tutorial_agent')
    @patch('cli.get_recommended_setup')
    def test_cmd_interactive_keyboard_interrupt(self, mock_get_setup, mock_create_agent, mock_input, capsys):
        """Test interactive mode with keyboard interrupt."""
        mock_get_setup.return_value = {
            'status': 'ready',
            'recommended_provider': 'openai',
            'recommended_model': 'gpt-4'
        }

        mock_agent = Mock()
        mock_create_agent.return_value = mock_agent
        mock_input.side_effect = KeyboardInterrupt()

        cmd_interactive("tools")

        captured = capsys.readouterr()
        assert "👋 Goodbye!" in captured.out

    @patch('cli.get_recommended_setup')
    def test_cmd_interactive_agent_creation_error(self, mock_get_setup, capsys):
        """Test interactive mode when agent creation fails."""
        mock_get_setup.return_value = {
            'status': 'ready',
            'recommended_provider': 'openai',
            'recommended_model': 'gpt-4'
        }

        with patch('cli.create_tutorial_agent') as mock_create_agent:
            mock_create_agent.side_effect = Exception("Failed to create agent")

            cmd_interactive("tools")

            captured = capsys.readouterr()
            assert "❌ Failed to create agent: Failed to create agent" in captured.out


class TestCmdTutorial:
    """Test cmd_tutorial function."""

    @patch('cli.get_recommended_setup')
    def test_cmd_tutorial_not_ready(self, mock_get_setup, capsys):
        """Test tutorial when system is not ready."""
        mock_get_setup.return_value = {
            'status': 'no_providers'
        }

        cmd_tutorial()

        captured = capsys.readouterr()
        assert "📚 Agent Tutorial Progression" in captured.out
        assert "❌ No providers available" in captured.out
        assert "Setup Instructions:" in captured.out

    @patch('cli.create_tutorial_agent')
    @patch('cli.get_recommended_setup')
    def test_cmd_tutorial_success(self, mock_get_setup, mock_create_agent, capsys):
        """Test successful tutorial execution."""
        mock_get_setup.return_value = {
            'status': 'ready',
            'recommended_provider': 'openai',
            'recommended_model': 'gpt-4'
        }

        # Mock different agents and their responses
        mock_basic_agent = Mock()
        mock_basic_result = Mock()
        mock_basic_result.output = "I am a basic agent"
        mock_basic_agent.run_sync.return_value = mock_basic_result

        mock_math_agent = Mock()
        mock_math_result = Mock()
        mock_math_result.output = "15"
        mock_math_agent.run_sync.return_value = mock_math_result

        mock_tools_agent = Mock()
        mock_tools_result = Mock()
        mock_tools_result.output = "It's 2:30 PM and 'hello' backwards is 'olleh'"
        mock_tools_agent.run_sync.return_value = mock_tools_result

        mock_create_agent.side_effect = [mock_basic_agent, mock_math_agent, mock_tools_agent]

        cmd_tutorial()

        captured = capsys.readouterr()
        assert "📚 Agent Tutorial Progression" in captured.out
        assert "Step 1: Basic Agent" in captured.out
        assert "Step 2: Math Agent" in captured.out
        assert "Step 3: Tool Agent" in captured.out
        assert "I am a basic agent" in captured.out
        assert "15" in captured.out
        assert "It's 2:30 PM and 'hello' backwards is 'olleh'" in captured.out

        # Verify all three agent types were created
        expected_calls = [call("basic"), call("math"), call("tools")]
        mock_create_agent.assert_has_calls(expected_calls)

    @patch('cli.create_tutorial_agent')
    @patch('cli.get_recommended_setup')
    def test_cmd_tutorial_with_errors(self, mock_get_setup, mock_create_agent, capsys):
        """Test tutorial with agent creation errors."""
        mock_get_setup.return_value = {
            'status': 'ready',
            'recommended_provider': 'openai',
            'recommended_model': 'gpt-4'
        }

        mock_create_agent.side_effect = Exception("Agent creation failed")

        cmd_tutorial()

        captured = capsys.readouterr()
        assert "Error: Agent creation failed" in captured.out


class TestMain:
    """Test main CLI function."""

    @patch('cli.cmd_status')
    @patch('sys.argv', ['cli.py', 'status'])
    def test_main_status_command(self, mock_cmd_status):
        """Test main function with status command."""
        main()
        mock_cmd_status.assert_called_once()

    @patch('cli.cmd_demo')
    @patch('sys.argv', ['cli.py', 'demo'])
    def test_main_demo_command_default(self, mock_cmd_demo):
        """Test main function with demo command using defaults."""
        main()
        mock_cmd_demo.assert_called_once_with('math', None)

    @patch('cli.cmd_demo')
    @patch('sys.argv', ['cli.py', 'demo', '--type', 'tools', '--question', 'test question'])
    def test_main_demo_command_with_args(self, mock_cmd_demo):
        """Test main function with demo command with arguments."""
        main()
        mock_cmd_demo.assert_called_once_with('tools', 'test question')

    @patch('cli.cmd_interactive')
    @patch('sys.argv', ['cli.py', 'interactive'])
    def test_main_interactive_command_default(self, mock_cmd_interactive):
        """Test main function with interactive command using defaults."""
        main()
        mock_cmd_interactive.assert_called_once_with('tools')

    @patch('cli.cmd_interactive')
    @patch('sys.argv', ['cli.py', 'interactive', '--type', 'basic'])
    def test_main_interactive_command_with_args(self, mock_cmd_interactive):
        """Test main function with interactive command with arguments."""
        main()
        mock_cmd_interactive.assert_called_once_with('basic')

    @patch('cli.cmd_tutorial')
    @patch('sys.argv', ['cli.py', 'tutorial'])
    def test_main_tutorial_command(self, mock_cmd_tutorial):
        """Test main function with tutorial command."""
        main()
        mock_cmd_tutorial.assert_called_once()

    @patch('subprocess.run')
    @patch('sys.argv', ['cli.py', 'app'])
    def test_main_app_command(self, mock_subprocess_run):
        """Test main function with app command."""
        main()
        mock_subprocess_run.assert_called_once_with([
            sys.executable, '-m', 'streamlit', 'run', 'src/agent_intro/app.py'
        ])

    @patch('sys.argv', ['cli.py'])
    def test_main_no_command(self, capsys):
        """Test main function with no command (should print help)."""
        with patch('argparse.ArgumentParser.print_help') as mock_print_help:
            main()
            mock_print_help.assert_called_once()

    @patch('sys.argv', ['cli.py', '--help'])
    def test_main_help_flag(self):
        """Test main function with help flag."""
        with pytest.raises(SystemExit):
            main()

    @patch('sys.argv', ['cli.py', 'demo', '--type', 'invalid'])
    def test_main_invalid_agent_type(self):
        """Test main function with invalid agent type."""
        with pytest.raises(SystemExit):
            main()


class TestDefaultQuestions:
    """Test default question selection in cmd_demo."""

    @patch('cli.create_tutorial_agent')
    @patch('cli.get_recommended_setup')
    def test_default_question_basic(self, mock_get_setup, mock_create_agent, capsys):
        """Test default question for basic agent."""
        mock_get_setup.return_value = {
            'status': 'ready',
            'recommended_provider': 'openai',
            'recommended_model': 'gpt-4'
        }

        mock_agent = Mock()
        mock_result = Mock()
        mock_result.output = "I am an AI assistant"
        mock_agent.run_sync.return_value = mock_result
        mock_create_agent.return_value = mock_agent

        cmd_demo("basic")

        captured = capsys.readouterr()
        assert "Hello! Can you tell me what you are?" in captured.out

    @patch('cli.create_tutorial_agent')
    @patch('cli.get_recommended_setup')
    def test_default_question_math(self, mock_get_setup, mock_create_agent, capsys):
        """Test default question for math agent."""
        mock_get_setup.return_value = {
            'status': 'ready',
            'recommended_provider': 'openai',
            'recommended_model': 'gpt-4'
        }

        mock_agent = Mock()
        mock_result = Mock()
        mock_result.output = "15"
        mock_agent.run_sync.return_value = mock_result
        mock_create_agent.return_value = mock_agent

        cmd_demo("math")

        captured = capsys.readouterr()
        assert "What is 3 plus 12?" in captured.out

    @patch('cli.create_tutorial_agent')
    @patch('cli.get_recommended_setup')
    def test_default_question_tools(self, mock_get_setup, mock_create_agent, capsys):
        """Test default question for tools agent."""
        mock_get_setup.return_value = {
            'status': 'ready',
            'recommended_provider': 'openai',
            'recommended_model': 'gpt-4'
        }

        mock_agent = Mock()
        mock_result = Mock()
        mock_result.output = "It's 2:30 PM and 35"
        mock_agent.run_sync.return_value = mock_result
        mock_create_agent.return_value = mock_agent

        cmd_demo("tools")

        captured = capsys.readouterr()
        assert "What time is it and what's 5 multiplied by 7?" in captured.out

    @patch('cli.create_tutorial_agent')
    @patch('cli.get_recommended_setup')
    def test_fallback_question(self, mock_get_setup, mock_create_agent, capsys):
        """Test fallback question for unknown agent type."""
        mock_get_setup.return_value = {
            'status': 'ready',
            'recommended_provider': 'openai',
            'recommended_model': 'gpt-4'
        }

        mock_agent = Mock()
        mock_result = Mock()
        mock_result.output = "Hello!"
        mock_agent.run_sync.return_value = mock_result
        mock_create_agent.return_value = mock_agent

        cmd_demo("unknown_type")

        captured = capsys.readouterr()
        assert "Hello!" in captured.out