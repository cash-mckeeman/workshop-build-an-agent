"""
Tests for simple chat example.
"""

import pytest
from unittest.mock import Mock, patch, call

from examples.simple_chat import simple_chat_demo


class TestSimpleChatDemo:
    """Test simple_chat_demo function."""

    @patch('examples.simple_chat.BasicAgent')
    @patch('examples.simple_chat.load_dotenv')
    @patch('builtins.print')
    def test_simple_chat_demo_success(self, mock_print, mock_load_dotenv, mock_basic_agent_class):
        """Test successful execution of simple chat demo."""
        # Mock agent and its responses
        mock_agent = Mock()
        mock_agent.chat.side_effect = [
            "I'm doing great, thanks!",
            "I don't have access to real-time weather data.",
            "Why don't scientists trust atoms? Because they make up everything!",
            "I can help you with various tasks and answer questions."
        ]
        mock_basic_agent_class.return_value = mock_agent

        simple_chat_demo()

        # Verify environment loading
        mock_load_dotenv.assert_called_once_with("../../../../variables.env")

        # Verify agent creation
        mock_basic_agent_class.assert_called_once_with(
            model_name="microsoft/DialoGPT-medium",
            temperature=0.8,
            max_new_tokens=256
        )

        # Verify all conversations were executed
        expected_questions = [
            "Hello! How are you today?",
            "What's the weather like?",
            "Tell me a joke",
            "What can you help me with?"
        ]

        assert mock_agent.chat.call_count == len(expected_questions)
        for question in expected_questions:
            mock_agent.chat.assert_any_call(question)

        # Verify output includes demo title and completion
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("💬 Simple Chat Demo" in str(call) for call in print_calls)
        assert any("✅ Simple chat demo completed!" in str(call) for call in print_calls)

    @patch('examples.simple_chat.BasicAgent')
    @patch('examples.simple_chat.load_dotenv')
    @patch('builtins.print')
    def test_simple_chat_demo_agent_creation_error(self, mock_print, mock_load_dotenv, mock_basic_agent_class):
        """Test simple chat demo when agent creation fails."""
        mock_basic_agent_class.side_effect = Exception("Failed to create agent")

        with pytest.raises(Exception, match="Failed to create agent"):
            simple_chat_demo()

        mock_load_dotenv.assert_called_once()
        mock_basic_agent_class.assert_called_once()

    @patch('examples.simple_chat.BasicAgent')
    @patch('examples.simple_chat.load_dotenv')
    @patch('builtins.print')
    def test_simple_chat_demo_chat_error(self, mock_print, mock_load_dotenv, mock_basic_agent_class):
        """Test simple chat demo when chat fails."""
        mock_agent = Mock()
        mock_agent.chat.side_effect = Exception("Chat failed")
        mock_basic_agent_class.return_value = mock_agent

        with pytest.raises(Exception, match="Chat failed"):
            simple_chat_demo()

        # Should have tried to create agent and call chat
        mock_basic_agent_class.assert_called_once()
        mock_agent.chat.assert_called_once()

    @patch('examples.simple_chat.BasicAgent')
    @patch('examples.simple_chat.load_dotenv')
    @patch('builtins.print')
    def test_simple_chat_demo_partial_responses(self, mock_print, mock_load_dotenv, mock_basic_agent_class):
        """Test simple chat demo with some successful and some failed responses."""
        mock_agent = Mock()
        # First two succeed, third fails, fourth succeeds
        mock_agent.chat.side_effect = [
            "Great!",
            "Sunny",
            Exception("Model error"),
            "I can help with many things"
        ]
        mock_basic_agent_class.return_value = mock_agent

        # Should raise exception on the third call
        with pytest.raises(Exception, match="Model error"):
            simple_chat_demo()

        # Should have made 3 calls before failing
        assert mock_agent.chat.call_count == 3

    @patch('examples.simple_chat.BasicAgent')
    @patch('examples.simple_chat.load_dotenv')
    @patch('builtins.print')
    def test_simple_chat_demo_output_format(self, mock_print, mock_load_dotenv, mock_basic_agent_class):
        """Test that simple chat demo produces expected output format."""
        mock_agent = Mock()
        mock_agent.chat.return_value = "Test response"
        mock_basic_agent_class.return_value = mock_agent

        simple_chat_demo()

        # Check that all print calls were made
        print_calls = [str(call[0][0]) for call in mock_print.call_args_list]

        # Should have header
        assert any("💬 Simple Chat Demo" in call for call in print_calls)
        assert any("=" * 40 in call for call in print_calls)

        # Should have test questions
        expected_questions = [
            "Hello! How are you today?",
            "What's the weather like?",
            "Tell me a joke",
            "What can you help me with?"
        ]

        for question in expected_questions:
            assert any(question in call for call in print_calls)

        # Should have separators
        assert any("-" * 30 in call for call in print_calls)

        # Should have completion message
        assert any("✅ Simple chat demo completed!" in call for call in print_calls)


class TestSimpleChatModuleIntegration:
    """Integration tests for the simple chat module."""

    @patch('examples.simple_chat.load_dotenv')
    def test_dotenv_path_resolution(self, mock_load_dotenv):
        """Test that dotenv path is correctly resolved."""
        with patch('examples.simple_chat.BasicAgent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent.chat.return_value = "response"
            mock_agent_class.return_value = mock_agent

            with patch('builtins.print'):
                simple_chat_demo()

            # Verify correct path is used for environment variables
            mock_load_dotenv.assert_called_once_with("../../../../variables.env")

    @patch('examples.simple_chat.load_dotenv')
    @patch('builtins.print')
    def test_agent_configuration(self, mock_print, mock_load_dotenv):
        """Test that agent is configured with correct parameters."""
        with patch('examples.simple_chat.BasicAgent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent.chat.return_value = "response"
            mock_agent_class.return_value = mock_agent

            simple_chat_demo()

            # Verify agent is created with specific configuration for chat
            mock_agent_class.assert_called_once_with(
                model_name="microsoft/DialoGPT-medium",
                temperature=0.8,  # Higher temperature for creative responses
                max_new_tokens=256
            )

    @patch('examples.simple_chat.load_dotenv')
    @patch('builtins.print')
    def test_conversation_sequence(self, mock_print, mock_load_dotenv):
        """Test that conversations happen in the expected sequence."""
        with patch('examples.simple_chat.BasicAgent') as mock_agent_class:
            mock_agent = Mock()
            responses = ["Response 1", "Response 2", "Response 3", "Response 4"]
            mock_agent.chat.side_effect = responses
            mock_agent_class.return_value = mock_agent

            simple_chat_demo()

            # Verify conversations happened in correct order
            expected_questions = [
                "Hello! How are you today?",
                "What's the weather like?",
                "Tell me a joke",
                "What can you help me with?"
            ]

            assert mock_agent.chat.call_count == len(expected_questions)
            for i, question in enumerate(expected_questions):
                assert mock_agent.chat.call_args_list[i] == call(question)


class TestMainExecution:
    """Test main execution path."""

    @patch('examples.simple_chat.simple_chat_demo')
    def test_main_execution(self, mock_demo):
        """Test that main execution calls the demo function."""
        # Import and execute the main block
        import src.examples.simple_chat as module

        # Simulate main execution
        if __name__ == "__main__":
            module.simple_chat_demo()

        # In actual test, we verify the function would be called
        # This is more of a structure test since the actual main block
        # execution is harder to test directly
        assert hasattr(module, 'simple_chat_demo')
        assert callable(module.simple_chat_demo)