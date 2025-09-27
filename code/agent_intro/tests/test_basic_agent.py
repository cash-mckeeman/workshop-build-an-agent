"""
Tests for the BasicAgent class and core functionality.
"""

import pytest
from unittest.mock import Mock, patch
from agent_intro.basic_agent import BasicAgent
from agent_intro.tools import add, multiply, subtract, execute_tool


class TestBasicAgent:
    """Test cases for BasicAgent functionality."""

    @patch('agent_intro.basic_agent.get_hf_client')
    def test_agent_initialization(self, mock_get_hf_client):
        """Test that BasicAgent initializes correctly."""
        mock_get_hf_client.return_value = Mock()
        agent = BasicAgent()

        assert agent.model is not None
        assert len(agent.tools_schema) == 3
        assert len(agent.memory) == 0

        # Check tool names
        tool_names = [tool['function']['name'] for tool in agent.tools_schema]
        assert 'add' in tool_names
        assert 'multiply' in tool_names
        assert 'subtract' in tool_names

    @patch('agent_intro.basic_agent.get_hf_client')
    def test_add_user_message(self, mock_get_hf_client):
        """Test adding user messages to memory."""
        mock_get_hf_client.return_value = Mock()
        agent = BasicAgent()
        test_message = "Hello, how are you?"

        agent.add_user_message(test_message)

        assert len(agent.memory) == 1
        assert agent.memory[0]['role'] == 'user'
        assert agent.memory[0]['content'] == test_message

    @patch('agent_intro.basic_agent.get_hf_client')
    def test_clear_memory(self, mock_get_hf_client):
        """Test clearing agent memory."""
        mock_get_hf_client.return_value = Mock()
        agent = BasicAgent()
        agent.add_user_message("Test message")
        assert len(agent.memory) == 1

        agent.clear_memory()
        assert len(agent.memory) == 0

    @patch('agent_intro.basic_agent.get_hf_client')
    def test_get_conversation_history(self, mock_get_hf_client):
        """Test getting conversation history."""
        mock_get_hf_client.return_value = Mock()
        agent = BasicAgent()
        agent.add_user_message("Test message")

        history = agent.get_conversation_history()

        assert len(history) == 1
        assert history[0]['role'] == 'user'
        # Ensure it's a copy, not the original
        assert history is not agent.memory


class TestTools:
    """Test cases for agent tools."""

    def test_add_function(self):
        """Test the add function."""
        assert add(3, 12) == 15
        assert add("5", "7") == 12  # String inputs should work
        assert add(-2, 5) == 3

    def test_multiply_function(self):
        """Test the multiply function."""
        assert multiply(3, 4) == 12
        assert multiply("5", "6") == 30  # String inputs should work
        assert multiply(-2, 3) == -6

    def test_subtract_function(self):
        """Test the subtract function."""
        assert subtract(10, 3) == 7
        assert subtract("15", "8") == 7  # String inputs should work
        assert subtract(5, 10) == -5

    def test_execute_tool(self):
        """Test the execute_tool function."""
        assert execute_tool('add', {'a': 3, 'b': 12}) == 15
        assert execute_tool('multiply', {'a': 4, 'b': 5}) == 20
        assert execute_tool('subtract', {'a': 10, 'b': 3}) == 7

    def test_execute_tool_unknown(self):
        """Test execute_tool with unknown tool name."""
        with pytest.raises(ValueError, match="Unknown tool"):
            execute_tool('unknown_tool', {'a': 1, 'b': 2})


class TestIntegration:
    """Integration tests for the complete agent workflow."""

    @patch('agent_intro.basic_agent.get_hf_client')
    def test_tool_calling_workflow(self, mock_get_hf_client):
        """Test the complete tool calling workflow."""
        # Mock the chat model with proper responses
        mock_response_1 = Mock()
        mock_response_1.content = None
        mock_response_1.tool_calls = [{
            "id": "call_123",
            "name": "add",
            "args": {"a": 3, "b": 12}
        }]

        mock_model = Mock()
        mock_model.bind_tools.return_value.invoke.return_value = mock_response_1
        mock_get_hf_client.return_value = mock_model

        agent = BasicAgent()
        response = agent.chat("What is 3 plus 12?")

        # Verify the workflow
        assert "15" in response
        assert len(agent.memory) >= 3  # user message, assistant tool call, tool result, final response

        # Check memory structure
        assert agent.memory[0]['role'] == 'user'
        assert agent.memory[1]['role'] == 'assistant'
        assert agent.memory[2]['role'] == 'tool'
        assert agent.memory[2]['content'] == '15'

    @patch('agent_intro.basic_agent.get_hf_client')
    def test_chat_without_tools(self, mock_get_hf_client):
        """Test chat functionality without tool calling."""
        # Mock the chat model with simple response
        mock_response = Mock()
        mock_response.content = "Hello! I'm doing well, thank you for asking."
        mock_response.tool_calls = None

        mock_model = Mock()
        mock_model.bind_tools.return_value.invoke.return_value = mock_response
        mock_get_hf_client.return_value = mock_model

        agent = BasicAgent()
        response = agent.chat("How are you?")

        assert "Hello!" in response
        assert len(agent.memory) == 2  # user message and assistant response


if __name__ == "__main__":
    pytest.main([__file__])