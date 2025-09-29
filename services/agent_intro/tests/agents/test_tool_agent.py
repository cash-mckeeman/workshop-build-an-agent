"""Tests for tool_agent module."""

import pytest
from unittest.mock import patch, MagicMock
from pydantic_ai import Agent
from agents.tool_agent import create_tool_agent


class TestToolAgent:
    """Test cases for tool agent functionality."""

    def test_create_tool_agent_default(self):
        """Test creating a tool agent with default parameters."""
        agent = create_tool_agent()

        assert isinstance(agent, Agent)
        assert agent.model == "openai:gpt-4o-mini"
        assert "helpful assistant" in agent.instructions.lower()

    def test_create_tool_agent_custom_model(self):
        """Test creating a tool agent with custom model."""
        custom_model = "anthropic:claude-3-haiku-20240307"
        agent = create_tool_agent(model_provider=custom_model)

        assert isinstance(agent, Agent)
        assert agent.model == custom_model

    def test_create_tool_agent_without_math_tools(self):
        """Test creating a tool agent without math tools."""
        agent = create_tool_agent(include_math_tools=False)

        assert isinstance(agent, Agent)
        # Basic verification that agent is created properly

    def test_create_tool_agent_with_math_tools(self):
        """Test creating a tool agent with math tools included."""
        agent = create_tool_agent(include_math_tools=True)

        assert isinstance(agent, Agent)
        # Basic verification that agent is created properly

    @patch('agents.tool_agent.register_multiple_tools')
    def test_basic_tools_registration(self, mock_register):
        """Test that basic tools are registered."""
        create_tool_agent(include_math_tools=False)

        # Should be called once for basic tools
        assert mock_register.call_count == 1

        # Verify the basic tools dict contains expected tools
        call_args = mock_register.call_args_list[0]
        agent_arg, tools_dict = call_args[0]

        assert isinstance(agent_arg, Agent)
        expected_tools = {
            'get_current_time',
            'get_fun_fact',
            'count_words',
            'reverse_text',
            'check_palindrome'
        }
        assert set(tools_dict.keys()) == expected_tools

    @patch('agents.tool_agent.register_multiple_tools')
    def test_math_tools_registration(self, mock_register):
        """Test that math tools are registered when enabled."""
        create_tool_agent(include_math_tools=True)

        # Should be called twice: once for basic tools, once for math tools
        assert mock_register.call_count == 2

        # Check both calls
        basic_call = mock_register.call_args_list[0]
        math_call = mock_register.call_args_list[1]

        # Verify basic tools
        _, basic_tools = basic_call[0]
        expected_basic = {
            'get_current_time',
            'get_fun_fact',
            'count_words',
            'reverse_text',
            'check_palindrome'
        }
        assert set(basic_tools.keys()) == expected_basic

        # Verify math tools
        _, math_tools = math_call[0]
        expected_math = {'add', 'multiply'}
        assert set(math_tools.keys()) == expected_math

    def test_agent_instructions_content(self):
        """Test that the agent has appropriate instructions."""
        agent = create_tool_agent()

        instructions = agent.instructions
        assert "helpful assistant" in instructions.lower()
        assert "tools" in instructions.lower()

    def test_different_model_providers(self):
        """Test agent creation with different model providers."""
        providers = [
            "openai:gpt-4o-mini",
            "anthropic:claude-3-haiku-20240307",
            "groq:llama3-8b-8192"
        ]

        for provider in providers:
            agent = create_tool_agent(model_provider=provider)
            assert isinstance(agent, Agent)
            assert agent.model == provider

    def test_agent_instances_independent(self):
        """Test that multiple agent instances are independent."""
        agent1 = create_tool_agent()
        agent2 = create_tool_agent(include_math_tools=True)

        # Agents should be separate instances
        assert agent1 is not agent2
        assert isinstance(agent1, Agent)
        assert isinstance(agent2, Agent)

    def test_boolean_parameter_types(self):
        """Test that boolean parameters are handled correctly."""
        # Test with explicit False
        agent_false = create_tool_agent(include_math_tools=False)
        assert isinstance(agent_false, Agent)

        # Test with explicit True
        agent_true = create_tool_agent(include_math_tools=True)
        assert isinstance(agent_true, Agent)

        # Both should be valid agents but potentially different configurations
        assert agent_false is not agent_true