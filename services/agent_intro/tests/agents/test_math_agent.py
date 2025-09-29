"""Tests for math_agent module."""

import pytest
from unittest.mock import patch, MagicMock
from pydantic_ai import Agent
from agents.math_agent import create_math_agent


class TestMathAgent:
    """Test cases for math agent functionality."""

    def test_create_math_agent_default(self):
        """Test creating a math agent with default parameters."""
        agent = create_math_agent()

        assert isinstance(agent, Agent)
        assert agent.model == "openai:gpt-4o-mini"
        assert "math assistant" in agent.instructions

    def test_create_math_agent_custom_model(self):
        """Test creating a math agent with custom model."""
        custom_model = "anthropic:claude-3-haiku-20240307"
        agent = create_math_agent(model_provider=custom_model)

        assert isinstance(agent, Agent)
        assert agent.model == custom_model

    def test_agent_has_math_tools(self):
        """Test that math agent has the expected tools."""
        agent = create_math_agent()

        # The agent should be created and configured with math tools
        assert isinstance(agent, Agent)
        # Note: Testing tool registration would require more complex mocking
        # or integration tests. For unit tests, we verify the agent is created properly.

    def test_agent_instructions_content(self):
        """Test that the agent has appropriate instructions."""
        agent = create_math_agent()

        instructions = agent.instructions
        assert "math assistant" in instructions.lower()
        assert "tools" in instructions.lower()
        assert "calculations" in instructions.lower()

    @patch('agents.math_agent.register_multiple_tools')
    def test_tools_registration_called(self, mock_register):
        """Test that tool registration is called during agent creation."""
        create_math_agent()

        # Verify that register_multiple_tools was called
        mock_register.assert_called_once()

        # Verify the tools dict contains expected tools
        call_args = mock_register.call_args
        agent_arg, tools_dict = call_args[0]

        assert isinstance(agent_arg, Agent)
        assert 'add' in tools_dict
        assert 'multiply' in tools_dict

    def test_different_model_providers(self):
        """Test agent creation with different model providers."""
        providers = [
            "openai:gpt-4o-mini",
            "anthropic:claude-3-haiku-20240307",
            "groq:llama3-8b-8192"
        ]

        for provider in providers:
            agent = create_math_agent(model_provider=provider)
            assert isinstance(agent, Agent)
            assert agent.model == provider

    def test_agent_configuration_immutable(self):
        """Test that multiple agent instances are independent."""
        agent1 = create_math_agent()
        agent2 = create_math_agent()

        # Agents should be separate instances
        assert agent1 is not agent2
        assert isinstance(agent1, Agent)
        assert isinstance(agent2, Agent)