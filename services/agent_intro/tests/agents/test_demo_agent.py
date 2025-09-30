"""Tests for demo_agent module."""

from unittest.mock import patch

import pytest
from pydantic_ai import Agent

from src.agents.demo_agent import create_demo_agent


@pytest.mark.integration
@pytest.mark.requires_api
class TestDemoAgent:
    """Test cases for demo agent functionality."""

    def test_create_demo_agent_default(self):
        """Test creating a demo agent with default parameters."""
        agent = create_demo_agent()

        assert isinstance(agent, Agent)
        assert agent.model == "openai:gpt-4o-mini"
        assert "helpful assistant" in agent.instructions
        assert "utility tools" in agent.instructions

    def test_create_demo_agent_custom_model(self):
        """Test creating a demo agent with custom model."""
        custom_model = "anthropic:claude-3-haiku-20240307"
        agent = create_demo_agent(model_provider=custom_model)

        assert isinstance(agent, Agent)
        assert agent.model == custom_model

    def test_agent_instructions_content(self):
        """Test that the agent has appropriate instructions."""
        agent = create_demo_agent()

        instructions = agent.instructions
        assert "utility tools" in instructions.lower()
        assert "current time" in instructions.lower()
        assert "cannot perform mathematical calculations" in instructions.lower()

    @patch("src.agents.demo_agent.register_multiple_tools")
    def test_tools_registration_called(self, mock_register):
        """Test that tool registration is called during agent creation."""
        create_demo_agent()

        # Verify that register_multiple_tools was called
        mock_register.assert_called_once()

        # Verify the tools dict contains expected tools
        call_args = mock_register.call_args
        agent_arg, tools_dict = call_args[0]

        assert isinstance(agent_arg, Agent)
        assert "get_current_time" in tools_dict
        assert "get_fun_fact" in tools_dict
        assert "count_words" in tools_dict
        assert "reverse_text" in tools_dict
        assert "check_palindrome" in tools_dict

    def test_different_model_providers(self):
        """Test agent creation with different model providers."""
        providers = [
            "openai:gpt-4o-mini",
            "anthropic:claude-3-haiku-20240307",
            "groq:llama3-8b-8192",
        ]

        for provider in providers:
            agent = create_demo_agent(model_provider=provider)
            assert isinstance(agent, Agent)
            assert agent.model == provider

    def test_agent_configuration_immutable(self):
        """Test that multiple agent instances are independent."""
        agent1 = create_demo_agent()
        agent2 = create_demo_agent()

        # Agents should be separate instances
        assert agent1 is not agent2
        assert isinstance(agent1, Agent)
        assert isinstance(agent2, Agent)
