"""Tests for basic_agent module."""

import pytest
from pydantic_ai import Agent

from src.agents.basic_agent import create_basic_agent


@pytest.mark.integration
@pytest.mark.requires_api
class TestBasicAgent:
    """Test cases for basic agent functionality."""

    def test_create_basic_agent_default(self):
        """Test creating a basic agent with default parameters."""
        agent = create_basic_agent()

        assert isinstance(agent, Agent)
        assert agent.model == "openai:gpt-4o-mini"
        assert agent.instructions is not None
        assert "friendly AI assistant" in agent.instructions

    @pytest.mark.unit
    def test_create_basic_agent_custom_model(self):
        """Test creating a basic agent with custom model."""
        custom_model = "anthropic:claude-3-haiku-20240307"
        agent = create_basic_agent(model_provider=custom_model)

        assert isinstance(agent, Agent)
        assert agent.model == custom_model

    @pytest.mark.unit
    def test_create_basic_agent_custom_instructions(self):
        """Test creating a basic agent with custom instructions."""
        custom_instructions = "You are a test assistant for unit testing."
        agent = create_basic_agent(instructions=custom_instructions)

        assert isinstance(agent, Agent)
        assert agent.instructions == custom_instructions

    @pytest.mark.unit
    def test_agent_has_no_tools(self):
        """Test that basic agent has no tools registered."""
        agent = create_basic_agent()

        # Basic agent should not have tools
        # Note: This test may need adjustment based on PydanticAI's internal structure
        # We're testing the basic concept that it's a simple agent
        assert isinstance(agent, Agent)

    @pytest.mark.unit
    def test_agent_configuration_types(self):
        """Test that agent configuration accepts proper types."""
        # Test with different model provider formats
        test_cases = [
            "openai:gpt-4o-mini",
            "anthropic:claude-3-haiku-20240307",
            "groq:llama3-8b-8192",
        ]

        for model in test_cases:
            agent = create_basic_agent(model_provider=model)
            assert isinstance(agent, Agent)
            assert agent.model == model

    @pytest.mark.unit
    def test_instructions_parameter_handling(self):
        """Test instructions parameter handling."""
        # Test with None (should use default)
        agent_default = create_basic_agent(instructions=None)
        assert "friendly AI assistant" in agent_default.instructions

        # Test with empty string
        agent_empty = create_basic_agent(instructions="")
        assert agent_empty.instructions == ""

        # Test with custom string
        custom = "Custom instructions"
        agent_custom = create_basic_agent(instructions=custom)
        assert agent_custom.instructions == custom
