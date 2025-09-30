"""Tests for factory module."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic_ai import Agent

from src.factory import (
    AgentFactory,
    AgentType,
    create_agent,
    create_best_available_agent,
    create_tutorial_agent,
    get_recommended_setup,
)


@pytest.mark.unit
class TestAgentType:
    """Test cases for AgentType enum."""

    def test_agent_type_values(self):
        """Test that AgentType has expected values."""
        assert AgentType.BASIC == "basic"
        assert AgentType.MATH == "math"
        assert AgentType.DEMO == "demo"
        assert AgentType.TOOL == "tool"


@pytest.mark.unit
class TestAgentFactory:
    """Test cases for AgentFactory class."""

    @patch("src.factory.load_model_settings")
    @patch("src.factory.HealthChecker")
    def test_factory_initialization(self, mock_health_checker, mock_load_settings):
        """Test AgentFactory initialization."""
        factory = AgentFactory()

        assert factory.settings is not None
        assert factory.health_checker is not None
        mock_load_settings.assert_called_once()
        mock_health_checker.assert_called_once()

    @patch("src.factory.create_basic_agent")
    @patch("src.factory.get_best_available_provider")
    @patch("src.factory.load_model_settings")
    @patch("src.factory.HealthChecker")
    def test_create_agent_basic(
        self,
        mock_health_checker,
        mock_load_settings,
        mock_get_provider,
        mock_create_basic,
    ):
        """Test creating a basic agent."""
        mock_get_provider.return_value = "openai"
        mock_agent = MagicMock(spec=Agent)
        mock_create_basic.return_value = mock_agent

        factory = AgentFactory()
        agent = factory.create_agent(AgentType.BASIC)

        assert agent == mock_agent
        mock_create_basic.assert_called_once()

    @patch("src.factory.create_math_agent")
    @patch("src.factory.get_best_available_provider")
    @patch("src.factory.load_model_settings")
    @patch("src.factory.HealthChecker")
    def test_create_agent_math(
        self,
        mock_health_checker,
        mock_load_settings,
        mock_get_provider,
        mock_create_math,
    ):
        """Test creating a math agent."""
        mock_get_provider.return_value = "openai"
        mock_agent = MagicMock(spec=Agent)
        mock_create_math.return_value = mock_agent

        factory = AgentFactory()
        agent = factory.create_agent(AgentType.MATH)

        assert agent == mock_agent
        mock_create_math.assert_called_once()

    @patch("src.factory.create_demo_agent")
    @patch("src.factory.get_best_available_provider")
    @patch("src.factory.load_model_settings")
    @patch("src.factory.HealthChecker")
    def test_create_agent_demo(
        self,
        mock_health_checker,
        mock_load_settings,
        mock_get_provider,
        mock_create_demo,
    ):
        """Test creating a demo agent."""
        mock_get_provider.return_value = "openai"
        mock_agent = MagicMock(spec=Agent)
        mock_create_demo.return_value = mock_agent

        factory = AgentFactory()
        agent = factory.create_agent(AgentType.DEMO)

        assert agent == mock_agent
        mock_create_demo.assert_called_once()

    @patch("src.factory.create_tool_agent")
    @patch("src.factory.get_best_available_provider")
    @patch("src.factory.load_model_settings")
    @patch("src.factory.HealthChecker")
    def test_create_agent_tool(
        self,
        mock_health_checker,
        mock_load_settings,
        mock_get_provider,
        mock_create_tool,
    ):
        """Test creating a tool agent."""
        mock_get_provider.return_value = "openai"
        mock_agent = MagicMock(spec=Agent)
        mock_create_tool.return_value = mock_agent

        factory = AgentFactory()
        agent = factory.create_agent(AgentType.TOOL)

        assert agent == mock_agent
        mock_create_tool.assert_called_once()

    @patch("src.factory.get_best_available_provider")
    @patch("src.factory.load_model_settings")
    @patch("src.factory.HealthChecker")
    def test_create_agent_no_provider_available(
        self, mock_health_checker, mock_load_settings, mock_get_provider
    ):
        """Test creating agent when no provider is available."""
        mock_get_provider.return_value = None

        factory = AgentFactory()

        with pytest.raises(ValueError, match="No healthy providers available"):
            factory.create_agent(AgentType.BASIC)

    @patch("src.factory.get_provider_config")
    @patch("src.factory.load_model_settings")
    @patch("src.factory.HealthChecker")
    def test_create_agent_with_specific_provider(
        self, mock_health_checker, mock_load_settings, mock_get_provider_config
    ):
        """Test creating agent with specific provider."""
        mock_provider = MagicMock()
        mock_provider.is_available = True
        mock_provider.get_model_id.return_value = "openai:gpt-4o-mini"
        mock_get_provider_config.return_value = mock_provider

        factory = AgentFactory()

        with patch("src.factory.create_basic_agent") as mock_create:
            mock_agent = MagicMock(spec=Agent)
            mock_create.return_value = mock_agent

            agent = factory.create_agent(AgentType.BASIC, provider_name="openai")

            assert agent == mock_agent
            mock_get_provider_config.assert_called_with("openai")

    @patch("src.factory.get_available_providers")
    @patch("src.factory.load_model_settings")
    @patch("src.factory.HealthChecker")
    def test_get_available_configurations(
        self, mock_health_checker, mock_load_settings, mock_get_available
    ):
        """Test getting available configurations."""
        mock_provider = MagicMock()
        mock_provider.available_models = ["model1", "model2"]
        mock_get_available.return_value = {"openai": mock_provider}

        factory = AgentFactory()
        configs = factory.get_available_configurations()

        assert "openai" in configs
        assert configs["openai"] == ["model1", "model2"]

    @patch("src.factory.get_best_available_provider")
    @patch("src.factory.load_model_settings")
    @patch("src.factory.HealthChecker")
    def test_create_tutorial_agent(
        self, mock_health_checker, mock_load_settings, mock_get_provider
    ):
        """Test creating a tutorial agent."""
        mock_get_provider.return_value = "openai"

        factory = AgentFactory()

        with patch.object(factory, "create_agent") as mock_create:
            mock_agent = MagicMock(spec=Agent)
            mock_create.return_value = mock_agent

            agent = factory.create_tutorial_agent("math")

            assert agent == mock_agent
            # When no provider_preference is given, provider_name is None
            mock_create.assert_called_once_with(AgentType.MATH, provider_name=None)

    @patch("src.factory.load_model_settings")
    @patch("src.factory.HealthChecker")
    def test_create_tutorial_agent_invalid_step(
        self, mock_health_checker, mock_load_settings
    ):
        """Test creating tutorial agent with invalid step."""
        factory = AgentFactory()

        with pytest.raises(ValueError, match="Unknown tutorial step"):
            factory.create_tutorial_agent("invalid_step")


@pytest.mark.unit
class TestConvenienceFunctions:
    """Test cases for convenience functions."""

    @patch("src.factory.AgentFactory")
    def test_create_agent_convenience(self, mock_factory_class):
        """Test create_agent convenience function."""
        mock_factory = MagicMock()
        mock_agent = MagicMock(spec=Agent)
        mock_factory.create_agent.return_value = mock_agent
        mock_factory_class.return_value = mock_factory

        agent = create_agent("basic")

        assert agent == mock_agent
        mock_factory.create_agent.assert_called_once_with(AgentType.BASIC, None, None)

    @patch("src.factory.AgentFactory")
    def test_create_tutorial_agent_convenience(self, mock_factory_class):
        """Test create_tutorial_agent convenience function."""
        mock_factory = MagicMock()
        mock_agent = MagicMock(spec=Agent)
        mock_factory.create_tutorial_agent.return_value = mock_agent
        mock_factory_class.return_value = mock_factory

        agent = create_tutorial_agent("math")

        assert agent == mock_agent
        mock_factory.create_tutorial_agent.assert_called_once_with("math")

    @patch("src.factory.AgentFactory")
    def test_create_best_available_agent_convenience(self, mock_factory_class):
        """Test create_best_available_agent convenience function."""
        mock_factory = MagicMock()
        mock_agent = MagicMock(spec=Agent)
        mock_factory.create_agent.return_value = mock_agent
        mock_factory_class.return_value = mock_factory

        agent = create_best_available_agent("tool")

        assert agent == mock_agent
        mock_factory.create_agent.assert_called_once_with(AgentType.TOOL)

    @patch("src.factory.get_best_available_provider")
    @patch("src.factory.get_provider_config")
    def test_get_recommended_setup_available(self, mock_get_config, mock_get_provider):
        """Test get_recommended_setup with available provider."""
        mock_get_provider.return_value = "openai"
        mock_provider = MagicMock()
        mock_provider.default_model = "gpt-4o-mini"
        mock_provider.description = "OpenAI models"
        mock_provider.get_model_id.return_value = "openai:gpt-4o-mini"
        mock_get_config.return_value = mock_provider

        setup = get_recommended_setup()

        assert setup["status"] == "ready"
        assert setup["recommended_provider"] == "openai"
        assert setup["recommended_model"] == "gpt-4o-mini"
        assert setup["provider_description"] == "OpenAI models"
        assert setup["full_model_id"] == "openai:gpt-4o-mini"
        assert "tutorial_steps" in setup

    @patch("src.factory.get_best_available_provider")
    def test_get_recommended_setup_no_providers(self, mock_get_provider):
        """Test get_recommended_setup with no available providers."""
        mock_get_provider.return_value = None

        setup = get_recommended_setup()

        assert setup["status"] == "no_providers"
        assert "No providers available" in setup["message"]
        assert "setup_instructions" in setup

    def test_agent_type_enum_completeness(self):
        """Test that all expected agent types are available."""
        expected_types = {"basic", "math", "demo", "tool"}
        actual_types = {t.value for t in AgentType}

        assert actual_types == expected_types
