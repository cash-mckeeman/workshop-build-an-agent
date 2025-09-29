"""
Agent Factory for PydanticAI Tutorial

This module provides a centralized factory for creating different types of agents
with automatic provider selection, health checking, and configuration management.
"""

from typing import Optional, Dict, Any, List
from enum import Enum

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from models import get_available_providers, get_provider_config, load_model_settings
from models.health import get_best_available_provider, HealthChecker
from agents.basic_agent import create_basic_agent
from agents.math_agent import create_math_agent
from agents.demo_agent import create_demo_agent
from agents.tool_agent import create_tool_agent


class AgentType(str, Enum):
    """Types of agents that can be created."""
    BASIC = "basic"
    MATH = "math"
    DEMO = "demo"
    TOOL = "tool"


class AgentFactory:
    """Factory for creating configured PydanticAI agents."""

    def __init__(self):
        self.settings = load_model_settings()
        self.health_checker = HealthChecker()

    def create_agent(
        self,
        agent_type: AgentType,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs
    ) -> Agent[None, str]:
        """Create an agent of the specified type.

        Args:
            agent_type: Type of agent to create
            provider_name: Specific provider to use (auto-selected if None)
            model_name: Specific model to use (provider default if None)
            **kwargs: Additional arguments passed to agent creation

        Returns:
            Configured PydanticAI agent

        Raises:
            ValueError: If no suitable provider is available
            RuntimeError: If agent creation fails
        """
        # Get the model provider string
        model_provider = self._get_model_provider(provider_name, model_name)

        # Create the appropriate agent type
        try:
            if agent_type == AgentType.BASIC:
                return create_basic_agent(model_provider, **kwargs)
            elif agent_type == AgentType.MATH:
                return create_math_agent(model_provider, **kwargs)
            elif agent_type == AgentType.DEMO:
                return create_demo_agent(model_provider, **kwargs)
            elif agent_type == AgentType.TOOL:
                return create_tool_agent(model_provider, include_math_tools=True, **kwargs)
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")

        except Exception as e:
            raise RuntimeError(f"Failed to create {agent_type} agent: {str(e)}") from e

    def _get_model_provider(
        self,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        """Get the configured model for PydanticAI.

        Args:
            provider_name: Specific provider to use
            model_name: Specific model to use

        Returns:
            Configured model object or model string

        Raises:
            ValueError: If no suitable provider is available
        """
        # If provider is specified, use it
        if provider_name:
            provider = get_provider_config(provider_name)
            if not provider:
                raise ValueError(f"Provider '{provider_name}' not found")

            if not provider.is_available:
                raise ValueError(f"Provider '{provider_name}' is not available")

            model = model_name or provider.default_model

            # Special handling for Ollama
            if provider.provider_type.value == "ollama":
                return self._create_ollama_model(model, provider.base_url or "http://localhost:11434")

            # For other providers, use the string format
            return provider.get_model_id(model)

        # Otherwise, auto-select the best available provider
        best_provider = get_best_available_provider()
        if not best_provider:
            raise ValueError(
                "No healthy providers available. Please configure API keys or start Ollama."
            )

        provider = get_provider_config(best_provider)
        model = model_name or provider.default_model

        # Special handling for Ollama
        if provider.provider_type.value == "ollama":
            return self._create_ollama_model(model, provider.base_url or "http://localhost:11434")

        return provider.get_model_id(model)

    def _create_ollama_model(self, model_name: str, base_url: str) -> OpenAIChatModel:
        """Create an Ollama model using OpenAI-compatible API.

        Args:
            model_name: Name of the Ollama model
            base_url: Base URL for Ollama server

        Returns:
            Configured OpenAIChatModel for Ollama
        """
        # Ensure base_url has /v1 suffix for OpenAI compatibility
        if not base_url.endswith('/v1'):
            base_url = f"{base_url}/v1"

        # Remove :latest tag if present, as Ollama API doesn't need it
        if model_name.endswith(':latest'):
            model_name = model_name.replace(':latest', '')

        return OpenAIChatModel(
            model_name=model_name,
            provider=OpenAIProvider(base_url=base_url)
        )

    def get_available_configurations(self) -> Dict[str, List[str]]:
        """Get all available provider and model configurations.

        Returns:
            Dictionary mapping provider names to available models
        """
        available_providers = get_available_providers()
        return {
            name: provider.available_models
            for name, provider in available_providers.items()
        }

    def create_tutorial_agent(
        self,
        step: str,
        provider_preference: Optional[List[str]] = None
    ) -> Agent[None, str]:
        """Create an agent configured for a specific tutorial step.

        Args:
            step: Tutorial step ("basic", "math", "tools")
            provider_preference: Preferred providers in order

        Returns:
            Configured agent for the tutorial step
        """
        # Determine agent type from step
        step_to_type = {
            "basic": AgentType.BASIC,
            "math": AgentType.MATH,
            "demo": AgentType.DEMO,
            "tools": AgentType.TOOL,
            "tool": AgentType.TOOL,  # Alternative name
        }

        agent_type = step_to_type.get(step.lower())
        if not agent_type:
            raise ValueError(f"Unknown tutorial step: {step}")

        # Get best provider from preference list
        provider_name = None
        if provider_preference:
            best_provider = get_best_available_provider(provider_preference)
            if best_provider:
                provider_name = best_provider

        return self.create_agent(agent_type, provider_name=provider_name)

    def create_math_tutorial_agent(
        self,
        provider_name: Optional[str] = None
    ) -> Agent[None, str]:
        """Create the exact math agent from the original tutorial.

        This replicates the original intro_to_agents.ipynb workflow:
        - Agent with add() tool
        - "What is 3 plus 12?" question
        - All four components demonstration

        Args:
            provider_name: Specific provider to use

        Returns:
            Math agent configured for the original tutorial
        """
        return self.create_agent(AgentType.MATH, provider_name=provider_name)

    def print_status(self):
        """Print current factory status and available configurations."""
        print("🏭 Agent Factory Status")
        print("=" * 50)

        # Show current settings
        print(f"Default Provider: {self.settings.default_provider}")
        print(f"Default Model: {self.settings.default_model}")
        print(f"Tutorial Mode: {self.settings.tutorial_mode}")
        print()

        # Show available configurations
        configurations = self.get_available_configurations()
        print("📋 Available Configurations:")
        for provider, models in configurations.items():
            print(f"  {provider.upper()}:")
            for model in models[:3]:  # Show first 3 models
                print(f"    - {model}")
            if len(models) > 3:
                print(f"    - ... and {len(models) - 3} more")
        print()

        # Show health status
        from models.health import print_health_status
        print_health_status()


# Convenience functions for easy agent creation
def create_agent(
    agent_type: str,
    provider_name: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs
) -> Agent[None, str]:
    """Convenience function to create an agent.

    Args:
        agent_type: Type of agent ("basic", "math", "demo", "tool")
        provider_name: Specific provider to use
        model_name: Specific model to use
        **kwargs: Additional arguments

    Returns:
        Configured PydanticAI agent
    """
    factory = AgentFactory()
    agent_type_enum = AgentType(agent_type.lower())
    return factory.create_agent(agent_type_enum, provider_name, model_name, **kwargs)


def create_tutorial_agent(step: str) -> Agent[None, str]:
    """Create an agent for a specific tutorial step.

    Args:
        step: Tutorial step ("basic", "math", "demo", "tools")

    Returns:
        Configured agent for the tutorial step
    """
    factory = AgentFactory()
    return factory.create_tutorial_agent(step)


def create_best_available_agent(agent_type: str = "math") -> Agent[None, str]:
    """Create an agent using the best available provider.

    Args:
        agent_type: Type of agent to create

    Returns:
        Configured agent using best available provider
    """
    factory = AgentFactory()
    agent_type_enum = AgentType(agent_type.lower())
    return factory.create_agent(agent_type_enum)


def get_recommended_setup() -> Dict[str, Any]:
    """Get recommended setup for the tutorial based on available providers.

    Returns:
        Dictionary with recommended configuration
    """
    factory = AgentFactory()
    best_provider = get_best_available_provider()

    if not best_provider:
        return {
            "status": "no_providers",
            "message": "No providers available. Please configure API keys or start Ollama.",
            "setup_instructions": {
                "option_1": "Set OPENAI_API_KEY environment variable",
                "option_2": "Set ANTHROPIC_API_KEY environment variable",
                "option_3": "Install and start Ollama: 'ollama serve'"
            }
        }

    provider = get_provider_config(best_provider)
    return {
        "status": "ready",
        "recommended_provider": best_provider,
        "recommended_model": provider.default_model,
        "provider_description": provider.description,
        "full_model_id": provider.get_model_id(),
        "tutorial_steps": ["basic", "math", "demo", "tools"]
    }


def demo_factory():
    """Demonstrate the agent factory capabilities."""
    print("🎯 Agent Factory Demo")
    print("=" * 40)

    factory = AgentFactory()

    # Show status
    factory.print_status()

    print("\n" + "=" * 40)

    # Get recommendations
    setup = get_recommended_setup()
    print("🎯 Recommended Setup:")
    if setup["status"] == "ready":
        print(f"Provider: {setup['recommended_provider']}")
        print(f"Model: {setup['recommended_model']}")
        print(f"Description: {setup['provider_description']}")
        print(f"Full ID: {setup['full_model_id']}")

        # Try creating each agent type
        print("\n🤖 Creating Sample Agents:")
        for agent_type in ["basic", "math", "demo", "tool"]:
            try:
                agent = factory.create_agent(AgentType(agent_type))
                print(f"✅ {agent_type.capitalize()} agent created successfully")
            except Exception as e:
                print(f"❌ Failed to create {agent_type} agent: {e}")

    else:
        print(f"❌ {setup['message']}")
        print("Setup instructions:")
        for option, instruction in setup["setup_instructions"].items():
            print(f"  {option}: {instruction}")


if __name__ == "__main__":
    # Demo the factory
    demo_factory()

    print("\n" + "=" * 40)

    # Try the original tutorial workflow
    print("🧮 Original Tutorial Recreation:")
    try:
        agent = create_tutorial_agent("math")
        result = agent.run_sync("What is 3 plus 12?")
        print(f"Question: What is 3 plus 12?")
        print(f"Answer: {result.output}")
    except Exception as e:
        print(f"❌ Error: {e}")