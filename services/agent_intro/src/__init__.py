"""
Agent Intro Module - PydanticAI Tutorial

A tutorial module demonstrating the four core components of AI agents:
1. Model - The LLM that powers the agent (PydanticAI with multiple providers)
2. Tools - Functions the agent can call to interact with the world
3. Memory - Conversation history and context managed by PydanticAI
4. Routing - Logic to decide what to do next (automatic tool calling)

This module supports multiple model providers:
- OpenAI (recommended for beginners)
- Anthropic (excellent reasoning)
- HuggingFace (open source models)
- Groq (fast inference)
- Ollama (local/private models)
"""

from factory import (
    AgentFactory,
    create_agent,
    create_tutorial_agent,
    create_best_available_agent,
    get_recommended_setup
)

from models import (
    get_available_providers,
    get_provider_config,
    load_model_settings,
    check_all_providers
)

# Import main agent types for convenience
from agents.basic_agent import create_basic_agent
from agents.math_agent import create_math_agent
from agents.demo_agent import create_demo_agent
from agents.tool_agent import create_tool_agent

__version__ = "1.0.0"

__all__ = [
    # Factory functions
    "AgentFactory",
    "create_agent",
    "create_tutorial_agent",
    "create_best_available_agent",
    "get_recommended_setup",

    # Model management
    "get_available_providers",
    "get_provider_config",
    "load_model_settings",
    "check_all_providers",

    # Direct agent creators
    "create_basic_agent",
    "create_math_agent",
    "create_demo_agent",
    "create_tool_agent",
]