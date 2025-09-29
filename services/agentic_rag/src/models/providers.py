"""
Model Provider Configuration for Agentic RAG Service

This module defines available model providers for the RAG agents,
building on the patterns from the agent_intro service.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class ProviderType(str, Enum):
    """Enum for different provider types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    OLLAMA = "ollama"


@dataclass
class ModelProvider:
    """Configuration for a model provider."""
    name: str
    provider_type: ProviderType
    default_model: str
    available_models: List[str]
    requires_api_key: bool
    env_var: Optional[str] = None
    base_url: Optional[str] = None
    description: str = ""

    @property
    def is_available(self) -> bool:
        """Check if this provider is available."""
        if not self.requires_api_key:
            return True
        if self.env_var:
            return bool(os.getenv(self.env_var))
        return False

    def get_model_id(self, model_name: Optional[str] = None) -> str:
        """Get the full model identifier for PydanticAI."""
        model = model_name or self.default_model
        return f"{self.provider_type.value}:{model}"

    def get_api_key(self) -> Optional[str]:
        """Get the API key for this provider."""
        if self.env_var:
            return os.getenv(self.env_var)
        return None

    def get_available_models(self) -> List[str]:
        """Get the list of available models for this provider."""
        return self.available_models

    def get_default_model(self) -> str:
        """Get the default model for this provider."""
        return self.default_model


# Define supported providers for RAG
PROVIDERS: Dict[str, ModelProvider] = {
    "openai": ModelProvider(
        name="OpenAI",
        provider_type=ProviderType.OPENAI,
        default_model="gpt-4o-mini",
        available_models=["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
        requires_api_key=True,
        env_var="OPENAI_API_KEY",
        description="OpenAI's GPT models - excellent for RAG applications"
    ),
    "anthropic": ModelProvider(
        name="Anthropic",
        provider_type=ProviderType.ANTHROPIC,
        default_model="claude-3-haiku-20240307",
        available_models=["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
        requires_api_key=True,
        env_var="ANTHROPIC_API_KEY",
        description="Anthropic's Claude models - great reasoning for RAG"
    ),
    "groq": ModelProvider(
        name="Groq",
        provider_type=ProviderType.GROQ,
        default_model="llama-3.1-8b-instant",
        available_models=["llama-3.1-8b-instant", "llama-3.1-70b-versatile"],
        requires_api_key=True,
        env_var="GROQ_API_KEY",
        description="Groq's fast inference models"
    ),
    "ollama": ModelProvider(
        name="Ollama",
        provider_type=ProviderType.OLLAMA,
        default_model="llama3.2:latest",
        available_models=[
            "llama3.2:latest",
            "llama3.3:latest",
            "mistral",
            "codellama",
            "phi3",
            "gemma2"
        ],
        requires_api_key=False,
        base_url="http://localhost:11434",
        description="Local Ollama models - privacy-focused, no API key required"
    )
}


def get_available_providers() -> Dict[str, ModelProvider]:
    """Get all available model providers."""
    return {name: provider for name, provider in PROVIDERS.items() if provider.is_available}


def get_provider_config(provider_name: str) -> Optional[ModelProvider]:
    """Get configuration for a specific provider."""
    return PROVIDERS.get(provider_name)


def get_best_available_provider() -> Optional[str]:
    """Get the best available provider name."""
    # Try providers in order of preference
    for provider_name in ["openai", "anthropic", "groq", "ollama"]:
        if provider_name in PROVIDERS and PROVIDERS[provider_name].is_available:
            return provider_name
    # Fallback to test model for development
    return "test"