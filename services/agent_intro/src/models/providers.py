"""
Model Provider Configuration for PydanticAI

This module defines the available model providers and their configurations,
making it easy to switch between different models for the tutorial.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum


class ProviderType(str, Enum):
    """Enum for different provider types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
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
    recommended_for: List[str] = None

    def __post_init__(self):
        if self.recommended_for is None:
            self.recommended_for = []

    @property
    def is_available(self) -> bool:
        """Check if this provider is available (has required env vars, etc.)."""
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


# Define all supported providers
PROVIDERS: Dict[str, ModelProvider] = {
    "openai": ModelProvider(
        name="OpenAI",
        provider_type=ProviderType.OPENAI,
        default_model="gpt-4o-mini",
        available_models=[
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo"
        ],
        requires_api_key=True,
        env_var="OPENAI_API_KEY",
        description="OpenAI's GPT models - reliable and well-documented",
        recommended_for=["beginners", "production", "reliability"]
    ),

    "anthropic": ModelProvider(
        name="Anthropic",
        provider_type=ProviderType.ANTHROPIC,
        default_model="claude-3-haiku-20240307",
        available_models=[
            "claude-3-5-sonnet-20241022",
            "claude-3-haiku-20240307",
            "claude-3-sonnet-20240229"
        ],
        requires_api_key=True,
        env_var="ANTHROPIC_API_KEY",
        description="Anthropic's Claude models - excellent for reasoning and safety",
        recommended_for=["reasoning", "safety", "complex-tasks"]
    ),

    "huggingface": ModelProvider(
        name="HuggingFace",
        provider_type=ProviderType.HUGGINGFACE,
        default_model="Qwen/Qwen3-14B",
        available_models=[
            "Qwen/Qwen3-14B",
            "microsoft/DialoGPT-medium",
            "meta-llama/Llama-2-7b-chat-hf",
            "mistralai/Mistral-7B-Instruct-v0.1",
            "HuggingFaceH4/zephyr-7b-beta"
        ],
        requires_api_key=True,
        env_var="HUGGINGFACE_API_KEY",
        description="HuggingFace models via Inference API - diverse open-source options",
        recommended_for=["open-source", "research", "cost-effective"]
    ),

    "groq": ModelProvider(
        name="Groq",
        provider_type=ProviderType.GROQ,
        default_model="llama3-8b-8192",
        available_models=[
            "llama3-8b-8192",
            "mixtral-8x7b-32768",
            "gemma-7b-it"
        ],
        requires_api_key=True,
        env_var="GROQ_API_KEY",
        description="Groq's fast inference models - optimized for speed",
        recommended_for=["speed", "low-latency", "real-time"]
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
        description="Local Ollama models - privacy-focused, no API keys needed",
        recommended_for=["privacy", "local", "offline", "development"]
    )
}


def get_available_providers() -> Dict[str, ModelProvider]:
    """Get all available providers (those with proper configuration)."""
    return {
        name: provider
        for name, provider in PROVIDERS.items()
        if provider.is_available
    }


def get_provider_config(provider_name: str) -> Optional[ModelProvider]:
    """Get configuration for a specific provider."""
    return PROVIDERS.get(provider_name)


def get_recommended_provider(use_case: str) -> Optional[ModelProvider]:
    """Get a recommended provider for a specific use case."""
    available = get_available_providers()

    for provider in available.values():
        if use_case in provider.recommended_for:
            return provider

    # Fallback to any available provider
    if available:
        return next(iter(available.values()))

    return None


def list_models_for_provider(provider_name: str) -> List[str]:
    """List all available models for a provider."""
    provider = get_provider_config(provider_name)
    if provider:
        return provider.available_models
    return []


def get_model_id(provider_name: str, model_name: Optional[str] = None) -> Optional[str]:
    """Get the full PydanticAI model identifier."""
    provider = get_provider_config(provider_name)
    if provider:
        return provider.get_model_id(model_name)
    return None


def validate_model_combination(provider_name: str, model_name: str) -> bool:
    """Validate that a model is available for a provider."""
    provider = get_provider_config(provider_name)
    if not provider:
        return False

    return model_name in provider.available_models


def get_tutorial_recommendations() -> Dict[str, str]:
    """Get model recommendations for different tutorial scenarios."""
    available = get_available_providers()

    recommendations = {}

    # Beginner-friendly
    if "openai" in available:
        recommendations["beginner"] = "openai:gpt-4o-mini"
    elif "anthropic" in available:
        recommendations["beginner"] = "anthropic:claude-3-haiku-20240307"

    # Local development
    if "ollama" in available:
        recommendations["local"] = "ollama:llama3.2"

    # Free/cost-effective
    if "huggingface" in available:
        recommendations["free"] = "huggingface:microsoft/DialoGPT-medium"

    # Fast inference
    if "groq" in available:
        recommendations["fast"] = "groq:llama3-8b-8192"

    return recommendations


def print_provider_status():
    """Print the status of all providers for debugging."""
    print("🔍 Provider Status:")
    print("=" * 50)

    for name, provider in PROVIDERS.items():
        status = "✅ Available" if provider.is_available else "❌ Not Available"
        api_key_status = ""

        if provider.requires_api_key and provider.env_var:
            key = provider.get_api_key()
            if key:
                api_key_status = f" (API key: {key[:8]}...)"
            else:
                api_key_status = f" (Missing {provider.env_var})"

        print(f"{name}: {status}{api_key_status}")
        print(f"  Default model: {provider.default_model}")
        print(f"  Description: {provider.description}")
        print()


if __name__ == "__main__":
    # Demo the provider configuration
    print_provider_status()

    available = get_available_providers()
    print(f"\n📋 Available providers: {list(available.keys())}")

    recommendations = get_tutorial_recommendations()
    print(f"\n🎯 Tutorial recommendations: {recommendations}")
