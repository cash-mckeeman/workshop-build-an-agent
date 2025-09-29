"""
Model Settings and Configuration Management

This module handles loading and managing settings for different model providers,
including YAML configuration files and environment variable overrides.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path


@dataclass
class ModelSettings:
    """Settings for model configuration."""
    # Provider settings
    default_provider: str = "openai"
    default_model: str = "gpt-4o-mini"

    # Tutorial-specific settings
    tutorial_mode: str = "beginner"  # beginner, intermediate, advanced
    enable_streaming: bool = False
    max_retries: int = 3
    timeout_seconds: int = 30

    # Provider-specific overrides
    provider_settings: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Model-specific instructions
    custom_instructions: Dict[str, str] = field(default_factory=dict)


def get_config_path() -> Path:
    """Get the path to the configuration directory."""
    # Look for config in several places
    possible_paths = [
        Path.cwd() / "config",
        Path(__file__).parent.parent.parent.parent / "config",
        Path.home() / ".agent_intro",
    ]

    for path in possible_paths:
        if path.exists():
            return path

    # Create default config directory
    default_path = Path.cwd() / "config"
    default_path.mkdir(exist_ok=True)
    return default_path


def load_yaml_config(config_file: str = "models.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = get_config_path() / config_file

    if not config_path.exists():
        return {}

    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Warning: Could not load {config_file}: {e}")
        return {}


def load_model_settings() -> ModelSettings:
    """Load model settings from configuration files and environment variables."""
    # Load base configuration
    config = load_yaml_config("models.yaml")

    # Extract tutorial settings
    tutorial_config = config.get("tutorial", {})

    settings = ModelSettings(
        default_provider=tutorial_config.get("default_provider", "openai"),
        default_model=tutorial_config.get("default_model", "gpt-4o-mini"),
        tutorial_mode=tutorial_config.get("mode", "beginner"),
        enable_streaming=tutorial_config.get("enable_streaming", False),
        max_retries=tutorial_config.get("max_retries", 3),
        timeout_seconds=tutorial_config.get("timeout_seconds", 30),
    )

    # Load provider-specific settings
    providers_config = config.get("providers", {})
    for provider_name, provider_config in providers_config.items():
        if "settings" in provider_config:
            settings.provider_settings[provider_name] = provider_config["settings"]

    # Load custom instructions
    instructions_config = config.get("instructions", {})
    settings.custom_instructions = instructions_config

    # Environment variable overrides
    if env_provider := os.getenv("AGENT_DEFAULT_PROVIDER"):
        settings.default_provider = env_provider

    if env_model := os.getenv("AGENT_DEFAULT_MODEL"):
        settings.default_model = env_model

    if env_mode := os.getenv("AGENT_TUTORIAL_MODE"):
        settings.tutorial_mode = env_mode

    return settings


def get_instructions_for_mode(mode: str = "beginner") -> str:
    """Get appropriate instructions based on tutorial mode."""
    instructions = {
        "beginner": (
            "You are a friendly AI assistant helping users learn about agents. "
            "Be very clear, patient, and educational. Explain concepts simply "
            "and provide practical examples. Always encourage learning."
        ),
        "intermediate": (
            "You are an AI assistant helping users understand advanced agent concepts. "
            "Provide detailed explanations with technical depth. Show practical "
            "applications and real-world use cases."
        ),
        "advanced": (
            "You are an expert AI assistant for advanced agent development. "
            "Provide comprehensive technical guidance, best practices, and "
            "production considerations. Focus on optimization and scalability."
        )
    }

    return instructions.get(mode, instructions["beginner"])


def get_provider_settings(provider_name: str, settings: ModelSettings) -> Dict[str, Any]:
    """Get settings for a specific provider."""
    return settings.provider_settings.get(provider_name, {})


def create_default_config() -> Dict[str, Any]:
    """Create a default configuration structure."""
    return {
        "tutorial": {
            "default_provider": "openai",
            "default_model": "gpt-4o-mini",
            "mode": "beginner",
            "enable_streaming": False,
            "max_retries": 3,
            "timeout_seconds": 30,
            "recommended_providers": {
                "beginner": "openai",
                "local": "ollama",
                "free": "huggingface"
            },
            "demo_models": {
                "fast": "openai:gpt-4o-mini",
                "capable": "anthropic:claude-3-haiku-20240307",
                "local": "ollama:llama3.2"
            }
        },
        "providers": {
            "openai": {
                "default_model": "gpt-4o-mini",
                "available_models": [
                    "gpt-4o",
                    "gpt-4o-mini",
                    "gpt-3.5-turbo"
                ],
                "env_var": "OPENAI_API_KEY",
                "description": "OpenAI's GPT models",
                "settings": {
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
            },
            "anthropic": {
                "default_model": "claude-3-haiku-20240307",
                "available_models": [
                    "claude-3-5-sonnet-20241022",
                    "claude-3-haiku-20240307",
                    "claude-3-sonnet-20240229"
                ],
                "env_var": "ANTHROPIC_API_KEY",
                "description": "Anthropic's Claude models",
                "settings": {
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
            },
            "huggingface": {
                "default_model": "microsoft/DialoGPT-medium",
                "available_models": [
                    "microsoft/DialoGPT-medium",
                    "meta-llama/Llama-2-7b-chat-hf",
                    "mistralai/Mistral-7B-Instruct-v0.1"
                ],
                "env_var": "HUGGINGFACE_API_KEY",
                "description": "HuggingFace models via Inference API"
            },
            "ollama": {
                "default_model": "llama3.2",
                "available_models": [
                    "llama3.2",
                    "mistral",
                    "codellama",
                    "phi3"
                ],
                "connection": {
                    "host": "localhost",
                    "port": 11434
                },
                "description": "Local Ollama models"
            },
            "groq": {
                "default_model": "llama3-8b-8192",
                "available_models": [
                    "llama3-8b-8192",
                    "mixtral-8x7b-32768",
                    "gemma-7b-it"
                ],
                "env_var": "GROQ_API_KEY",
                "description": "Groq's fast inference models"
            }
        },
        "instructions": {
            "beginner": "You are a friendly AI assistant helping users learn about agents.",
            "math_tutor": "You are a helpful math assistant. Use tools to perform calculations.",
            "tool_demo": "You are an assistant with access to various tools. Use them appropriately."
        }
    }


def save_default_config(config_file: str = "models.yaml"):
    """Save a default configuration file."""
    config_path = get_config_path() / config_file
    config = create_default_config()

    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        print(f"✅ Created default configuration: {config_path}")
    except Exception as e:
        print(f"❌ Error creating config file: {e}")


def print_current_settings():
    """Print current settings for debugging."""
    settings = load_model_settings()

    print("⚙️ Current Model Settings:")
    print("=" * 40)
    print(f"Default Provider: {settings.default_provider}")
    print(f"Default Model: {settings.default_model}")
    print(f"Tutorial Mode: {settings.tutorial_mode}")
    print(f"Max Retries: {settings.max_retries}")
    print(f"Timeout: {settings.timeout_seconds}s")

    if settings.provider_settings:
        print("\n🔧 Provider Settings:")
        for provider, config in settings.provider_settings.items():
            print(f"  {provider}: {config}")

    if settings.custom_instructions:
        print("\n📝 Custom Instructions:")
        for key, instruction in settings.custom_instructions.items():
            print(f"  {key}: {instruction[:50]}...")


if __name__ == "__main__":
    # Demo the settings system
    config_path = get_config_path()
    print(f"📂 Config directory: {config_path}")

    if not (config_path / "models.yaml").exists():
        print("No configuration found. Creating default...")
        save_default_config()

    print_current_settings()