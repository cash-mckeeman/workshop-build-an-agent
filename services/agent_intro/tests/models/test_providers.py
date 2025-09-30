"""Tests for models.providers module."""

import os
from unittest.mock import patch

import pytest

from src.models.providers import (
    PROVIDERS,
    ModelProvider,
    ProviderType,
    get_available_providers,
    get_model_id,
    get_provider_config,
    get_recommended_provider,
    get_tutorial_recommendations,
    list_models_for_provider,
    validate_model_combination,
)


@pytest.mark.unit
class TestProviderType:
    """Test cases for ProviderType enum."""

    def test_provider_type_values(self):
        """Test that ProviderType has expected values."""
        assert ProviderType.OPENAI == "openai"
        assert ProviderType.ANTHROPIC == "anthropic"
        assert ProviderType.HUGGINGFACE == "huggingface"
        assert ProviderType.GROQ == "groq"
        assert ProviderType.OLLAMA == "ollama"


@pytest.mark.unit
class TestModelProvider:
    """Test cases for ModelProvider dataclass."""

    def test_model_provider_creation(self):
        """Test creating a ModelProvider instance."""
        provider = ModelProvider(
            name="Test Provider",
            provider_type=ProviderType.OPENAI,
            default_model="test-model",
            available_models=["test-model", "other-model"],
            requires_api_key=True,
            env_var="TEST_API_KEY",
            description="Test provider",
        )

        assert provider.name == "Test Provider"
        assert provider.provider_type == ProviderType.OPENAI
        assert provider.default_model == "test-model"
        assert provider.available_models == ["test-model", "other-model"]
        assert provider.requires_api_key == True
        assert provider.env_var == "TEST_API_KEY"
        assert provider.description == "Test provider"
        assert provider.recommended_for == []  # Should initialize empty list

    def test_model_provider_post_init(self):
        """Test ModelProvider post_init behavior."""
        provider = ModelProvider(
            name="Test",
            provider_type=ProviderType.OPENAI,
            default_model="test",
            available_models=["test"],
            requires_api_key=False,
        )

        assert provider.recommended_for == []

        provider_with_recommendations = ModelProvider(
            name="Test",
            provider_type=ProviderType.OPENAI,
            default_model="test",
            available_models=["test"],
            requires_api_key=False,
            recommended_for=["testing"],
        )

        assert provider_with_recommendations.recommended_for == ["testing"]

    @patch.dict(os.environ, {"TEST_KEY": "test_value"})
    def test_is_available_with_api_key(self):
        """Test is_available when API key is required and present."""
        provider = ModelProvider(
            name="Test",
            provider_type=ProviderType.OPENAI,
            default_model="test",
            available_models=["test"],
            requires_api_key=True,
            env_var="TEST_KEY",
        )

        assert provider.is_available == True

    def test_is_available_without_required_api_key(self):
        """Test is_available when API key is required but missing."""
        provider = ModelProvider(
            name="Test",
            provider_type=ProviderType.OPENAI,
            default_model="test",
            available_models=["test"],
            requires_api_key=True,
            env_var="MISSING_KEY",
        )

        assert provider.is_available == False

    def test_is_available_no_api_key_required(self):
        """Test is_available when no API key is required."""
        provider = ModelProvider(
            name="Test",
            provider_type=ProviderType.OLLAMA,
            default_model="test",
            available_models=["test"],
            requires_api_key=False,
        )

        assert provider.is_available == True

    def test_get_model_id(self):
        """Test getting model ID for PydanticAI."""
        provider = ModelProvider(
            name="Test",
            provider_type=ProviderType.OPENAI,
            default_model="default-model",
            available_models=["default-model", "other-model"],
            requires_api_key=False,
        )

        # Test with default model
        assert provider.get_model_id() == "openai:default-model"

        # Test with custom model
        assert provider.get_model_id("other-model") == "openai:other-model"

    @patch.dict(os.environ, {"TEST_KEY": "secret_key_value"})
    def test_get_api_key(self):
        """Test getting API key from environment."""
        provider = ModelProvider(
            name="Test",
            provider_type=ProviderType.OPENAI,
            default_model="test",
            available_models=["test"],
            requires_api_key=True,
            env_var="TEST_KEY",
        )

        assert provider.get_api_key() == "secret_key_value"

    def test_get_api_key_no_env_var(self):
        """Test getting API key when no env_var specified."""
        provider = ModelProvider(
            name="Test",
            provider_type=ProviderType.OLLAMA,
            default_model="test",
            available_models=["test"],
            requires_api_key=False,
        )

        assert provider.get_api_key() is None


@pytest.mark.unit
class TestProviderFunctions:
    """Test cases for provider utility functions."""

    def test_providers_exist(self):
        """Test that expected providers are defined."""
        assert "openai" in PROVIDERS
        assert "anthropic" in PROVIDERS
        assert "huggingface" in PROVIDERS
        assert "groq" in PROVIDERS
        assert "ollama" in PROVIDERS

    def test_provider_configurations(self):
        """Test that provider configurations are valid."""
        for _name, provider in PROVIDERS.items():
            assert isinstance(provider, ModelProvider)
            assert provider.name
            assert provider.default_model
            assert provider.available_models
            assert provider.default_model in provider.available_models

    def test_get_provider_config(self):
        """Test getting provider configuration."""
        openai_config = get_provider_config("openai")
        assert openai_config is not None
        assert openai_config.name == "OpenAI"

        missing_config = get_provider_config("nonexistent")
        assert missing_config is None

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_get_available_providers(self):
        """Test getting available providers."""
        available = get_available_providers()

        # OpenAI should be available with API key
        assert "openai" in available
        # Ollama should be available (no API key required)
        assert "ollama" in available

    def test_list_models_for_provider(self):
        """Test listing models for a provider."""
        openai_models = list_models_for_provider("openai")
        assert "gpt-4o-mini" in openai_models

        missing_models = list_models_for_provider("nonexistent")
        assert missing_models == []

    def test_get_model_id(self):
        """Test getting model ID."""
        model_id = get_model_id("openai", "gpt-4o")
        assert model_id == "openai:gpt-4o"

        default_id = get_model_id("openai")
        assert default_id == "openai:gpt-4o-mini"

        missing_id = get_model_id("nonexistent")
        assert missing_id is None

    def test_validate_model_combination(self):
        """Test validating model combinations."""
        # Valid combinations
        assert validate_model_combination("openai", "gpt-4o-mini") == True
        assert (
            validate_model_combination("anthropic", "claude-3-haiku-20240307") == True
        )

        # Invalid combinations
        assert validate_model_combination("openai", "claude-3-haiku") == False
        assert validate_model_combination("nonexistent", "any-model") == False

    @patch("src.models.providers.get_available_providers")
    def test_get_recommended_provider(self, mock_available):
        """Test getting recommended provider for use case."""
        mock_providers = {"openai": PROVIDERS["openai"], "groq": PROVIDERS["groq"]}
        mock_available.return_value = mock_providers

        # Test specific recommendation
        fast_provider = get_recommended_provider("speed")
        assert fast_provider is not None
        assert fast_provider.name == "Groq"

        # Test fallback
        general_provider = get_recommended_provider("general")
        assert general_provider is not None  # Should return first available

        # Test no providers available
        mock_available.return_value = {}
        no_provider = get_recommended_provider("anything")
        assert no_provider is None

    @patch("src.models.providers.get_available_providers")
    def test_get_tutorial_recommendations(self, mock_available):
        """Test getting tutorial recommendations."""
        mock_available.return_value = {
            "openai": PROVIDERS["openai"],
            "ollama": PROVIDERS["ollama"],
            "huggingface": PROVIDERS["huggingface"],
            "groq": PROVIDERS["groq"],
        }

        recommendations = get_tutorial_recommendations()

        assert "beginner" in recommendations
        assert "local" in recommendations
        assert "free" in recommendations
        assert "fast" in recommendations

        # Test specific recommendations
        assert recommendations["beginner"] == "openai:gpt-4o-mini"
        assert recommendations["local"] == "ollama:llama3.2"
        assert recommendations["fast"] == "groq:llama3-8b-8192"
