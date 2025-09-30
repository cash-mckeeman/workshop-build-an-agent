"""Tests for models.settings module."""

import os
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from src.models.settings import (
    ModelSettings,
    create_default_config,
    get_instructions_for_mode,
    get_provider_settings,
    load_model_settings,
    load_yaml_config,
)


@pytest.mark.unit
class TestModelSettings:
    """Test cases for ModelSettings dataclass."""

    def test_model_settings_defaults(self):
        """Test ModelSettings with default values."""
        settings = ModelSettings()

        assert settings.default_provider == "openai"
        assert settings.default_model == "gpt-4o-mini"
        assert settings.tutorial_mode == "beginner"
        assert settings.enable_streaming == False
        assert settings.max_retries == 3
        assert settings.timeout_seconds == 30
        assert settings.provider_settings == {}
        assert settings.custom_instructions == {}

    def test_model_settings_custom_values(self):
        """Test ModelSettings with custom values."""
        custom_provider_settings = {"openai": {"temperature": 0.8}}
        custom_instructions = {"test": "test instruction"}

        settings = ModelSettings(
            default_provider="anthropic",
            default_model="claude-3-haiku",
            tutorial_mode="advanced",
            enable_streaming=True,
            max_retries=5,
            timeout_seconds=60,
            provider_settings=custom_provider_settings,
            custom_instructions=custom_instructions,
        )

        assert settings.default_provider == "anthropic"
        assert settings.default_model == "claude-3-haiku"
        assert settings.tutorial_mode == "advanced"
        assert settings.enable_streaming == True
        assert settings.max_retries == 5
        assert settings.timeout_seconds == 60
        assert settings.provider_settings == custom_provider_settings
        assert settings.custom_instructions == custom_instructions


@pytest.mark.unit
class TestConfigFunctions:
    """Test cases for configuration functions."""

    def test_get_instructions_for_mode(self):
        """Test getting instructions for different modes."""
        beginner = get_instructions_for_mode("beginner")
        intermediate = get_instructions_for_mode("intermediate")
        advanced = get_instructions_for_mode("advanced")
        invalid = get_instructions_for_mode("invalid")

        assert "friendly" in beginner.lower()
        assert "patient" in beginner.lower()
        assert "detailed" in intermediate.lower()
        assert "expert" in advanced.lower()
        assert "production" in advanced.lower()

        # Invalid mode should return beginner instructions
        assert invalid == beginner

    def test_get_provider_settings(self):
        """Test getting provider-specific settings."""
        settings = ModelSettings()
        settings.provider_settings = {
            "openai": {"temperature": 0.7, "max_tokens": 1000},
            "anthropic": {"temperature": 0.5},
        }

        openai_settings = get_provider_settings("openai", settings)
        anthropic_settings = get_provider_settings("anthropic", settings)
        missing_settings = get_provider_settings("missing", settings)

        assert openai_settings == {"temperature": 0.7, "max_tokens": 1000}
        assert anthropic_settings == {"temperature": 0.5}
        assert missing_settings == {}

    def test_create_default_config(self):
        """Test creating default configuration."""
        config = create_default_config()

        assert "tutorial" in config
        assert "providers" in config
        assert "instructions" in config

        # Check tutorial section
        tutorial = config["tutorial"]
        assert tutorial["default_provider"] == "openai"
        assert tutorial["default_model"] == "gpt-4o-mini"
        assert tutorial["mode"] == "beginner"

        # Check providers section has expected providers
        providers = config["providers"]
        assert "openai" in providers
        assert "anthropic" in providers
        assert "ollama" in providers

        # Check instructions section
        instructions = config["instructions"]
        assert "beginner" in instructions
        assert "math_tutor" in instructions

    @patch("builtins.open", mock_open(read_data='{"test": "data"}'))
    @patch("yaml.safe_load")
    def test_load_yaml_config_success(self, mock_yaml_load):
        """Test successful YAML config loading."""
        mock_yaml_load.return_value = {"test": "data"}

        with patch("src.models.settings.get_config_path") as mock_path:
            mock_path.return_value = Path("/fake/path")
            with patch.object(Path, "exists", return_value=True):
                result = load_yaml_config("test.yaml")

        assert result == {"test": "data"}

    @patch("builtins.open", side_effect=Exception("File error"))
    def test_load_yaml_config_error(self, mock_open):
        """Test YAML config loading with file error."""
        with patch("src.models.settings.get_config_path") as mock_path:
            mock_path.return_value = Path("/fake/path")
            with patch.object(Path, "exists", return_value=True):
                result = load_yaml_config("test.yaml")

        assert result == {}

    def test_load_yaml_config_missing_file(self):
        """Test YAML config loading with missing file."""
        with patch("src.models.settings.get_config_path") as mock_path:
            mock_path.return_value = Path("/fake/path")
            with patch.object(Path, "exists", return_value=False):
                result = load_yaml_config("missing.yaml")

        assert result == {}


@pytest.mark.unit
class TestEnvironmentVariables:
    """Test cases for environment variable handling."""

    @patch.dict(
        os.environ,
        {
            "AGENT_DEFAULT_PROVIDER": "anthropic",
            "AGENT_DEFAULT_MODEL": "claude-3-haiku",
            "AGENT_TUTORIAL_MODE": "advanced",
        },
    )
    @patch("src.models.settings.load_yaml_config")
    def test_load_model_settings_with_env_vars(self, mock_load_yaml):
        """Test loading settings with environment variable overrides."""
        mock_load_yaml.return_value = {
            "tutorial": {
                "default_provider": "openai",
                "default_model": "gpt-4o-mini",
                "mode": "beginner",
            }
        }

        settings = load_model_settings()

        # Environment variables should override config
        assert settings.default_provider == "anthropic"
        assert settings.default_model == "claude-3-haiku"
        assert settings.tutorial_mode == "advanced"

    @patch("src.models.settings.load_yaml_config")
    def test_load_model_settings_without_env_vars(self, mock_load_yaml):
        """Test loading settings without environment variables."""
        mock_load_yaml.return_value = {
            "tutorial": {
                "default_provider": "groq",
                "default_model": "llama3-8b",
                "mode": "intermediate",
                "enable_streaming": True,
                "max_retries": 5,
                "timeout_seconds": 45,
            },
            "providers": {"groq": {"settings": {"temperature": 0.9}}},
            "instructions": {"custom": "Custom instruction"},
        }

        settings = load_model_settings()

        assert settings.default_provider == "groq"
        assert settings.default_model == "llama3-8b"
        assert settings.tutorial_mode == "intermediate"
        assert settings.enable_streaming == True
        assert settings.max_retries == 5
        assert settings.timeout_seconds == 45
        assert "groq" in settings.provider_settings
        assert settings.custom_instructions["custom"] == "Custom instruction"

    @patch("src.models.settings.load_yaml_config")
    def test_load_model_settings_empty_config(self, mock_load_yaml):
        """Test loading settings with empty config."""
        mock_load_yaml.return_value = {}

        settings = load_model_settings()

        # Should use defaults
        assert settings.default_provider == "openai"
        assert settings.default_model == "gpt-4o-mini"
        assert settings.tutorial_mode == "beginner"
