"""
Tests for the health checking module.
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import dataclass

from models.health import (
    HealthStatus,
    HealthCheckResult,
    HealthChecker,
    check_provider_health,
    check_all_providers,
    get_healthy_providers,
    get_best_available_provider,
    print_health_status,
)
from models.providers import ModelProvider, ProviderType


@pytest.fixture
def mock_provider():
    """Create a mock provider for testing."""
    provider = Mock(spec=ModelProvider)
    provider.name = "test_provider"
    provider.provider_type = ProviderType.OPENAI
    provider.is_available = True
    provider.requires_api_key = True
    provider.env_var = "TEST_API_KEY"
    provider.default_model = "test-model"
    provider.get_api_key.return_value = "test_api_key_1234567890"
    provider.base_url = None
    return provider


@pytest.fixture
def mock_ollama_provider():
    """Create a mock Ollama provider for testing."""
    provider = Mock(spec=ModelProvider)
    provider.name = "ollama"
    provider.provider_type = ProviderType.OLLAMA
    provider.is_available = True
    provider.requires_api_key = False
    provider.env_var = None
    provider.default_model = "llama2"
    provider.get_api_key.return_value = None
    provider.base_url = "http://localhost:11434"
    return provider


@pytest.fixture
def health_checker():
    """Create a HealthChecker instance."""
    return HealthChecker()


class TestHealthStatus:
    """Test HealthStatus enum."""

    def test_health_status_values(self):
        """Test health status enum values."""
        assert HealthStatus.HEALTHY == "healthy"
        assert HealthStatus.UNHEALTHY == "unhealthy"
        assert HealthStatus.UNKNOWN == "unknown"


class TestHealthCheckResult:
    """Test HealthCheckResult dataclass."""

    def test_health_check_result_creation(self):
        """Test creating HealthCheckResult."""
        result = HealthCheckResult(
            provider_name="test_provider",
            status=HealthStatus.HEALTHY,
            message="All good",
            response_time_ms=100,
            model_tested="test-model",
            error=None
        )

        assert result.provider_name == "test_provider"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "All good"
        assert result.response_time_ms == 100
        assert result.model_tested == "test-model"
        assert result.error is None

    def test_health_check_result_minimal(self):
        """Test creating HealthCheckResult with minimal parameters."""
        result = HealthCheckResult(
            provider_name="test",
            status=HealthStatus.UNHEALTHY,
            message="Failed"
        )

        assert result.provider_name == "test"
        assert result.status == HealthStatus.UNHEALTHY
        assert result.message == "Failed"
        assert result.response_time_ms is None
        assert result.model_tested is None
        assert result.error is None


class TestHealthChecker:
    """Test HealthChecker class."""

    def test_health_checker_init(self):
        """Test HealthChecker initialization."""
        checker = HealthChecker()
        assert checker.timeout_seconds == 30

    @pytest.mark.asyncio
    async def test_check_provider_health_unavailable(self, health_checker, mock_provider):
        """Test health check when provider is not available."""
        mock_provider.is_available = False

        result = await health_checker.check_provider_health(mock_provider)

        assert result.provider_name == "test_provider"
        assert result.status == HealthStatus.UNHEALTHY
        assert "Provider not available" in result.message
        assert result.error == "Configuration issue"

    @pytest.mark.asyncio
    async def test_check_provider_health_missing_api_key(self, health_checker, mock_provider):
        """Test health check when API key is missing."""
        mock_provider.get_api_key.return_value = None

        result = await health_checker.check_provider_health(mock_provider)

        assert result.provider_name == "test_provider"
        assert result.status == HealthStatus.UNHEALTHY
        assert "Missing API key" in result.message
        assert result.error == "Authentication issue"

    @pytest.mark.asyncio
    async def test_check_provider_health_invalid_api_key(self, health_checker, mock_provider):
        """Test health check when API key is too short."""
        mock_provider.get_api_key.return_value = "short"

        result = await health_checker.check_provider_health(mock_provider)

        assert result.provider_name == "test_provider"
        assert result.status == HealthStatus.UNHEALTHY
        assert "API key appears to be invalid" in result.message
        assert result.error == "Authentication issue"

    @pytest.mark.asyncio
    async def test_check_provider_health_success(self, health_checker, mock_provider):
        """Test successful health check."""
        result = await health_checker.check_provider_health(mock_provider)

        assert result.provider_name == "test_provider"
        assert result.status == HealthStatus.HEALTHY
        assert "Provider configuration appears valid" in result.message
        assert result.response_time_ms is not None
        assert result.response_time_ms >= 0
        assert result.model_tested == "test-model"

    @pytest.mark.asyncio
    async def test_check_provider_health_exception(self, health_checker, mock_provider):
        """Test health check when exception occurs."""
        mock_provider.get_api_key.side_effect = Exception("Test error")

        result = await health_checker.check_provider_health(mock_provider)

        assert result.provider_name == "test_provider"
        assert result.status == HealthStatus.UNHEALTHY
        assert "Health check failed" in result.message
        assert result.error == "Test error"

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_check_ollama_health_success(self, mock_client_class, health_checker, mock_ollama_provider):
        """Test successful Ollama health check."""
        # Mock httpx response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama2"},
                {"name": "mistral"}
            ]
        }

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value.__aenter__.return_value = mock_client

        result = await health_checker._check_ollama_health(mock_ollama_provider)

        assert result.status == HealthStatus.HEALTHY
        assert "Ollama server healthy" in result.message
        assert result.model_tested == "llama2"

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_check_ollama_health_model_not_found(self, mock_client_class, health_checker, mock_ollama_provider):
        """Test Ollama health check when model is not available."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "mistral"}
            ]
        }

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value.__aenter__.return_value = mock_client

        result = await health_checker._check_ollama_health(mock_ollama_provider)

        assert result.status == HealthStatus.UNHEALTHY
        assert "Model 'llama2' not found" in result.message
        assert result.error == "Model not available"

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_check_ollama_health_server_error(self, mock_client_class, health_checker, mock_ollama_provider):
        """Test Ollama health check when server returns error."""
        mock_response = Mock()
        mock_response.status_code = 500

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value.__aenter__.return_value = mock_client

        result = await health_checker._check_ollama_health(mock_ollama_provider)

        assert result.status == HealthStatus.UNHEALTHY
        assert "Ollama server returned status 500" in result.message
        assert result.error == "Server error"

    @pytest.mark.asyncio
    async def test_check_ollama_health_import_error(self, health_checker, mock_ollama_provider):
        """Test Ollama health check when httpx is not available."""
        # Mock the import to raise ImportError when httpx is imported
        def mock_import(name, *args, **kwargs):
            if name == 'httpx':
                raise ImportError("No module named 'httpx'")
            return __import__(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import):
            result = await health_checker._check_ollama_health(mock_ollama_provider)

        assert result.status == HealthStatus.UNHEALTHY
        assert "httpx not available" in result.message
        assert result.error == "Missing dependency"

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_check_ollama_health_connection_error(self, mock_client_class, health_checker, mock_ollama_provider):
        """Test Ollama health check when connection fails."""
        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("Connection refused")
        mock_client_class.return_value.__aenter__.return_value = mock_client

        result = await health_checker._check_ollama_health(mock_ollama_provider)

        assert result.status == HealthStatus.UNHEALTHY
        assert "Cannot connect to Ollama server" in result.message
        assert result.error == "Connection refused"

    def test_get_availability_reason_missing_env_var(self, health_checker, mock_provider):
        """Test getting availability reason for missing environment variable."""
        mock_provider.env_var = "MISSING_API_KEY"

        with patch('os.getenv') as mock_getenv:
            mock_getenv.return_value = None
            reason = health_checker._get_availability_reason(mock_provider)

            assert "Missing environment variable: MISSING_API_KEY" in reason

    def test_get_availability_reason_unknown(self, health_checker, mock_provider):
        """Test getting availability reason when cause is unknown."""
        mock_provider.requires_api_key = False
        mock_provider.env_var = None

        reason = health_checker._get_availability_reason(mock_provider)

        assert reason == "Unknown configuration issue"

    @pytest.mark.asyncio
    async def test_check_all_providers(self, health_checker):
        """Test checking all providers."""
        with patch('models.health.PROVIDERS') as mock_providers:
            mock_provider1 = Mock()
            mock_provider1.name = "provider1"
            mock_provider2 = Mock()
            mock_provider2.name = "provider2"

            mock_providers.items.return_value = [
                ("provider1", mock_provider1),
                ("provider2", mock_provider2)
            ]

            # Mock the health check method
            async def mock_check(provider):
                return HealthCheckResult(
                    provider_name=provider.name,
                    status=HealthStatus.HEALTHY,
                    message="OK"
                )

            health_checker.check_provider_health = mock_check

            results = await health_checker.check_all_providers()

            assert len(results) == 2
            assert "provider1" in results
            assert "provider2" in results
            assert results["provider1"].status == HealthStatus.HEALTHY
            assert results["provider2"].status == HealthStatus.HEALTHY


class TestSyncFunctions:
    """Test synchronous convenience functions."""

    def test_check_provider_health_not_found(self):
        """Test sync health check for non-existent provider."""
        with patch('models.health.PROVIDERS', {}):
            result = check_provider_health("nonexistent")

            assert result.provider_name == "nonexistent"
            assert result.status == HealthStatus.UNHEALTHY
            assert "Provider 'nonexistent' not found" in result.message
            assert result.error == "Provider not found"

    def test_check_provider_health_success(self):
        """Test sync health check success."""
        mock_provider = Mock()
        mock_provider.name = "test"

        with patch('models.health.PROVIDERS', {"test": mock_provider}):
            with patch('models.health.HealthChecker') as mock_checker_class:
                mock_checker = Mock()
                mock_result = HealthCheckResult(
                    provider_name="test",
                    status=HealthStatus.HEALTHY,
                    message="OK"
                )

                # Mock the async method
                async def mock_check(provider):
                    return mock_result

                mock_checker.check_provider_health = mock_check
                mock_checker_class.return_value = mock_checker

                with patch('asyncio.get_event_loop') as mock_get_loop:
                    mock_loop = Mock()
                    mock_loop.run_until_complete.return_value = mock_result
                    mock_get_loop.return_value = mock_loop

                    result = check_provider_health("test")

                    assert result.provider_name == "test"
                    assert result.status == HealthStatus.HEALTHY

    def test_check_all_providers_sync(self):
        """Test sync check all providers."""
        with patch('models.health.HealthChecker') as mock_checker_class:
            mock_checker = Mock()
            mock_results = {
                "provider1": HealthCheckResult("provider1", HealthStatus.HEALTHY, "OK"),
                "provider2": HealthCheckResult("provider2", HealthStatus.UNHEALTHY, "Failed")
            }

            async def mock_check_all():
                return mock_results

            mock_checker.check_all_providers = mock_check_all
            mock_checker_class.return_value = mock_checker

            with patch('asyncio.get_event_loop') as mock_get_loop:
                mock_loop = Mock()
                mock_loop.run_until_complete.return_value = mock_results
                mock_get_loop.return_value = mock_loop

                results = check_all_providers()

                assert len(results) == 2
                assert results["provider1"].status == HealthStatus.HEALTHY
                assert results["provider2"].status == HealthStatus.UNHEALTHY

    def test_get_healthy_providers(self):
        """Test getting healthy providers."""
        mock_results = {
            "provider1": HealthCheckResult("provider1", HealthStatus.HEALTHY, "OK"),
            "provider2": HealthCheckResult("provider2", HealthStatus.UNHEALTHY, "Failed"),
            "provider3": HealthCheckResult("provider3", HealthStatus.HEALTHY, "OK")
        }

        with patch('models.health.check_all_providers', return_value=mock_results):
            healthy = get_healthy_providers()

            assert len(healthy) == 2
            assert "provider1" in healthy
            assert "provider3" in healthy
            assert "provider2" not in healthy

    def test_get_best_available_provider_with_preferences(self):
        """Test getting best available provider with preferences."""
        with patch('models.health.get_healthy_providers', return_value=["groq", "ollama"]):
            # Should return groq since it's preferred and available
            best = get_best_available_provider(["openai", "groq", "ollama"])
            assert best == "groq"

    def test_get_best_available_provider_default_preferences(self):
        """Test getting best available provider with default preferences."""
        with patch('models.health.get_healthy_providers', return_value=["groq", "ollama"]):
            # Should return groq since it comes before ollama in default preferences
            best = get_best_available_provider()
            assert best == "groq"

    def test_get_best_available_provider_fallback(self):
        """Test getting best available provider with fallback."""
        with patch('models.health.get_healthy_providers', return_value=["unknown_provider"]):
            # Should return the only available provider even if not in preferences
            best = get_best_available_provider(["openai", "anthropic"])
            assert best == "unknown_provider"

    def test_get_best_available_provider_none_available(self):
        """Test getting best available provider when none available."""
        with patch('models.health.get_healthy_providers', return_value=[]):
            best = get_best_available_provider()
            assert best is None

    def test_print_health_status(self, capsys):
        """Test printing health status."""
        mock_results = {
            "provider1": HealthCheckResult(
                provider_name="provider1",
                status=HealthStatus.HEALTHY,
                message="All good",
                response_time_ms=100,
                model_tested="test-model"
            ),
            "provider2": HealthCheckResult(
                provider_name="provider2",
                status=HealthStatus.UNHEALTHY,
                message="Failed",
                error="Connection error"
            )
        }

        with patch('models.health.check_all_providers', return_value=mock_results):
            print_health_status()

            captured = capsys.readouterr()
            assert "Provider Health Status" in captured.out
            assert "✅ PROVIDER1: healthy" in captured.out
            assert "❌ PROVIDER2: unhealthy" in captured.out
            assert "All good" in captured.out
            assert "Failed" in captured.out
            assert "Response time: 100ms" in captured.out
            assert "Model tested: test-model" in captured.out
            assert "Error: Connection error" in captured.out
            assert "Summary: 1/2 providers healthy" in captured.out

    def test_print_health_status_no_healthy_providers(self, capsys):
        """Test printing health status when no providers are healthy."""
        mock_results = {
            "provider1": HealthCheckResult("provider1", HealthStatus.UNHEALTHY, "Failed")
        }

        with patch('models.health.check_all_providers', return_value=mock_results):
            print_health_status()

            captured = capsys.readouterr()
            assert "Summary: 0/1 providers healthy" in captured.out
            assert "No providers are currently healthy!" in captured.out
            assert "Make sure you have API keys configured" in captured.out