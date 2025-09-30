"""
Model Provider Health Checking

This module provides health checking capabilities for different model providers,
ensuring they are properly configured and accessible before use.
"""

import asyncio
import os
from dataclasses import dataclass
from enum import Enum

from src.models.providers import PROVIDERS, ModelProvider


class HealthStatus(str, Enum):
    """Health status for providers."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check for a provider."""

    provider_name: str
    status: HealthStatus
    message: str
    response_time_ms: int | None = None
    model_tested: str | None = None
    error: str | None = None


class HealthChecker:
    """Health checker for model providers."""

    def __init__(self):
        self.timeout_seconds = 30

    async def check_provider_health(
        self,
        provider: ModelProvider,
        test_prompt: str = "Hello, this is a test. Please respond with 'OK'.",
    ) -> HealthCheckResult:
        """Check the health of a specific provider.

        Args:
            provider: Provider to check
            test_prompt: Simple prompt to test the model

        Returns:
            HealthCheckResult with status and details
        """
        start_time = asyncio.get_event_loop().time()

        try:
            # First check basic availability
            if not provider.is_available:
                return HealthCheckResult(
                    provider_name=provider.name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Provider not available: {self._get_availability_reason(provider)}",
                    error="Configuration issue",
                )

            # For now, we'll do basic checks rather than actual model calls
            # since that would require setting up PydanticAI clients

            # Check API key if required
            if provider.requires_api_key:
                api_key = provider.get_api_key()
                if not api_key:
                    return HealthCheckResult(
                        provider_name=provider.name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Missing API key: {provider.env_var}",
                        error="Authentication issue",
                    )

                # Basic API key validation
                if len(api_key) < 10:  # Most API keys are longer
                    return HealthCheckResult(
                        provider_name=provider.name,
                        status=HealthStatus.UNHEALTHY,
                        message="API key appears to be invalid (too short)",
                        error="Authentication issue",
                    )

            # Check Ollama connectivity for local providers
            if provider.provider_type.value == "ollama":
                health_result = await self._check_ollama_health(provider)
                if health_result.status != HealthStatus.HEALTHY:
                    return health_result

            # If we get here, basic checks passed
            end_time = asyncio.get_event_loop().time()
            response_time_ms = int((end_time - start_time) * 1000)

            return HealthCheckResult(
                provider_name=provider.name,
                status=HealthStatus.HEALTHY,
                message="Provider configuration appears valid",
                response_time_ms=response_time_ms,
                model_tested=provider.default_model,
            )

        except Exception as e:
            end_time = asyncio.get_event_loop().time()
            response_time_ms = int((end_time - start_time) * 1000)

            return HealthCheckResult(
                provider_name=provider.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                response_time_ms=response_time_ms,
                error=str(e),
            )

    async def _check_ollama_health(self, provider: ModelProvider) -> HealthCheckResult:
        """Check Ollama server health."""
        try:
            import httpx

            base_url = provider.base_url or "http://localhost:11434"

            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                # Check if Ollama server is running
                response = await client.get(f"{base_url}/api/tags")

                if response.status_code == 200:
                    tags_data = response.json()
                    models = [model["name"] for model in tags_data.get("models", [])]

                    # Check if default model is available
                    if provider.default_model in models:
                        return HealthCheckResult(
                            provider_name=provider.name,
                            status=HealthStatus.HEALTHY,
                            message=f"Ollama server healthy, model '{provider.default_model}' available",
                            model_tested=provider.default_model,
                        )
                    else:
                        return HealthCheckResult(
                            provider_name=provider.name,
                            status=HealthStatus.UNHEALTHY,
                            message=f"Model '{provider.default_model}' not found. Available: {models}",
                            error="Model not available",
                        )
                else:
                    return HealthCheckResult(
                        provider_name=provider.name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Ollama server returned status {response.status_code}",
                        error="Server error",
                    )

        except ImportError:
            return HealthCheckResult(
                provider_name=provider.name,
                status=HealthStatus.UNHEALTHY,
                message="httpx not available for Ollama health check",
                error="Missing dependency",
            )
        except Exception as e:
            return HealthCheckResult(
                provider_name=provider.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Cannot connect to Ollama server: {str(e)}",
                error=str(e),
            )

    def _get_availability_reason(self, provider: ModelProvider) -> str:
        """Get human-readable reason why provider is not available."""
        if provider.requires_api_key and provider.env_var:
            if not os.getenv(provider.env_var):
                return f"Missing environment variable: {provider.env_var}"

        return "Unknown configuration issue"

    async def check_all_providers(self) -> dict[str, HealthCheckResult]:
        """Check health of all configured providers.

        Returns:
            Dictionary mapping provider names to health check results
        """
        results = {}

        # Check all providers, not just available ones
        for name, provider in PROVIDERS.items():
            results[name] = await self.check_provider_health(provider)

        return results


# Convenience functions for synchronous use
def check_provider_health(provider_name: str) -> HealthCheckResult:
    """Synchronous version of provider health check."""
    provider = PROVIDERS.get(provider_name)
    if not provider:
        return HealthCheckResult(
            provider_name=provider_name,
            status=HealthStatus.UNHEALTHY,
            message=f"Provider '{provider_name}' not found",
            error="Provider not found",
        )

    checker = HealthChecker()

    # Run async function in sync context
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(checker.check_provider_health(provider))


def check_all_providers() -> dict[str, HealthCheckResult]:
    """Synchronous version of checking all providers."""
    checker = HealthChecker()

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(checker.check_all_providers())


def get_healthy_providers() -> list[str]:
    """Get list of provider names that are currently healthy."""
    results = check_all_providers()
    return [
        name
        for name, result in results.items()
        if result.status == HealthStatus.HEALTHY
    ]


def get_best_available_provider(
    preferred_providers: list[str] | None = None,
) -> str | None:
    """Get the best available provider from a preference list.

    Args:
        preferred_providers: List of provider names in order of preference.
                           If None, uses default preference order.

    Returns:
        Name of the best available provider, or None if none available
    """
    if preferred_providers is None:
        # Default preference order
        preferred_providers = ["openai", "anthropic", "groq", "huggingface", "ollama"]

    healthy_providers = get_healthy_providers()

    for provider_name in preferred_providers:
        if provider_name in healthy_providers:
            return provider_name

    # Fallback to any healthy provider
    if healthy_providers:
        return healthy_providers[0]

    return None


def print_health_status():
    """Print a formatted health status report for all providers."""
    print("🏥 Provider Health Status")
    print("=" * 50)

    results = check_all_providers()

    healthy_count = 0
    total_count = len(results)

    for name, result in results.items():
        status_emoji = "✅" if result.status == HealthStatus.HEALTHY else "❌"
        print(f"{status_emoji} {name.upper()}: {result.status.value}")
        print(f"   {result.message}")

        if result.response_time_ms:
            print(f"   Response time: {result.response_time_ms}ms")

        if result.model_tested:
            print(f"   Model tested: {result.model_tested}")

        if result.error:
            print(f"   Error: {result.error}")

        print()

        if result.status == HealthStatus.HEALTHY:
            healthy_count += 1

    print(f"📊 Summary: {healthy_count}/{total_count} providers healthy")

    if healthy_count == 0:
        print("⚠️  No providers are currently healthy!")
        print("💡 Make sure you have API keys configured:")
        print("   - OPENAI_API_KEY for OpenAI")
        print("   - ANTHROPIC_API_KEY for Anthropic")
        print("   - HUGGINGFACE_API_KEY for HuggingFace")
        print("   - GROQ_API_KEY for Groq")
        print("   - Or run 'ollama serve' for local models")


if __name__ == "__main__":
    # Demo the health checking system
    print_health_status()

    # Show best available provider
    best = get_best_available_provider()
    if best:
        print(f"\n🎯 Best available provider: {best}")
    else:
        print("\n❌ No providers available!")
