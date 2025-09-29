"""
Embedding providers for the Agentic RAG service.

This package provides embedding providers that implement a consistent interface
for generating text embeddings using different models and services.

Available Providers:
- LocalEmbeddingProvider: Uses sentence-transformers for offline embeddings
- OpenAIEmbeddingProvider: Uses OpenAI's API for cloud-based embeddings

Example Usage:
    from retrieval.embeddings import create_embedding_provider

    # Local provider
    local_provider = create_embedding_provider("local", "all-MiniLM-L6-v2")

    # OpenAI provider
    openai_provider = create_embedding_provider("openai", "ada-002")

    # Generate embeddings
    embedding = provider.embed_text("Your text here")
    embeddings = provider.embed_texts(["Text 1", "Text 2", "Text 3"])
"""

from .base import EmbeddingProvider, EmbeddingError
from .local import LocalEmbeddingProvider, create_local_embedding_provider, EMBEDDING_MODELS
from .openai import OpenAIEmbeddingProvider, create_openai_embedding_provider, OPENAI_EMBEDDING_MODELS

from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def create_embedding_provider(
    provider_type: str,
    model_key: str,
    **kwargs
) -> EmbeddingProvider:
    """Create an embedding provider with the specified configuration.

    Args:
        provider_type: Type of provider ("local" or "openai")
        model_key: Model key for the specific provider
        **kwargs: Additional arguments passed to provider constructor

    Returns:
        Configured embedding provider

    Raises:
        ValueError: If provider_type or model_key is not recognized
        EmbeddingError: If provider cannot be created
    """
    provider_type = provider_type.lower()

    try:
        if provider_type == "local":
            return create_local_embedding_provider(model_key, **kwargs)
        elif provider_type == "openai":
            return create_openai_embedding_provider(model_key, **kwargs)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}. Available: local, openai")

    except Exception as e:
        if isinstance(e, (ValueError, EmbeddingError)):
            raise
        raise EmbeddingError(f"Failed to create {provider_type} embedding provider: {str(e)}") from e


def get_best_available_provider() -> Optional[EmbeddingProvider]:
    """Get the best available embedding provider.

    Tries providers in order of preference:
    1. OpenAI (if API key available)
    2. Local sentence-transformers

    Returns:
        Best available embedding provider, or None if none are available
    """
    # Try OpenAI first (usually better quality)
    try:
        provider = create_embedding_provider("openai", "ada-002")
        if provider.is_available():
            logger.info("Using OpenAI embedding provider")
            return provider
    except Exception as e:
        logger.debug(f"OpenAI provider not available: {str(e)}")

    # Try local provider
    try:
        provider = create_embedding_provider("local", "all-MiniLM-L6-v2")
        if provider.is_available():
            logger.info("Using local embedding provider")
            return provider
    except Exception as e:
        logger.debug(f"Local provider not available: {str(e)}")

    logger.warning("No embedding providers available")
    return None


def get_available_providers() -> Dict[str, Dict[str, Any]]:
    """Get information about all available embedding providers.

    Returns:
        Dictionary mapping provider names to their model information
    """
    providers = {}

    # Check local provider
    try:
        local_provider = create_embedding_provider("local", "all-MiniLM-L6-v2")
        providers["local"] = {
            "available": local_provider.is_available(),
            "models": EMBEDDING_MODELS,
            "description": "Local sentence-transformers models"
        }
    except Exception:
        providers["local"] = {
            "available": False,
            "models": EMBEDDING_MODELS,
            "description": "Local sentence-transformers models (not available)"
        }

    # Check OpenAI provider
    try:
        openai_provider = create_embedding_provider("openai", "ada-002")
        providers["openai"] = {
            "available": openai_provider.is_available(),
            "models": OPENAI_EMBEDDING_MODELS,
            "description": "OpenAI embedding models"
        }
    except Exception:
        providers["openai"] = {
            "available": False,
            "models": OPENAI_EMBEDDING_MODELS,
            "description": "OpenAI embedding models (not available)"
        }

    return providers


def demo_all_providers():
    """Demonstrate all available embedding providers."""
    print("=$ Embedding Providers Demo")
    print("=" * 50)

    providers_info = get_available_providers()

    for provider_name, info in providers_info.items():
        print(f"\n📝 {provider_name.upper()} Provider:")
        print(f"   Available: {info['available']}")
        print(f"   Description: {info['description']}")

        if info['available']:
            try:
                # Get the first model for demo
                model_key = list(info['models'].keys())[0]
                provider = create_embedding_provider(provider_name, model_key)

                # Test embedding
                test_text = "This is a test sentence."
                embedding = provider.embed_text(test_text)

                print(f"    Test successful: {len(embedding)}-dimensional embedding")
                print(f"   Model: {provider.model_name}")

            except Exception as e:
                print(f"   L Test failed: {str(e)}")
        else:
            print(f"   L Not available")

    # Show best available
    print(f"\n<� Best Available Provider:")
    best_provider = get_best_available_provider()
    if best_provider:
        print(f"   {best_provider.__class__.__name__}: {best_provider.model_name}")
    else:
        print("   None available")


__all__ = [
    "EmbeddingProvider",
    "EmbeddingError",
    "LocalEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "create_embedding_provider",
    "create_local_embedding_provider",
    "create_openai_embedding_provider",
    "get_best_available_provider",
    "get_available_providers",
    "EMBEDDING_MODELS",
    "OPENAI_EMBEDDING_MODELS"
]


if __name__ == "__main__":
    demo_all_providers()