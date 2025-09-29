"""
Reranking providers for the Agentic RAG service.

This package provides reranking providers that improve retrieval quality by
re-scoring and re-ordering documents based on query-document relevance.

Available Providers:
- LocalRerankerProvider: Uses cross-encoder models for offline reranking

Example Usage:
    from retrieval.reranker import create_reranker_provider

    # Local provider
    reranker = create_reranker_provider("local", "ms-marco-MiniLM-L-6-v2")

    # Rerank documents
    results = reranker.rerank(query, documents, top_k=5)
"""

from .base import RerankerProvider, RerankerError
from .local import LocalRerankerProvider, create_local_reranker_provider, RERANKER_MODELS

from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def create_reranker_provider(
    provider_type: str,
    model_key: str,
    **kwargs
) -> RerankerProvider:
    """Create a reranker provider with the specified configuration.

    Args:
        provider_type: Type of provider ("local")
        model_key: Model key for the specific provider
        **kwargs: Additional arguments passed to provider constructor

    Returns:
        Configured reranker provider

    Raises:
        ValueError: If provider_type or model_key is not recognized
        RerankerError: If provider cannot be created
    """
    provider_type = provider_type.lower()

    try:
        if provider_type == "local":
            return create_local_reranker_provider(model_key, **kwargs)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}. Available: local")

    except Exception as e:
        if isinstance(e, (ValueError, RerankerError)):
            raise
        raise RerankerError(f"Failed to create {provider_type} reranker provider: {str(e)}") from e


def get_best_available_provider() -> Optional[RerankerProvider]:
    """Get the best available reranker provider.

    Returns:
        Best available reranker provider, or None if none are available
    """
    # Try local provider (currently the only one available)
    try:
        provider = create_reranker_provider("local", "ms-marco-MiniLM-L-6-v2")
        if provider.is_available():
            logger.info("Using local reranker provider")
            return provider
    except Exception as e:
        logger.debug(f"Local reranker provider not available: {str(e)}")

    logger.warning("No reranker providers available")
    return None


def get_available_providers() -> Dict[str, Dict[str, Any]]:
    """Get information about all available reranker providers.

    Returns:
        Dictionary mapping provider names to their model information
    """
    providers = {}

    # Check local provider
    try:
        local_provider = create_reranker_provider("local", "ms-marco-MiniLM-L-6-v2")
        providers["local"] = {
            "available": local_provider.is_available(),
            "models": RERANKER_MODELS,
            "description": "Local cross-encoder reranking models"
        }
    except Exception:
        providers["local"] = {
            "available": False,
            "models": RERANKER_MODELS,
            "description": "Local cross-encoder reranking models (not available)"
        }

    return providers


def demo_all_providers():
    """Demonstrate all available reranker providers."""
    print("= Reranker Providers Demo")
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
                provider = create_reranker_provider(provider_name, model_key)

                # Test reranking
                from ..base import Document
                query = "test query"
                documents = [
                    Document(content="Relevant test document", metadata={}),
                    Document(content="Less relevant content", metadata={})
                ]

                results = provider.rerank(query, documents)
                print(f"    Test successful: Reranked {len(results)} documents")
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
    "RerankerProvider",
    "RerankerError",
    "LocalRerankerProvider",
    "create_reranker_provider",
    "create_local_reranker_provider",
    "get_best_available_provider",
    "get_available_providers",
    "RERANKER_MODELS"
]


if __name__ == "__main__":
    demo_all_providers()