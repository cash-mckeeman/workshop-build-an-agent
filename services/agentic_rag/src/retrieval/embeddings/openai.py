"""
OpenAI embedding provider for cloud-based embedding generation.

This module implements the EmbeddingProvider interface using OpenAI's API,
providing high-quality embeddings through their text-embedding models.
"""

from typing import List, Optional
import os
import logging

from .base import EmbeddingProvider, EmbeddingError

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider using the OpenAI API.

    This provider uses OpenAI's text-embedding models to generate embeddings,
    requiring an API key for authentication.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-ada-002",
        api_key: Optional[str] = None,
        organization: Optional[str] = None
    ):
        """Initialize the OpenAI embedding provider.

        Args:
            model_name: Name of the OpenAI embedding model to use
            api_key: OpenAI API key. If None, will try to read from OPENAI_API_KEY env var
            organization: OpenAI organization ID (optional)
        """
        super().__init__(model_name)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.organization = organization
        self._client = None
        self._dimension_cache: Optional[int] = None

        # Model dimension mapping
        self._model_dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }

    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is not None:
            return self._client

        if not self.api_key:
            raise EmbeddingError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        try:
            from openai import OpenAI
        except ImportError as e:
            raise EmbeddingError(
                "OpenAI library not installed. Install with: pip install openai"
            ) from e

        try:
            self._client = OpenAI(
                api_key=self.api_key,
                organization=self.organization
            )
            return self._client
        except Exception as e:
            raise EmbeddingError(f"Failed to create OpenAI client: {str(e)}") from e

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            List of float values representing the embedding

        Raises:
            EmbeddingError: If embedding generation fails
        """
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of input texts to embed

        Returns:
            List of embeddings, one for each input text

        Raises:
            EmbeddingError: If embedding generation fails
        """
        validated_texts = self.validate_texts(texts)
        client = self._get_client()

        try:
            logger.debug(f"Generating embeddings for {len(validated_texts)} texts using {self.model_name}")

            response = client.embeddings.create(
                input=validated_texts,
                model=self.model_name
            )

            embeddings = [data.embedding for data in response.data]
            logger.debug(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings

        except Exception as e:
            error_msg = f"Failed to generate embeddings with OpenAI: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg) from e

    @property
    def dimension(self) -> int:
        """Return the embedding dimension.

        Returns:
            Dimension of the embedding vectors
        """
        if self._dimension_cache is not None:
            return self._dimension_cache

        # Try to get from predefined mapping
        if self.model_name in self._model_dimensions:
            self._dimension_cache = self._model_dimensions[self.model_name]
            return self._dimension_cache

        # Fallback: generate a test embedding to get dimension
        try:
            test_embedding = self.embed_text("test")
            self._dimension_cache = len(test_embedding)
            return self._dimension_cache
        except Exception as e:
            raise EmbeddingError(f"Failed to determine embedding dimension: {str(e)}") from e

    def is_available(self) -> bool:
        """Check if the embedding provider is available.

        Returns:
            True if OpenAI API key is configured and accessible
        """
        try:
            if not self.api_key:
                return False

            client = self._get_client()
            # Test with a simple embedding request
            response = client.embeddings.create(
                input=["test"],
                model=self.model_name
            )
            return len(response.data) > 0

        except Exception as e:
            logger.warning(f"OpenAI embedding provider not available: {str(e)}")
            return False

    def get_model_info(self) -> dict:
        """Get information about the OpenAI model.

        Returns:
            Dictionary with model information
        """
        if not self.is_available():
            return {"available": False, "error": "OpenAI API not available"}

        return {
            "available": True,
            "model_name": self.model_name,
            "dimension": self.dimension,
            "provider": "OpenAI",
            "api_key_configured": bool(self.api_key)
        }


# Predefined OpenAI model configurations
OPENAI_EMBEDDING_MODELS = {
    "ada-002": {
        "model_name": "text-embedding-ada-002",
        "dimension": 1536,
        "description": "OpenAI's most capable embedding model (legacy)"
    },
    "3-small": {
        "model_name": "text-embedding-3-small",
        "dimension": 1536,
        "description": "OpenAI's newer small embedding model"
    },
    "3-large": {
        "model_name": "text-embedding-3-large",
        "dimension": 3072,
        "description": "OpenAI's newer large embedding model"
    }
}


def create_openai_embedding_provider(
    model_key: str = "ada-002",
    api_key: Optional[str] = None
) -> OpenAIEmbeddingProvider:
    """Create an OpenAI embedding provider with predefined model configuration.

    Args:
        model_key: Key from OPENAI_EMBEDDING_MODELS dictionary
        api_key: OpenAI API key (optional, will use env var if not provided)

    Returns:
        Configured OpenAIEmbeddingProvider

    Raises:
        ValueError: If model_key is not recognized
    """
    if model_key not in OPENAI_EMBEDDING_MODELS:
        available_models = ", ".join(OPENAI_EMBEDDING_MODELS.keys())
        raise ValueError(f"Unknown model key: {model_key}. Available: {available_models}")

    model_config = OPENAI_EMBEDDING_MODELS[model_key]
    return OpenAIEmbeddingProvider(
        model_name=model_config["model_name"],
        api_key=api_key
    )


def get_available_models() -> dict:
    """Get information about available OpenAI embedding models.

    Returns:
        Dictionary with model information
    """
    return OPENAI_EMBEDDING_MODELS.copy()


# Demo function for testing
def demo_openai_embeddings():
    """Demonstrate the OpenAI embedding provider."""
    print("🔤 OpenAI Embedding Provider Demo")
    print("=" * 40)

    try:
        # Create provider
        provider = create_openai_embedding_provider("ada-002")

        # Check availability
        if not provider.is_available():
            print("❌ OpenAI embedding provider not available")
            print("Set OPENAI_API_KEY environment variable")
            return

        # Show model info
        info = provider.get_model_info()
        print(f"✅ Model: {info['model_name']}")
        print(f"📏 Dimension: {info['dimension']}")
        print(f"🔑 API Key configured: {info['api_key_configured']}")

        # Test single embedding
        text = "This is a test sentence for embedding."
        embedding = provider.embed_text(text)
        print(f"\n📝 Text: {text}")
        print(f"🔢 Embedding dimension: {len(embedding)}")
        print(f"🔢 First 5 values: {embedding[:5]}")

        # Test batch embedding
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        embeddings = provider.embed_texts(texts)
        print(f"\n📚 Batch embedding: {len(texts)} texts → {len(embeddings)} embeddings")

    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    demo_openai_embeddings()