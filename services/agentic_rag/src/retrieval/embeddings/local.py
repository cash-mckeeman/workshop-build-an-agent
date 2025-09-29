"""
Local embedding provider using sentence-transformers.

This module implements the EmbeddingProvider interface using local sentence-transformer
models, providing an offline option for embedding generation without requiring API keys.
"""

from typing import List, Optional
import logging

from .base import EmbeddingProvider, EmbeddingError

logger = logging.getLogger(__name__)


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local embedding provider using sentence-transformers.

    This provider uses local models from the sentence-transformers library,
    allowing for embedding generation without external API calls.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        """Initialize the local embedding provider.

        Args:
            model_name: Name of the sentence-transformers model to use
            device: Device to run the model on ('cpu', 'cuda', etc.). Auto-detected if None.
        """
        super().__init__(model_name)
        self.device = device
        self._model = None
        self._dimension_cache: Optional[int] = None

    def _load_model(self):
        """Load the sentence-transformers model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EmbeddingError(
                "sentence-transformers not installed. Install with: pip install sentence-transformers"
            ) from e

        try:
            logger.info(f"Loading sentence-transformers model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Model loaded successfully on device: {self._model.device}")
        except Exception as e:
            raise EmbeddingError(f"Failed to load model {self.model_name}: {str(e)}") from e

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            List of float values representing the embedding

        Raises:
            EmbeddingError: If embedding generation fails
        """
        validated_text = self.validate_text(text)
        self._load_model()

        try:
            embedding = self._model.encode(validated_text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embedding: {str(e)}") from e

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
        self._load_model()

        try:
            embeddings = self._model.encode(validated_texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embeddings: {str(e)}") from e

    @property
    def dimension(self) -> int:
        """Return the embedding dimension.

        Returns:
            Dimension of the embedding vectors
        """
        if self._dimension_cache is not None:
            return self._dimension_cache

        self._load_model()

        # Get dimension from model
        try:
            self._dimension_cache = self._model.get_sentence_embedding_dimension()
            return self._dimension_cache
        except Exception as e:
            # Fallback: generate a test embedding to get dimension
            try:
                test_embedding = self.embed_text("test")
                self._dimension_cache = len(test_embedding)
                return self._dimension_cache
            except Exception:
                raise EmbeddingError(f"Failed to determine embedding dimension: {str(e)}") from e

    def is_available(self) -> bool:
        """Check if the embedding provider is available.

        Returns:
            True if sentence-transformers is installed and model can be loaded
        """
        try:
            import sentence_transformers  # noqa: F401
            self._load_model()
            return True
        except Exception as e:
            logger.warning(f"Local embedding provider not available: {str(e)}")
            return False

    def get_model_info(self) -> dict:
        """Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        if not self.is_available():
            return {"available": False, "error": "Model not available"}

        self._load_model()

        return {
            "available": True,
            "model_name": self.model_name,
            "dimension": self.dimension,
            "device": str(self._model.device) if self._model else "unknown",
            "max_seq_length": getattr(self._model, "max_seq_length", "unknown")
        }


# Predefined model configurations
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "description": "Fast and efficient model, good for most use cases"
    },
    "all-mpnet-base-v2": {
        "model_name": "sentence-transformers/all-mpnet-base-v2",
        "dimension": 768,
        "description": "High quality model with better performance"
    },
    "all-MiniLM-L12-v2": {
        "model_name": "sentence-transformers/all-MiniLM-L12-v2",
        "dimension": 384,
        "description": "Balanced model between speed and quality"
    }
}


def create_local_embedding_provider(
    model_key: str = "all-MiniLM-L6-v2",
    device: Optional[str] = None
) -> LocalEmbeddingProvider:
    """Create a local embedding provider with predefined model configuration.

    Args:
        model_key: Key from EMBEDDING_MODELS dictionary
        device: Device to run the model on

    Returns:
        Configured LocalEmbeddingProvider

    Raises:
        ValueError: If model_key is not recognized
    """
    if model_key not in EMBEDDING_MODELS:
        available_models = ", ".join(EMBEDDING_MODELS.keys())
        raise ValueError(f"Unknown model key: {model_key}. Available: {available_models}")

    model_config = EMBEDDING_MODELS[model_key]
    return LocalEmbeddingProvider(
        model_name=model_config["model_name"],
        device=device
    )


def get_available_models() -> dict:
    """Get information about available embedding models.

    Returns:
        Dictionary with model information
    """
    return EMBEDDING_MODELS.copy()


# Demo function for testing
def demo_local_embeddings():
    """Demonstrate the local embedding provider."""
    print("🔤 Local Embedding Provider Demo")
    print("=" * 40)

    try:
        # Create provider
        provider = create_local_embedding_provider("all-MiniLM-L6-v2")

        # Check availability
        if not provider.is_available():
            print("❌ Local embedding provider not available")
            print("Install with: pip install sentence-transformers")
            return

        # Show model info
        info = provider.get_model_info()
        print(f"✅ Model: {info['model_name']}")
        print(f"📏 Dimension: {info['dimension']}")
        print(f"💻 Device: {info['device']}")

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
    demo_local_embeddings()