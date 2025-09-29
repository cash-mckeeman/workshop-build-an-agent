"""
Local reranking provider using cross-encoder models.

This module implements the RerankerProvider interface using local cross-encoder
models from sentence-transformers, providing offline reranking capabilities.
"""

from typing import List, Optional
import logging

from .base import RerankerProvider, RerankerError
from ..base import Document, SearchResult

logger = logging.getLogger(__name__)


class LocalRerankerProvider(RerankerProvider):
    """Local reranking provider using cross-encoder models.

    This provider uses local cross-encoder models from sentence-transformers
    to rerank documents based on query-document relevance scores.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None
    ):
        """Initialize the local reranker provider.

        Args:
            model_name: Name of the cross-encoder model to use
            device: Device to run the model on ('cpu', 'cuda', etc.). Auto-detected if None.
        """
        super().__init__(model_name)
        self.device = device
        self._model = None

    def _load_model(self):
        """Load the cross-encoder model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise RerankerError(
                "sentence-transformers not installed. Install with: pip install sentence-transformers"
            ) from e

        try:
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            self._model = CrossEncoder(self.model_name, device=self.device)
            logger.info(f"Model loaded successfully on device: {self._model.device}")
        except Exception as e:
            raise RerankerError(f"Failed to load model {self.model_name}: {str(e)}") from e

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """Rerank documents based on relevance to query.

        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Maximum number of results to return

        Returns:
            List of SearchResult objects, sorted by relevance score (highest first)

        Raises:
            RerankerError: If reranking fails
        """
        validated_query, validated_documents = self.validate_inputs(query, documents)

        # Generate scores
        scores = self.score_pairs(validated_query, validated_documents)

        # Create and return search results
        return self.create_search_results(validated_documents, scores, top_k)

    def score_pairs(
        self,
        query: str,
        documents: List[Document]
    ) -> List[float]:
        """Generate relevance scores for query-document pairs.

        Args:
            query: The search query
            documents: List of documents to score

        Returns:
            List of relevance scores, one for each document

        Raises:
            RerankerError: If scoring fails
        """
        validated_query, validated_documents = self.validate_inputs(query, documents)
        self._load_model()

        try:
            # Prepare query-document pairs for the cross-encoder
            pairs = []
            for doc in validated_documents:
                # Truncate document content if too long
                content = doc.content
                if len(content) > 512:  # Most cross-encoders have 512 token limit
                    content = content[:512]
                pairs.append([validated_query, content])

            # Get relevance scores
            scores = self._model.predict(pairs)

            # Convert to list if numpy array
            if hasattr(scores, 'tolist'):
                scores = scores.tolist()
            elif not isinstance(scores, list):
                scores = [float(scores)]

            logger.debug(f"Generated {len(scores)} relevance scores")
            return scores

        except Exception as e:
            raise RerankerError(f"Failed to generate relevance scores: {str(e)}") from e

    def is_available(self) -> bool:
        """Check if the reranker provider is available.

        Returns:
            True if sentence-transformers is installed and model can be loaded
        """
        try:
            from sentence_transformers import CrossEncoder  # noqa: F401
            self._load_model()
            return True
        except Exception as e:
            logger.warning(f"Local reranker provider not available: {str(e)}")
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
            "device": str(self._model.device) if self._model else "unknown",
            "max_length": getattr(self._model, "max_length", "unknown")
        }


# Predefined reranker model configurations
RERANKER_MODELS = {
    "ms-marco-MiniLM-L-6-v2": {
        "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "description": "Fast and efficient reranker trained on MS MARCO"
    },
    "ms-marco-MiniLM-L-12-v2": {
        "model_name": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "description": "Better quality reranker with more parameters"
    },
    "ms-marco-TinyBERT-L-2-v2": {
        "model_name": "cross-encoder/ms-marco-TinyBERT-L-2-v2",
        "description": "Very fast lightweight reranker"
    }
}


def create_local_reranker_provider(
    model_key: str = "ms-marco-MiniLM-L-6-v2",
    device: Optional[str] = None
) -> LocalRerankerProvider:
    """Create a local reranker provider with predefined model configuration.

    Args:
        model_key: Key from RERANKER_MODELS dictionary
        device: Device to run the model on

    Returns:
        Configured LocalRerankerProvider

    Raises:
        ValueError: If model_key is not recognized
    """
    if model_key not in RERANKER_MODELS:
        available_models = ", ".join(RERANKER_MODELS.keys())
        raise ValueError(f"Unknown model key: {model_key}. Available: {available_models}")

    model_config = RERANKER_MODELS[model_key]
    return LocalRerankerProvider(
        model_name=model_config["model_name"],
        device=device
    )


def get_available_models() -> dict:
    """Get information about available reranker models.

    Returns:
        Dictionary with model information
    """
    return RERANKER_MODELS.copy()


# Demo function for testing
def demo_local_reranker():
    """Demonstrate the local reranker provider."""
    print("🔄 Local Reranker Provider Demo")
    print("=" * 40)

    try:
        # Create provider
        provider = create_local_reranker_provider("ms-marco-MiniLM-L-6-v2")

        # Check availability
        if not provider.is_available():
            print("❌ Local reranker provider not available")
            print("Install with: pip install sentence-transformers")
            return

        # Show model info
        info = provider.get_model_info()
        print(f"✅ Model: {info['model_name']}")
        print(f"💻 Device: {info['device']}")

        # Create test documents
        query = "machine learning algorithms"
        documents = [
            Document(
                content="Machine learning algorithms are used to build predictive models.",
                metadata={"source": "doc1"}
            ),
            Document(
                content="Cooking recipes require precise measurements and timing.",
                metadata={"source": "doc2"}
            ),
            Document(
                content="Deep learning is a subset of machine learning that uses neural networks.",
                metadata={"source": "doc3"}
            ),
            Document(
                content="The weather forecast predicts rain tomorrow.",
                metadata={"source": "doc4"}
            )
        ]

        print(f"\n📝 Query: {query}")
        print(f"📚 Reranking {len(documents)} documents...")

        # Test reranking
        results = provider.rerank(query, documents, top_k=3)

        print(f"\n🏆 Top {len(results)} Results:")
        for result in results:
            print(f"   Rank {result.rank}: Score {result.score:.3f}")
            print(f"   Content: {result.document.content[:60]}...")
            print()

    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    demo_local_reranker()