"""
Abstract base class for embedding providers.

This module provides the base interface for embedding providers, following the
patterns established in the agent_intro service for clean abstraction and
consistent interfaces.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    This class defines the interface that all embedding providers must implement,
    whether they're local models (sentence-transformers) or cloud APIs (OpenAI).
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._dimension: Optional[int] = None

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            List of float values representing the embedding

        Raises:
            EmbeddingError: If embedding generation fails
        """
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        This method should be implemented to handle batch processing efficiently.

        Args:
            texts: List of input texts to embed

        Returns:
            List of embeddings, one for each input text

        Raises:
            EmbeddingError: If embedding generation fails
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension.

        Returns:
            Dimension of the embedding vectors
        """
        pass

    @property
    def model_name_property(self) -> str:
        """Return the model name."""
        return self.model_name

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the embedding provider is available.

        Returns:
            True if the provider can generate embeddings, False otherwise
        """
        pass

    def validate_text(self, text: str) -> str:
        """Validate and preprocess text for embedding.

        Args:
            text: Input text to validate

        Returns:
            Processed text ready for embedding

        Raises:
            ValueError: If text is invalid
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty or whitespace only")

        # Basic preprocessing
        text = text.strip()

        # Check length (most embedding models have limits)
        if len(text) > 8192:
            logger.warning(f"Text length ({len(text)}) exceeds recommended limit. Truncating.")
            text = text[:8192]

        return text

    def validate_texts(self, texts: List[str]) -> List[str]:
        """Validate and preprocess multiple texts for embedding.

        Args:
            texts: List of input texts to validate

        Returns:
            List of processed texts ready for embedding

        Raises:
            ValueError: If any text is invalid
        """
        if not texts:
            raise ValueError("Text list cannot be empty")

        return [self.validate_text(text) for text in texts]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"


class EmbeddingError(Exception):
    """Exception raised when embedding generation fails."""
    pass