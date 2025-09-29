"""
Abstract base class for reranking providers.

This module provides the base interface for reranking providers, which improve
retrieval quality by re-scoring and re-ordering documents based on query relevance.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import logging

from ..base import Document, SearchResult

logger = logging.getLogger(__name__)


class RerankerProvider(ABC):
    """Abstract base class for reranking providers.

    Rerankers take an initial set of retrieved documents and re-score them
    based on their relevance to the query, potentially improving retrieval quality.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
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
            top_k: Maximum number of results to return (None for all)

        Returns:
            List of SearchResult objects, sorted by relevance score (highest first)

        Raises:
            RerankerError: If reranking fails
        """
        pass

    @abstractmethod
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
        pass

    @property
    def model_name_property(self) -> str:
        """Return the model name."""
        return self.model_name

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the reranker provider is available.

        Returns:
            True if the provider can rerank documents, False otherwise
        """
        pass

    def validate_inputs(self, query: str, documents: List[Document]) -> Tuple[str, List[Document]]:
        """Validate and preprocess inputs for reranking.

        Args:
            query: Search query to validate
            documents: List of documents to validate

        Returns:
            Tuple of (validated_query, validated_documents)

        Raises:
            ValueError: If inputs are invalid
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty or whitespace only")

        if not documents:
            raise ValueError("Document list cannot be empty")

        # Validate documents
        for i, doc in enumerate(documents):
            if not doc.content or not doc.content.strip():
                raise ValueError(f"Document {i} has empty content")

        # Basic preprocessing
        query = query.strip()

        return query, documents

    def create_search_results(
        self,
        documents: List[Document],
        scores: List[float],
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """Create SearchResult objects from documents and scores.

        Args:
            documents: List of documents
            scores: List of relevance scores
            top_k: Maximum number of results to return

        Returns:
            List of SearchResult objects, sorted by score (highest first)
        """
        if len(documents) != len(scores):
            raise ValueError("Number of documents must match number of scores")

        # Create search results
        results = []
        for doc, score in zip(documents, scores):
            results.append(SearchResult(document=doc, score=score))

        # Sort by score (highest first)
        results.sort(key=lambda x: x.score, reverse=True)

        # Add rank information
        for rank, result in enumerate(results):
            result.rank = rank + 1

        # Apply top_k limit
        if top_k is not None:
            results = results[:top_k]

        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"


class RerankerError(Exception):
    """Exception raised when reranking fails."""
    pass