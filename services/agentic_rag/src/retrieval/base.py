"""
Base classes for retrieval components in the Agentic RAG service.

This module defines abstract base classes for embeddings, rerankers, and vector stores,
providing a consistent interface for different implementations while following the
patterns established in the agent_intro service.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Document:
    """A document in the knowledge base."""
    content: str
    metadata: Dict[str, Any]
    doc_id: Optional[str] = None

    def __post_init__(self):
        """Generate doc_id if not provided."""
        if self.doc_id is None:
            import hashlib
            content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
            self.doc_id = f"doc_{content_hash}"


@dataclass
class SearchResult:
    """A search result from vector store query."""
    document: Document
    score: float
    rank: Optional[int] = None


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            List of float values representing the embedding
        """
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of input texts to embed

        Returns:
            List of embeddings, one for each input text
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass


class RerankerProvider(ABC):
    """Abstract base class for reranking providers."""

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
            top_k: Maximum number of results to return

        Returns:
            List of SearchResult objects, sorted by relevance
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the reranker model name."""
        pass


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store.

        Args:
            documents: List of documents to add
        """
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents.

        Args:
            query: Search query
            k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of SearchResult objects
        """
        pass

    @abstractmethod
    def search_by_vector(
        self,
        vector: List[float],
        k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search using a pre-computed vector.

        Args:
            vector: Query vector
            k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of SearchResult objects
        """
        pass

    @abstractmethod
    def delete_documents(self, doc_ids: List[str]) -> None:
        """Delete documents by their IDs.

        Args:
            doc_ids: List of document IDs to delete
        """
        pass

    @abstractmethod
    def get_document_count(self) -> int:
        """Get the total number of documents in the store.

        Returns:
            Number of documents
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all documents from the store."""
        pass


class RetrievalChain(ABC):
    """Abstract base class for retrieval chains."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        reranker: Optional[RerankerProvider] = None
    ):
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.reranker = reranker

    @abstractmethod
    def retrieve(
        self,
        query: str,
        k: int = 10,
        rerank_top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Retrieve relevant documents for a query.

        Args:
            query: Search query
            k: Number of results from vector search
            rerank_top_k: Number of results after reranking (if reranker available)
            filter_metadata: Optional metadata filters

        Returns:
            List of SearchResult objects
        """
        pass