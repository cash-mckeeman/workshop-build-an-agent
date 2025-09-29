"""
Vector store implementation using FAISS for efficient similarity search.

This module provides a concrete implementation of the VectorStore interface
using FAISS (Facebook AI Similarity Search) for fast vector operations.
"""

from typing import List, Dict, Any, Optional
import logging
import numpy as np
import json
import pickle
from pathlib import Path

from .base import VectorStore, Document, SearchResult, EmbeddingProvider

logger = logging.getLogger(__name__)


class FAISSVectorStore(VectorStore):
    """Vector store implementation using FAISS for similarity search.

    This implementation uses FAISS for efficient vector operations with
    support for metadata filtering and document management.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        index_type: str = "flat",
        store_path: Optional[str] = None
    ):
        """Initialize the FAISS vector store.

        Args:
            embedding_provider: Provider for generating embeddings
            index_type: Type of FAISS index ("flat", "ivf", "hnsw")
            store_path: Path to persist the vector store (optional)
        """
        self.embedding_provider = embedding_provider
        self.index_type = index_type.lower()
        self.store_path = Path(store_path) if store_path else None

        self._index = None
        self._documents: List[Document] = []
        self._document_lookup: Dict[str, int] = {}  # doc_id -> index
        self._metadata: List[Dict[str, Any]] = []

        self._initialize_index()

    def _initialize_index(self):
        """Initialize the FAISS index."""
        try:
            import faiss
        except ImportError as e:
            raise ImportError(
                "FAISS not installed. Install with: pip install faiss-cpu"
            ) from e

        dimension = self.embedding_provider.dimension

        if self.index_type == "flat":
            # L2 distance (Euclidean)
            self._index = faiss.IndexFlatL2(dimension)
        elif self.index_type == "ip":
            # Inner product (for normalized embeddings)
            self._index = faiss.IndexFlatIP(dimension)
        elif self.index_type == "ivf":
            # IVF with 100 centroids
            quantizer = faiss.IndexFlatL2(dimension)
            self._index = faiss.IndexIVFFlat(quantizer, dimension, 100)
        elif self.index_type == "hnsw":
            # HNSW index
            self._index = faiss.IndexHNSWFlat(dimension, 32)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        logger.info(f"Initialized FAISS index: {self.index_type}, dimension: {dimension}")

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store.

        Args:
            documents: List of documents to add
        """
        if not documents:
            return

        # Generate embeddings for all documents
        texts = [doc.content for doc in documents]
        embeddings = self.embedding_provider.embed_texts(texts)

        # Convert to numpy array
        vectors = np.array(embeddings, dtype=np.float32)

        # Add to FAISS index
        self._index.add(vectors)

        # Store documents and metadata
        start_idx = len(self._documents)
        for i, doc in enumerate(documents):
            idx = start_idx + i
            self._documents.append(doc)
            self._metadata.append(doc.metadata)

            # Update lookup
            if doc.doc_id:
                self._document_lookup[doc.doc_id] = idx

        logger.info(f"Added {len(documents)} documents to vector store. Total: {len(self._documents)}")

        # Train index if needed (for IVF)
        if self.index_type == "ivf" and not self._index.is_trained:
            if self._index.ntotal >= 100:  # Need at least 100 vectors for training
                all_vectors = self._get_all_vectors()
                self._index.train(all_vectors)
                logger.info("Trained IVF index")

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
        # Generate query embedding
        query_embedding = self.embedding_provider.embed_text(query)
        return self.search_by_vector(query_embedding, k, filter_metadata)

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
        if self._index.ntotal == 0:
            return []

        # Convert to numpy array
        query_vector = np.array([vector], dtype=np.float32)

        # Search in FAISS
        distances, indices = self._index.search(query_vector, k)

        # Create search results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue

            document = self._documents[idx]
            metadata = self._metadata[idx]

            # Apply metadata filtering
            if filter_metadata and not self._matches_filter(metadata, filter_metadata):
                continue

            # Convert distance to similarity score
            # For L2 distance, convert to similarity
            if self.index_type in ["flat", "ivf", "hnsw"]:
                score = 1.0 / (1.0 + distance)
            else:  # inner product
                score = float(distance)

            results.append(SearchResult(
                document=document,
                score=score,
                rank=i + 1
            ))

        # Sort by score (highest first)
        results.sort(key=lambda x: x.score, reverse=True)

        # Update ranks
        for i, result in enumerate(results):
            result.rank = i + 1

        return results

    def delete_documents(self, doc_ids: List[str]) -> None:
        """Delete documents by their IDs.

        Note: FAISS doesn't support efficient deletion, so this recreates the index.

        Args:
            doc_ids: List of document IDs to delete
        """
        if not doc_ids:
            return

        # Find indices to remove
        indices_to_remove = set()
        for doc_id in doc_ids:
            if doc_id in self._document_lookup:
                indices_to_remove.add(self._document_lookup[doc_id])

        if not indices_to_remove:
            return

        # Recreate data structures without deleted documents
        new_documents = []
        new_metadata = []
        new_lookup = {}

        for i, (doc, metadata) in enumerate(zip(self._documents, self._metadata)):
            if i not in indices_to_remove:
                new_idx = len(new_documents)
                new_documents.append(doc)
                new_metadata.append(metadata)
                if doc.doc_id:
                    new_lookup[doc.doc_id] = new_idx

        # Recreate index
        self._documents = new_documents
        self._metadata = new_metadata
        self._document_lookup = new_lookup

        # Rebuild FAISS index
        self._initialize_index()
        if self._documents:
            self.add_documents(self._documents.copy())

        logger.info(f"Deleted {len(indices_to_remove)} documents. Remaining: {len(self._documents)}")

    def get_document_count(self) -> int:
        """Get the total number of documents in the store.

        Returns:
            Number of documents
        """
        return len(self._documents)

    def clear(self) -> None:
        """Clear all documents from the store."""
        self._documents.clear()
        self._metadata.clear()
        self._document_lookup.clear()
        self._initialize_index()
        logger.info("Cleared vector store")

    def save(self, path: Optional[str] = None) -> None:
        """Save the vector store to disk.

        Args:
            path: Path to save to (uses store_path if not provided)
        """
        save_path = Path(path) if path else self.store_path
        if not save_path:
            raise ValueError("No save path specified")

        save_path.mkdir(parents=True, exist_ok=True)

        try:
            import faiss

            # Save FAISS index
            faiss.write_index(self._index, str(save_path / "index.faiss"))

            # Save documents and metadata
            with open(save_path / "documents.pkl", "wb") as f:
                pickle.dump(self._documents, f)

            with open(save_path / "metadata.json", "w") as f:
                json.dump(self._metadata, f, indent=2)

            with open(save_path / "lookup.json", "w") as f:
                json.dump(self._document_lookup, f, indent=2)

            # Save configuration
            config = {
                "index_type": self.index_type,
                "dimension": self.embedding_provider.dimension,
                "embedding_provider": {
                    "class": self.embedding_provider.__class__.__name__,
                    "model_name": self.embedding_provider.model_name
                }
            }
            with open(save_path / "config.json", "w") as f:
                json.dump(config, f, indent=2)

            logger.info(f"Saved vector store to {save_path}")

        except Exception as e:
            logger.error(f"Failed to save vector store: {str(e)}")
            raise

    def load(self, path: Optional[str] = None) -> None:
        """Load the vector store from disk.

        Args:
            path: Path to load from (uses store_path if not provided)
        """
        load_path = Path(path) if path else self.store_path
        if not load_path or not load_path.exists():
            raise ValueError("Load path does not exist")

        try:
            import faiss

            # Load FAISS index
            self._index = faiss.read_index(str(load_path / "index.faiss"))

            # Load documents and metadata
            with open(load_path / "documents.pkl", "rb") as f:
                self._documents = pickle.load(f)

            with open(load_path / "metadata.json", "r") as f:
                self._metadata = json.load(f)

            with open(load_path / "lookup.json", "r") as f:
                self._document_lookup = json.load(f)

            logger.info(f"Loaded vector store from {load_path}")
            logger.info(f"Loaded {len(self._documents)} documents")

        except Exception as e:
            logger.error(f"Failed to load vector store: {str(e)}")
            raise

    def _get_all_vectors(self) -> np.ndarray:
        """Get all vectors from the index for training purposes."""
        all_texts = [doc.content for doc in self._documents]
        all_embeddings = self.embedding_provider.embed_texts(all_texts)
        return np.array(all_embeddings, dtype=np.float32)

    def _matches_filter(self, metadata: Dict[str, Any], filter_metadata: Dict[str, Any]) -> bool:
        """Check if metadata matches the filter criteria.

        Args:
            metadata: Document metadata
            filter_metadata: Filter criteria

        Returns:
            True if metadata matches filter
        """
        for key, value in filter_metadata.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True

    def get_info(self) -> Dict[str, Any]:
        """Get information about the vector store.

        Returns:
            Dictionary with store information
        """
        return {
            "index_type": self.index_type,
            "dimension": self.embedding_provider.dimension,
            "document_count": len(self._documents),
            "index_size": self._index.ntotal if self._index else 0,
            "embedding_provider": {
                "class": self.embedding_provider.__class__.__name__,
                "model_name": self.embedding_provider.model_name
            },
            "is_trained": getattr(self._index, "is_trained", True)
        }


def create_vector_store(
    embedding_provider: EmbeddingProvider,
    index_type: str = "flat",
    store_path: Optional[str] = None
) -> FAISSVectorStore:
    """Create a FAISS vector store with the specified configuration.

    Args:
        embedding_provider: Provider for generating embeddings
        index_type: Type of FAISS index ("flat", "ivf", "hnsw")
        store_path: Path to persist the vector store

    Returns:
        Configured FAISSVectorStore
    """
    return FAISSVectorStore(
        embedding_provider=embedding_provider,
        index_type=index_type,
        store_path=store_path
    )


# Demo function for testing
def demo_vector_store():
    """Demonstrate the FAISS vector store."""
    print("🗃️ FAISS Vector Store Demo")
    print("=" * 40)

    try:
        # Create embedding provider
        from .embeddings import get_best_available_provider
        embedding_provider = get_best_available_provider()

        if not embedding_provider:
            print("❌ No embedding provider available")
            return

        # Create vector store
        vector_store = create_vector_store(embedding_provider)
        print(f"✅ Created vector store with {embedding_provider.__class__.__name__}")

        # Add test documents
        documents = [
            Document(
                content="Machine learning is a subset of artificial intelligence.",
                metadata={"topic": "ai", "category": "definition"}
            ),
            Document(
                content="Python is a popular programming language for data science.",
                metadata={"topic": "programming", "category": "language"}
            ),
            Document(
                content="Deep learning uses neural networks with multiple layers.",
                metadata={"topic": "ai", "category": "technique"}
            ),
            Document(
                content="Vector databases enable efficient similarity search.",
                metadata={"topic": "database", "category": "technology"}
            )
        ]

        vector_store.add_documents(documents)
        print(f"📚 Added {len(documents)} documents")

        # Test search
        query = "artificial intelligence and machine learning"
        results = vector_store.search(query, k=3)

        print(f"\n🔍 Search results for: '{query}'")
        for result in results:
            print(f"   Score: {result.score:.3f}")
            print(f"   Content: {result.document.content}")
            print(f"   Metadata: {result.document.metadata}")
            print()

        # Test metadata filtering
        ai_results = vector_store.search(query, k=5, filter_metadata={"topic": "ai"})
        print(f"🎯 Filtered results (topic=ai): {len(ai_results)} documents")

        # Show info
        info = vector_store.get_info()
        print(f"\n📊 Vector Store Info:")
        print(f"   Documents: {info['document_count']}")
        print(f"   Dimension: {info['dimension']}")
        print(f"   Index type: {info['index_type']}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    demo_vector_store()