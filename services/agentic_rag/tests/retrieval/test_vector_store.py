"""
Tests for FAISS vector store implementation.

This module tests the FAISSVectorStore class which provides vector storage
and similarity search capabilities for the RAG system.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import shutil

from src.retrieval.vector_store import (
    FAISSVectorStore,
    create_vector_store
)
from src.retrieval.base import Document, SearchResult


class TestFAISSVectorStore:
    """Test cases for FAISSVectorStore class."""

    @patch('src.retrieval.vector_store.FAISS_AVAILABLE', True)
    @patch('src.retrieval.vector_store.faiss')
    def test_vector_store_initialization(self, mock_faiss, mock_embedding_provider):
        """Test vector store initialization with different index types."""
        # Setup mock FAISS
        mock_index = Mock()
        mock_quantizer = Mock()
        mock_faiss.IndexFlatL2.return_value = mock_index
        mock_faiss.IndexFlatIP.return_value = mock_index
        mock_faiss.IndexIVFFlat.return_value = mock_index
        mock_faiss.IndexHNSWFlat.return_value = mock_index

        # Test flat index (default)
        store = FAISSVectorStore(mock_embedding_provider)
        assert store.index_type == "flat"
        assert store.embedding_provider == mock_embedding_provider

        # Test different index types
        for index_type in ["flat", "ip", "ivf", "hnsw"]:
            store = FAISSVectorStore(mock_embedding_provider, index_type=index_type)
            assert store.index_type == index_type

    @patch('src.retrieval.vector_store.FAISS_AVAILABLE', True)
    @patch('src.retrieval.vector_store.faiss')
    def test_vector_store_invalid_index_type(self, mock_faiss, mock_embedding_provider):
        """Test vector store with invalid index type."""
        # Setup mock FAISS
        mock_index = Mock()
        mock_faiss.IndexFlatL2.return_value = mock_index

        with pytest.raises(ValueError, match="Unknown index type"):
            FAISSVectorStore(mock_embedding_provider, index_type="invalid")

    @patch('src.retrieval.vector_store.FAISS_AVAILABLE', True)
    @patch('src.retrieval.vector_store.faiss')
    def test_add_documents(self, mock_faiss, mock_embedding_provider, sample_documents):
        """Test adding documents to the vector store."""
        # Setup mock FAISS
        mock_index = Mock()
        mock_index.add = Mock()
        mock_faiss.IndexFlatL2.return_value = mock_index

        store = FAISSVectorStore(mock_embedding_provider)

        # Initially empty
        assert store.get_document_count() == 0

        # Add documents
        store.add_documents(sample_documents)

        # Check document count
        assert store.get_document_count() == len(sample_documents)

        # Check documents are stored
        assert len(store._documents) == len(sample_documents)
        assert len(store._metadata) == len(sample_documents)

        # Check document lookup
        for doc in sample_documents:
            if doc.doc_id:
                assert doc.doc_id in store._document_lookup

    @patch('src.retrieval.vector_store.FAISS_AVAILABLE', True)
    @patch('src.retrieval.vector_store.faiss')
    def test_add_empty_documents(self, mock_faiss, mock_embedding_provider):
        """Test adding empty document list."""
        # Setup mock FAISS
        mock_index = Mock()
        mock_faiss.IndexFlatL2.return_value = mock_index

        store = FAISSVectorStore(mock_embedding_provider)
        store.add_documents([])
        assert store.get_document_count() == 0

    def test_search_basic(self, mock_vector_store):
        """Test basic search functionality."""
        query = "AI framework"
        results = mock_vector_store.search(query, k=3)

        assert isinstance(results, list)
        assert len(results) <= 3
        assert len(results) > 0

        # Check result structure
        for result in results:
            assert isinstance(result, SearchResult)
            assert hasattr(result, 'document')
            assert hasattr(result, 'score')
            assert hasattr(result, 'rank')
            assert isinstance(result.score, float)
            assert isinstance(result.rank, int)

    def test_search_with_metadata_filter(self, mock_vector_store):
        """Test search with metadata filtering."""
        query = "programming"

        # Search without filter
        all_results = mock_vector_store.search(query, k=10)

        # Search with filter
        filtered_results = mock_vector_store.search(
            query,
            k=10,
            filter_metadata={"topic": "ai"}
        )

        # Filtered results should have fewer or equal results
        assert len(filtered_results) <= len(all_results)

        # All filtered results should match the filter
        for result in filtered_results:
            assert result.document.metadata.get("topic") == "ai"

    @patch('src.retrieval.vector_store.FAISS_AVAILABLE', True)
    @patch('src.retrieval.vector_store.faiss')
    def test_search_empty_store(self, mock_faiss, mock_embedding_provider):
        """Test search on empty vector store."""
        # Setup mock FAISS
        mock_index = Mock()
        mock_index.ntotal = 0
        mock_faiss.IndexFlatL2.return_value = mock_index

        store = FAISSVectorStore(mock_embedding_provider)
        results = store.search("test query")
        assert results == []

    def test_search_by_vector(self, mock_vector_store):
        """Test search using pre-computed vector."""
        # Create a test vector
        test_vector = [0.1] * 384  # Match embedding dimension

        results = mock_vector_store.search_by_vector(test_vector, k=2)

        assert isinstance(results, list)
        assert len(results) <= 2

        for result in results:
            assert isinstance(result, SearchResult)

    def test_delete_documents(self, mock_vector_store, sample_documents):
        """Test document deletion."""
        initial_count = mock_vector_store.get_document_count()
        assert initial_count == len(sample_documents)

        # Delete by doc_id
        doc_ids_to_delete = ["doc_1", "doc_3"]
        mock_vector_store.delete_documents(doc_ids_to_delete)

        # Check count decreased
        new_count = mock_vector_store.get_document_count()
        assert new_count == initial_count - len(doc_ids_to_delete)

        # Check deleted documents are gone
        for doc_id in doc_ids_to_delete:
            assert doc_id not in mock_vector_store._document_lookup

    def test_delete_nonexistent_documents(self, mock_vector_store):
        """Test deleting non-existent documents."""
        initial_count = mock_vector_store.get_document_count()

        # Try to delete non-existent documents
        mock_vector_store.delete_documents(["nonexistent_1", "nonexistent_2"])

        # Count should remain the same
        assert mock_vector_store.get_document_count() == initial_count

    def test_clear_store(self, mock_vector_store):
        """Test clearing the vector store."""
        # Initially has documents
        assert mock_vector_store.get_document_count() > 0

        # Clear
        mock_vector_store.clear()

        # Should be empty
        assert mock_vector_store.get_document_count() == 0
        assert len(mock_vector_store._documents) == 0
        assert len(mock_vector_store._metadata) == 0
        assert len(mock_vector_store._document_lookup) == 0

    def test_get_info(self, mock_vector_store):
        """Test getting vector store information."""
        info = mock_vector_store.get_info()

        assert isinstance(info, dict)
        assert "index_type" in info
        assert "dimension" in info
        assert "document_count" in info
        assert "index_size" in info
        assert "embedding_provider" in info

        assert info["index_type"] == "flat"
        assert info["dimension"] == 384
        assert info["document_count"] > 0

    @patch('src.retrieval.vector_store.FAISS_AVAILABLE', True)
    @patch('src.retrieval.vector_store.faiss')
    def test_save_and_load_vector_store(self, mock_faiss, mock_embedding_provider, sample_documents, temp_dir):
        """Test saving and loading vector store."""
        # Setup mock FAISS
        mock_index = Mock()
        mock_index.add = Mock()
        mock_faiss.IndexFlatL2.return_value = mock_index
        mock_faiss.write_index = Mock()
        mock_faiss.read_index = Mock(return_value=mock_index)

        store_path = Path(temp_dir) / "test_store"

        # Create and populate store
        store1 = FAISSVectorStore(mock_embedding_provider, store_path=str(store_path))
        store1.add_documents(sample_documents[:2])  # Add subset for testing

        # Save
        store1.save()
        assert store_path.exists()
        assert (store_path / "documents.pkl").exists()
        assert (store_path / "metadata.json").exists()
        assert (store_path / "config.json").exists()

        # Create new store and load
        store2 = FAISSVectorStore(mock_embedding_provider, store_path=str(store_path))
        store2.load()

        # Check loaded data
        assert store2.get_document_count() == len(sample_documents[:2])
        assert len(store2._documents) == len(sample_documents[:2])
        assert len(store2._metadata) == len(sample_documents[:2])

    @patch('src.retrieval.vector_store.FAISS_AVAILABLE', True)
    @patch('src.retrieval.vector_store.faiss')
    def test_save_without_path(self, mock_faiss, mock_embedding_provider):
        """Test saving without specifying path."""
        # Setup mock FAISS
        mock_index = Mock()
        mock_faiss.IndexFlatL2.return_value = mock_index

        store = FAISSVectorStore(mock_embedding_provider)

        with pytest.raises(ValueError, match="No save path specified"):
            store.save()

    @patch('src.retrieval.vector_store.FAISS_AVAILABLE', True)
    @patch('src.retrieval.vector_store.faiss')
    def test_load_nonexistent_path(self, mock_faiss, mock_embedding_provider):
        """Test loading from non-existent path."""
        # Setup mock FAISS
        mock_index = Mock()
        mock_faiss.IndexFlatL2.return_value = mock_index

        store = FAISSVectorStore(mock_embedding_provider, store_path="/nonexistent/path")

        with pytest.raises(ValueError, match="Load path does not exist"):
            store.load()

    @patch('src.retrieval.vector_store.FAISS_AVAILABLE', False)
    def test_faiss_import_error(self, mock_embedding_provider):
        """Test handling of FAISS import error."""
        # With FAISS_AVAILABLE set to False, initialization should fail
        with pytest.raises(ImportError, match="FAISS not installed"):
            FAISSVectorStore(mock_embedding_provider)

    @patch('src.retrieval.vector_store.FAISS_AVAILABLE', True)
    @patch('src.retrieval.vector_store.faiss')
    def test_ivf_index_training(self, mock_faiss, mock_embedding_provider, sample_documents):
        """Test IVF index training with sufficient data."""
        # Setup mock FAISS
        mock_index = Mock()
        mock_index.is_trained = True
        mock_index.ntotal = 120
        mock_quantizer = Mock()

        mock_faiss.IndexFlatL2.return_value = mock_quantizer
        mock_faiss.IndexIVFFlat.return_value = mock_index

        # Create IVF store
        store = FAISSVectorStore(mock_embedding_provider, index_type="ivf")

        # Create many documents to trigger training (need at least 100)
        many_docs = []
        for i in range(120):  # 120 documents > 100 threshold
            doc = Document(
                content=f"Test document {i} with varied content about topic {i % 10}",
                metadata={"id": i, "topic": f"topic_{i % 10}"},
                doc_id=f"doc_{i}"
            )
            many_docs.append(doc)

        # Add documents which should trigger training
        store.add_documents(many_docs)

        # Check that training was triggered
        assert store._index.is_trained
        assert store.get_document_count() == 120

    def test_metadata_filter_matching(self, mock_vector_store):
        """Test metadata filter matching logic."""
        # Test internal _matches_filter method
        metadata1 = {"topic": "ai", "category": "framework"}
        metadata2 = {"topic": "programming", "category": "language"}

        # Exact match
        filter1 = {"topic": "ai"}
        assert mock_vector_store._matches_filter(metadata1, filter1)
        assert not mock_vector_store._matches_filter(metadata2, filter1)

        # Multiple criteria
        filter2 = {"topic": "ai", "category": "framework"}
        assert mock_vector_store._matches_filter(metadata1, filter2)
        assert not mock_vector_store._matches_filter(metadata2, filter2)

        # Non-existent key
        filter3 = {"nonexistent": "value"}
        assert not mock_vector_store._matches_filter(metadata1, filter3)

    def test_search_result_ranking(self, mock_vector_store):
        """Test that search results are properly ranked."""
        query = "test query"
        results = mock_vector_store.search(query, k=5)

        if len(results) > 1:
            # Check ranks are consecutive
            for i, result in enumerate(results):
                assert result.rank == i + 1

            # Check scores are in descending order
            for i in range(len(results) - 1):
                assert results[i].score >= results[i + 1].score


class TestVectorStoreFactory:
    """Test cases for vector store factory functions."""

    @patch('src.retrieval.vector_store.FAISS_AVAILABLE', True)
    @patch('src.retrieval.vector_store.faiss')
    def test_create_vector_store(self, mock_faiss, mock_embedding_provider):
        """Test vector store creation factory function."""
        # Setup mock FAISS
        mock_index = Mock()
        mock_faiss.IndexFlatL2.return_value = mock_index

        store = create_vector_store(mock_embedding_provider)

        assert isinstance(store, FAISSVectorStore)
        assert store.embedding_provider == mock_embedding_provider
        assert store.index_type == "flat"

    @patch('src.retrieval.vector_store.FAISS_AVAILABLE', True)
    @patch('src.retrieval.vector_store.faiss')
    def test_create_vector_store_with_options(self, mock_faiss, mock_embedding_provider, temp_dir):
        """Test vector store creation with custom options."""
        # Setup mock FAISS
        mock_index = Mock()
        mock_faiss.IndexHNSWFlat.return_value = mock_index

        store_path = str(Path(temp_dir) / "custom_store")

        store = create_vector_store(
            embedding_provider=mock_embedding_provider,
            index_type="hnsw",
            store_path=store_path
        )

        assert isinstance(store, FAISSVectorStore)
        assert store.index_type == "hnsw"
        assert str(store.store_path) == store_path


class TestVectorStoreDemo:
    """Test cases for vector store demo functionality."""

    @patch('src.retrieval.vector_store.FAISS_AVAILABLE', True)
    @patch('src.retrieval.vector_store.faiss')
    @patch('src.retrieval.embeddings.get_best_available_provider')
    @patch('builtins.print')
    def test_demo_vector_store_success(self, mock_print, mock_get_provider, mock_faiss, mock_embedding_provider):
        """Test successful vector store demo."""
        # Setup mock FAISS
        mock_index = Mock()
        mock_index.ntotal = 4
        mock_index.search.return_value = (np.array([[0.1, 0.2, 0.3]]), np.array([[0, 1, 2]]))
        mock_faiss.IndexFlatL2.return_value = mock_index

        # Mock the provider getter
        mock_get_provider.return_value = mock_embedding_provider

        # Import and run demo
        from src.retrieval.vector_store import demo_vector_store
        demo_vector_store()

        # Check that print was called (demo ran)
        assert mock_print.called
        print_calls = [str(call.args[0]) if call.args else str(call) for call in mock_print.call_args_list]

        # Check for expected output patterns
        assert any("Vector Store Demo" in call for call in print_calls)
        assert any("Created vector store" in call for call in print_calls)

    @patch('src.retrieval.embeddings.get_best_available_provider')
    @patch('builtins.print')
    def test_demo_vector_store_no_provider(self, mock_print, mock_get_provider):
        """Test vector store demo with no provider available."""
        # Mock no provider available
        mock_get_provider.return_value = None

        # Import and run demo
        from src.retrieval.vector_store import demo_vector_store
        demo_vector_store()

        # Check that error message was printed
        print_calls = [str(call.args[0]) if call.args else str(call) for call in mock_print.call_args_list]
        assert any("No embedding provider available" in call for call in print_calls)

    @patch('src.retrieval.vector_store.FAISS_AVAILABLE', True)
    @patch('src.retrieval.vector_store.faiss')
    @patch('src.retrieval.embeddings.get_best_available_provider')
    @patch('builtins.print')
    def test_demo_vector_store_exception(self, mock_print, mock_get_provider, mock_faiss, mock_embedding_provider):
        """Test vector store demo with exception."""
        # Setup mock FAISS
        mock_index = Mock()
        mock_faiss.IndexFlatL2.return_value = mock_index

        # Mock provider but cause exception
        mock_embedding_provider.embed_texts.side_effect = Exception("Test error")
        mock_get_provider.return_value = mock_embedding_provider

        # Import and run demo
        from src.retrieval.vector_store import demo_vector_store
        demo_vector_store()

        # Check that error was handled
        print_calls = [str(call.args[0]) if call.args else str(call) for call in mock_print.call_args_list]
        assert any("Error" in call for call in print_calls)


# Integration tests requiring external dependencies
@pytest.mark.integration
class TestVectorStoreIntegration:
    """Integration tests for vector store with real FAISS."""

    def test_real_faiss_integration(self, mock_embedding_provider, sample_documents):
        """Test integration with real FAISS library."""
        pytest.importorskip("faiss")

        # Create store with real FAISS
        store = FAISSVectorStore(mock_embedding_provider)
        store.add_documents(sample_documents)

        # Test search
        results = store.search("AI framework", k=3)
        assert len(results) > 0

        # Test different index types
        for index_type in ["flat", "ip"]:
            store = FAISSVectorStore(mock_embedding_provider, index_type=index_type)
            store.add_documents(sample_documents[:2])
            results = store.search("test", k=1)
            assert len(results) > 0

    def test_large_document_set(self, mock_embedding_provider):
        """Test with larger document set to verify performance."""
        pytest.importorskip("faiss")

        # Create many documents
        large_doc_set = []
        for i in range(1000):
            doc = Document(
                content=f"Document {i} with content about topic {i % 10}",
                metadata={"id": i, "topic": f"topic_{i % 10}"},
                doc_id=f"doc_{i}"
            )
            large_doc_set.append(doc)

        # Add to store
        store = FAISSVectorStore(mock_embedding_provider)
        store.add_documents(large_doc_set)

        assert store.get_document_count() == 1000

        # Test search
        results = store.search("topic", k=10)
        assert len(results) == 10