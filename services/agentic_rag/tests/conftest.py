"""
Pytest configuration and fixtures for agentic_rag service tests.

This module provides shared fixtures and configuration for all tests in the
agentic_rag service, following the patterns from agent_intro service.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock

# Import base classes for mocking
from src.retrieval.base import (
    Document, SearchResult, EmbeddingProvider, RerankerProvider
)
from src.retrieval.vector_store import FAISSVectorStore


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    return [
        Document(
            content="PydanticAI is a modern framework for building AI agents with type safety and structured outputs.",
            metadata={"topic": "ai", "category": "framework", "source": "docs"},
            doc_id="doc_1"
        ),
        Document(
            content="Retrieval Augmented Generation (RAG) combines language models with external knowledge retrieval.",
            metadata={"topic": "rag", "category": "technique", "source": "guide"},
            doc_id="doc_2"
        ),
        Document(
            content="Vector databases like FAISS enable efficient similarity search over high-dimensional vectors.",
            metadata={"topic": "vector-db", "category": "database", "source": "technical"},
            doc_id="doc_3"
        ),
        Document(
            content="Cross-encoder reranking models improve retrieval quality by considering query-document pairs.",
            metadata={"topic": "reranking", "category": "technique", "source": "research"},
            doc_id="doc_4"
        ),
        Document(
            content="Python is excellent for machine learning and data science applications.",
            metadata={"topic": "programming", "category": "language", "source": "tutorial"},
            doc_id="doc_5"
        )
    ]


@pytest.fixture
def mock_embedding_provider():
    """Mock embedding provider for testing."""
    provider = Mock(spec=EmbeddingProvider)
    provider.model_name = "test-embedding-model"
    provider.dimension = 384

    # Mock embedding generation
    def mock_embed_text(text: str) -> List[float]:
        # Simple hash-based embedding for testing
        import hashlib
        hash_value = hashlib.md5(text.encode()).hexdigest()
        # Convert to list of floats (384 dimensions)
        hash_values = [float(int(hash_value[i:i+2], 16)) / 255.0 for i in range(0, len(hash_value), 2)]
        # Pad or repeat to get exactly 384 dimensions
        while len(hash_values) < 384:
            hash_values.extend(hash_values[:min(len(hash_values), 384 - len(hash_values))])
        return hash_values[:384]

    def mock_embed_texts(texts: List[str]) -> List[List[float]]:
        return [mock_embed_text(text) for text in texts]

    provider.embed_text.side_effect = mock_embed_text
    provider.embed_texts.side_effect = mock_embed_texts

    return provider


@pytest.fixture
def mock_reranker():
    """Mock reranker for testing."""
    reranker = Mock(spec=RerankerProvider)
    reranker.model_name = "test-reranker"

    def mock_rerank(query: str, documents: List[Document], top_k: int = 10) -> List[SearchResult]:
        # Simple mock reranking: reverse order with different scores
        results = []
        for i, doc in enumerate(reversed(documents[:top_k])):
            score = 0.9 - (i * 0.1)  # Decreasing scores
            results.append(SearchResult(
                document=doc,
                score=score,
                rank=i + 1
            ))
        return results

    reranker.rerank.side_effect = mock_rerank
    return reranker


@pytest.fixture
def mock_vector_store(mock_embedding_provider, sample_documents):
    """Mock vector store with sample data that doesn't depend on FAISS being installed."""
    from unittest.mock import Mock
    import numpy as np

    # Create a fully mocked vector store that behaves like FAISSVectorStore
    vector_store = Mock()

    # Setup basic attributes that tests expect
    vector_store.embedding_provider = mock_embedding_provider
    vector_store.index_type = "flat"
    vector_store._documents = sample_documents.copy()
    vector_store._metadata = [doc.metadata for doc in sample_documents]
    vector_store._document_lookup = {doc.doc_id: i for i, doc in enumerate(sample_documents) if doc.doc_id}

    # Mock methods
    vector_store.get_document_count.return_value = len(sample_documents)

    def mock_search(query, k=10, filter_metadata=None):
        # Simple mock search that returns filtered results
        results = []
        for i, doc in enumerate(sample_documents[:k]):
            if filter_metadata:
                # Check if document matches filter
                if not all(doc.metadata.get(key) == value for key, value in filter_metadata.items()):
                    continue

            from src.retrieval.base import SearchResult
            result = SearchResult(
                document=doc,
                score=0.9 - (i * 0.1),
                rank=i + 1
            )
            results.append(result)
        return results

    def mock_search_by_vector(vector, k=10, filter_metadata=None):
        return mock_search("", k, filter_metadata)

    def mock_delete_documents(doc_ids):
        # Remove documents from mock store
        new_docs = []
        new_metadata = []
        new_lookup = {}

        for i, doc in enumerate(vector_store._documents):
            if doc.doc_id not in doc_ids:
                new_idx = len(new_docs)
                new_docs.append(doc)
                new_metadata.append(vector_store._metadata[i])
                if doc.doc_id:
                    new_lookup[doc.doc_id] = new_idx

        vector_store._documents = new_docs
        vector_store._metadata = new_metadata
        vector_store._document_lookup = new_lookup
        vector_store.get_document_count.return_value = len(new_docs)

    def mock_clear():
        vector_store._documents.clear()
        vector_store._metadata.clear()
        vector_store._document_lookup.clear()
        vector_store.get_document_count.return_value = 0

    def mock_get_info():
        return {
            "index_type": "flat",
            "dimension": 384,
            "document_count": len(vector_store._documents),
            "index_size": len(vector_store._documents),
            "embedding_provider": {
                "class": mock_embedding_provider.__class__.__name__,
                "model_name": mock_embedding_provider.model_name
            },
            "is_trained": True
        }

    def mock_matches_filter(metadata, filter_metadata):
        """Check if metadata matches the filter criteria."""
        for key, value in filter_metadata.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True

    # Assign mock methods
    vector_store.search.side_effect = mock_search
    vector_store.search_by_vector.side_effect = mock_search_by_vector
    vector_store.delete_documents.side_effect = mock_delete_documents
    vector_store.clear.side_effect = mock_clear
    vector_store.get_info.side_effect = mock_get_info
    vector_store._matches_filter.side_effect = mock_matches_filter

    # Mock other methods that tests might use
    vector_store.add_documents = Mock()
    vector_store.save = Mock()
    vector_store.load = Mock()

    return vector_store


@pytest.fixture
def pure_mock_vector_store():
    """Pure mock vector store for testing tools that need mocked behavior."""
    mock_store = Mock()
    mock_store.get_document_count.return_value = 5
    mock_store.search.return_value = []
    mock_store.add_documents.return_value = None
    mock_store.get_info.return_value = {
        "document_count": 5,
        "dimension": 384,
        "index_type": "flat"
    }
    mock_store._documents = []
    mock_store._metadata = []
    return mock_store


@pytest.fixture
def mock_agent():
    """Mock PydanticAI agent for testing."""
    agent = Mock()
    agent.run = Mock()
    agent.tool = Mock()

    # Mock run method to return a result
    mock_result = Mock()
    mock_result.output = "Test agent response"
    agent.run.return_value = mock_result

    return agent


@pytest.fixture
def sample_retrieval_result():
    """Sample retrieval result for testing."""
    from src.tools.retrieval_tools import RetrievalResult, DocumentResult

    return RetrievalResult(
        query="test query",
        documents=[
            DocumentResult(
                content="Test document content",
                score=0.95,
                rank=1,
                metadata={"topic": "test"},
                doc_id="test_doc_1"
            )
        ],
        total_results=1,
        reranked=False
    )


@pytest.fixture
def api_key_env(monkeypatch):
    """Set test API keys in environment."""
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test_anthropic_key")
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "test_hf_key")


@pytest.fixture
def no_api_keys(monkeypatch):
    """Remove API keys from environment."""
    for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "HUGGINGFACE_API_KEY", "GROQ_API_KEY"]:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def mock_streamlit():
    """Mock Streamlit components for testing."""
    mock_st = Mock()

    # Mock common Streamlit components
    mock_st.title = Mock()
    mock_st.write = Mock()
    mock_st.text_input = Mock(return_value="test input")
    mock_st.button = Mock(return_value=False)
    mock_st.selectbox = Mock(return_value="option1")
    mock_st.sidebar = Mock()
    mock_st.columns = Mock(return_value=[Mock(), Mock()])
    mock_st.expander = Mock()
    mock_st.container = Mock()
    mock_st.empty = Mock()
    mock_st.success = Mock()
    mock_st.error = Mock()
    mock_st.warning = Mock()
    mock_st.info = Mock()

    # Mock session state
    mock_st.session_state = {}

    # Mock file uploader
    mock_st.file_uploader = Mock(return_value=None)

    return mock_st


@pytest.fixture
def mock_knowledge_files(temp_dir):
    """Create mock knowledge base files for testing."""
    knowledge_dir = Path(temp_dir) / "knowledge"
    knowledge_dir.mkdir()

    # Create sample files
    files = {
        "python_basics.md": """# Python Basics
Python is a high-level programming language known for its simplicity and readability.

## Features
- Easy to learn syntax
- Extensive standard library
- Cross-platform compatibility
""",
        "machine_learning.md": """# Machine Learning
Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience.

## Types
- Supervised learning
- Unsupervised learning
- Reinforcement learning
""",
        "data_structures.txt": """Data Structures Overview

Arrays: Fixed-size sequential collection of elements
Lists: Dynamic arrays that can grow or shrink
Dictionaries: Key-value pair collections
Sets: Unordered collections of unique elements
""",
        "algorithms.pdf": "Mock PDF content for algorithms"  # Simplified for testing
    }

    for filename, content in files.items():
        file_path = knowledge_dir / filename
        file_path.write_text(content)

    return knowledge_dir


# Test utilities
class MockAPIResponse:
    """Mock API response for testing external API calls."""

    def __init__(self, data: Dict[str, Any], status_code: int = 200):
        self.data = data
        self.status_code = status_code

    def json(self):
        return self.data

    @property
    def ok(self):
        return 200 <= self.status_code < 300


def create_mock_search_results(query: str, count: int = 3) -> List[SearchResult]:
    """Create mock search results for testing."""
    results = []
    for i in range(count):
        doc = Document(
            content=f"Mock document {i+1} for query: {query}",
            metadata={"rank": i+1, "query": query},
            doc_id=f"mock_doc_{i+1}"
        )
        result = SearchResult(
            document=doc,
            score=0.9 - (i * 0.1),
            rank=i + 1
        )
        results.append(result)
    return results


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test requiring external dependencies"
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers",
        "api: mark test as requiring API keys"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip certain tests in CI."""
    import os

    # Skip API tests if no API keys available
    skip_api = pytest.mark.skip(reason="API keys not available")

    for item in items:
        if "api" in item.keywords:
            if not any(os.getenv(key) for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "HUGGINGFACE_API_KEY"]):
                item.add_marker(skip_api)