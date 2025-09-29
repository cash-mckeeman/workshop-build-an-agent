"""
Tests for RAG retrieval tools implementation.

This module tests the retrieval tools that agents use to search and retrieve
documents from the knowledge base.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.tools.retrieval_tools import (
    RAGRetrievalTools,
    RetrievalQuery,
    DocumentResult,
    RetrievalResult,
    register_retrieval_tools,
    create_retrieval_tools,
    demo_retrieval_tools
)
from src.retrieval.base import Document, SearchResult


class TestRetrievalModels:
    """Test cases for Pydantic models used in retrieval tools."""

    def test_retrieval_query_model(self):
        """Test RetrievalQuery model validation."""
        # Valid query
        query = RetrievalQuery(query="test query")
        assert query.query == "test query"
        assert query.max_results == 5  # default
        assert query.use_reranking is True  # default
        assert query.metadata_filter is None  # default

        # Query with all parameters
        query_full = RetrievalQuery(
            query="complex query",
            max_results=10,
            use_reranking=False,
            metadata_filter={"topic": "ai"}
        )
        assert query_full.max_results == 10
        assert query_full.use_reranking is False
        assert query_full.metadata_filter == {"topic": "ai"}

        # Test validation limits
        with pytest.raises(ValueError):
            RetrievalQuery(query="test", max_results=0)  # Below minimum

        with pytest.raises(ValueError):
            RetrievalQuery(query="test", max_results=25)  # Above maximum

    def test_document_result_model(self):
        """Test DocumentResult model."""
        doc_result = DocumentResult(
            content="Test content",
            score=0.95,
            rank=1,
            metadata={"topic": "test"},
            doc_id="test_doc"
        )

        assert doc_result.content == "Test content"
        assert doc_result.score == 0.95
        assert doc_result.rank == 1
        assert doc_result.metadata == {"topic": "test"}
        assert doc_result.doc_id == "test_doc"

    def test_retrieval_result_model(self):
        """Test RetrievalResult model."""
        doc_result = DocumentResult(
            content="Test content",
            score=0.95,
            rank=1,
            metadata={"topic": "test"},
            doc_id="test_doc"
        )

        result = RetrievalResult(
            query="test query",
            documents=[doc_result],
            total_results=1,
            reranked=True
        )

        assert result.query == "test query"
        assert len(result.documents) == 1
        assert result.total_results == 1
        assert result.reranked is True


class TestRAGRetrievalTools:
    """Test cases for RAGRetrievalTools class."""

    def test_tools_initialization(self, mock_vector_store, mock_reranker):
        """Test RAGRetrievalTools initialization."""
        tools = RAGRetrievalTools(mock_vector_store, mock_reranker)

        assert tools.vector_store == mock_vector_store
        assert tools.reranker == mock_reranker

        # Test without reranker
        tools_no_reranker = RAGRetrievalTools(mock_vector_store)
        assert tools_no_reranker.vector_store == mock_vector_store
        assert tools_no_reranker.reranker is None

    def test_search_knowledge_base(self, pure_mock_vector_store, sample_documents):
        """Test basic knowledge base search."""
        tools = RAGRetrievalTools(pure_mock_vector_store)

        # Mock search results
        search_results = [
            SearchResult(
                document=sample_documents[0],
                score=0.95,
                rank=1
            ),
            SearchResult(
                document=sample_documents[1],
                score=0.85,
                rank=2
            )
        ]
        pure_mock_vector_store.search.return_value = search_results

        # Test search
        result = tools.search_knowledge_base("AI framework")

        assert isinstance(result, RetrievalResult)
        assert result.query == "AI framework"
        assert len(result.documents) == 2
        assert result.total_results == 2
        assert result.reranked is False

        # Check document structure
        doc_result = result.documents[0]
        assert doc_result.content == sample_documents[0].content
        assert doc_result.score == 0.95
        assert doc_result.rank == 1

    def test_advanced_search(self, pure_mock_vector_store, mock_reranker, sample_documents):
        """Test advanced search functionality."""
        tools = RAGRetrievalTools(pure_mock_vector_store, mock_reranker)

        # Mock search results
        search_results = [
            SearchResult(document=sample_documents[0], score=0.95, rank=1),
            SearchResult(document=sample_documents[1], score=0.85, rank=2),
            SearchResult(document=sample_documents[2], score=0.75, rank=3)
        ]
        pure_mock_vector_store.search.return_value = search_results

        # Mock reranker results
        reranked_results = [
            SearchResult(document=sample_documents[1], score=0.98, rank=1),
            SearchResult(document=sample_documents[0], score=0.92, rank=2)
        ]
        mock_reranker.rerank.return_value = reranked_results

        # Test advanced search with reranking
        query = RetrievalQuery(
            query="test query",
            max_results=2,
            use_reranking=True,
            metadata_filter={"topic": "ai"}
        )

        result = tools.advanced_search(query)

        assert isinstance(result, RetrievalResult)
        assert result.query == "test query"
        assert len(result.documents) == 2
        assert result.reranked is True

        # Check that reranker was called
        mock_reranker.rerank.assert_called_once()

        # Check vector store search was called with correct parameters
        pure_mock_vector_store.search.assert_called_once_with(
            query="test query",
            k=4,  # max_results * 2 for reranking
            filter_metadata={"topic": "ai"}
        )

    def test_advanced_search_no_reranking(self, pure_mock_vector_store, sample_documents):
        """Test advanced search without reranking."""
        tools = RAGRetrievalTools(pure_mock_vector_store)

        search_results = [
            SearchResult(document=sample_documents[0], score=0.95, rank=1)
        ]
        pure_mock_vector_store.search.return_value = search_results

        query = RetrievalQuery(
            query="test query",
            max_results=3,
            use_reranking=False
        )

        result = tools.advanced_search(query)

        assert result.reranked is False
        assert len(result.documents) == 1

        # Check vector store search was called with correct k
        pure_mock_vector_store.search.assert_called_once_with(
            query="test query",
            k=3,  # same as max_results
            filter_metadata=None
        )

    def test_advanced_search_reranking_failure(self, pure_mock_vector_store, mock_reranker, sample_documents):
        """Test advanced search when reranking fails."""
        tools = RAGRetrievalTools(pure_mock_vector_store, mock_reranker)

        search_results = [
            SearchResult(document=sample_documents[0], score=0.95, rank=1)
        ]
        pure_mock_vector_store.search.return_value = search_results

        # Mock reranker failure
        mock_reranker.rerank.side_effect = Exception("Reranking failed")

        query = RetrievalQuery(query="test", use_reranking=True)
        result = tools.advanced_search(query)

        # Should fall back to original results
        assert result.reranked is False
        assert len(result.documents) == 1

    def test_advanced_search_error_handling(self, mock_vector_store):
        """Test advanced search error handling."""
        tools = RAGRetrievalTools(mock_vector_store)

        # Mock search failure
        mock_vector_store.search.side_effect = Exception("Search failed")

        query = RetrievalQuery(query="test")
        result = tools.advanced_search(query)

        # Should return empty result
        assert result.query == "test"
        assert len(result.documents) == 0
        assert result.total_results == 0
        assert result.reranked is False

    def test_get_document_by_id(self, mock_vector_store, sample_documents):
        """Test document retrieval by ID."""
        tools = RAGRetrievalTools(mock_vector_store)

        # Mock internal document storage
        mock_vector_store._documents = sample_documents

        # Test successful retrieval
        result = tools.get_document_by_id("doc_1")

        assert isinstance(result, DocumentResult)
        assert result.content == sample_documents[0].content
        assert result.doc_id == "doc_1"
        assert result.score == 1.0  # Perfect match for direct retrieval
        assert result.rank == 1

    def test_get_document_by_id_not_found(self, mock_vector_store, sample_documents):
        """Test document retrieval for non-existent ID."""
        tools = RAGRetrievalTools(mock_vector_store)
        mock_vector_store._documents = sample_documents

        result = tools.get_document_by_id("nonexistent")
        assert result is None

    def test_get_document_by_id_error(self, pure_mock_vector_store):
        """Test document retrieval error handling."""
        tools = RAGRetrievalTools(pure_mock_vector_store)

        # Mock error in document access - create a list that raises exception when iterated
        error_documents = Mock()
        error_documents.__iter__ = Mock(side_effect=Exception("Access error"))
        pure_mock_vector_store._documents = error_documents

        result = tools.get_document_by_id("doc_1")
        assert result is None

    def test_get_knowledge_base_info(self, mock_vector_store, sample_documents):
        """Test knowledge base information retrieval."""
        tools = RAGRetrievalTools(mock_vector_store)

        # Mock vector store info
        mock_vector_store.get_info.return_value = {
            "document_count": 5,
            "dimension": 384,
            "index_type": "flat"
        }

        # Mock metadata access
        mock_vector_store._metadata = [doc.metadata for doc in sample_documents]

        info = tools.get_knowledge_base_info()

        assert isinstance(info, dict)
        assert "document_count" in info
        assert "available_topics" in info
        assert "available_categories" in info
        assert "reranker_available" in info

        # Check extracted topics and categories
        expected_topics = {doc.metadata.get("topic") for doc in sample_documents if "topic" in doc.metadata}
        assert set(info["available_topics"]) == expected_topics

    def test_get_knowledge_base_info_error(self, mock_vector_store):
        """Test knowledge base info error handling."""
        tools = RAGRetrievalTools(mock_vector_store)

        # Mock error
        mock_vector_store.get_info.side_effect = Exception("Info error")

        info = tools.get_knowledge_base_info()
        assert "error" in info


class TestRetrievalToolsRegistration:
    """Test cases for tool registration with PydanticAI agents."""

    def test_register_basic_tools(self, mock_agent, mock_vector_store):
        """Test registering basic retrieval tools."""
        register_retrieval_tools(
            agent=mock_agent,
            vector_store=mock_vector_store,
            include_advanced=False
        )

        # Should register only basic search tool
        assert mock_agent.tool.call_count == 1

    def test_register_advanced_tools(self, mock_agent, mock_vector_store, mock_reranker):
        """Test registering advanced retrieval tools."""
        register_retrieval_tools(
            agent=mock_agent,
            vector_store=mock_vector_store,
            reranker=mock_reranker,
            include_advanced=True
        )

        # Should register all tools (basic + 3 advanced)
        assert mock_agent.tool.call_count == 4

    def test_tool_closures(self, mock_agent, mock_vector_store, mock_reranker):
        """Test that tool closures properly capture dependencies."""
        register_retrieval_tools(
            agent=mock_agent,
            vector_store=mock_vector_store,
            reranker=mock_reranker,
            include_advanced=True
        )

        # Get the registered tool functions
        tool_calls = mock_agent.tool.call_args_list
        registered_functions = [call[0][0] for call in tool_calls]

        # Test that functions can be called (basic smoke test)
        for func in registered_functions:
            assert callable(func)

    def test_create_retrieval_tools_factory(self, mock_vector_store, mock_reranker):
        """Test the factory function for creating retrieval tools."""
        tools = create_retrieval_tools(mock_vector_store, mock_reranker)

        assert isinstance(tools, RAGRetrievalTools)
        assert tools.vector_store == mock_vector_store
        assert tools.reranker == mock_reranker


class TestRetrievalToolsDemo:
    """Test cases for retrieval tools demo functionality."""

    @patch('src.retrieval.embeddings.get_best_available_provider')
    @patch('src.retrieval.vector_store.create_vector_store')
    @patch('src.retrieval.reranker.get_best_available_provider')
    @patch('builtins.print')
    def test_demo_retrieval_tools_success(
        self, mock_print, mock_get_reranker, mock_create_store,
        mock_get_provider, mock_embedding_provider, mock_vector_store, mock_reranker
    ):
        """Test successful retrieval tools demo."""
        # Setup mocks
        mock_get_provider.return_value = mock_embedding_provider
        mock_create_store.return_value = mock_vector_store
        mock_get_reranker.return_value = mock_reranker

        # Mock search results
        mock_result = RetrievalResult(
            query="AI frameworks for building agents",
            documents=[
                DocumentResult(
                    content="PydanticAI is a framework for building AI agents...",
                    score=0.95,
                    rank=1,
                    metadata={"topic": "ai"},
                    doc_id="doc_1"
                )
            ],
            total_results=1,
            reranked=False
        )

        # Mock tools behavior
        with patch.object(RAGRetrievalTools, 'search_knowledge_base', return_value=mock_result):
            with patch.object(RAGRetrievalTools, 'advanced_search', return_value=mock_result):
                with patch.object(RAGRetrievalTools, 'get_knowledge_base_info', return_value={"document_count": 3}):
                    demo_retrieval_tools()

        # Check that demo ran successfully
        assert mock_print.called
        print_calls = [call.args[0] for call in mock_print.call_args_list]
        assert any("RAG Retrieval Tools Demo" in call for call in print_calls)

    @patch('src.retrieval.embeddings.get_best_available_provider')
    @patch('builtins.print')
    def test_demo_no_provider(self, mock_print, mock_get_provider):
        """Test demo with no embedding provider available."""
        mock_get_provider.return_value = None

        demo_retrieval_tools()

        print_calls = [call.args[0] for call in mock_print.call_args_list]
        assert any("No embedding provider available" in call for call in print_calls)

    @patch('src.retrieval.embeddings.get_best_available_provider')
    @patch('builtins.print')
    def test_demo_exception_handling(self, mock_print, mock_get_provider, mock_embedding_provider):
        """Test demo exception handling."""
        mock_get_provider.return_value = mock_embedding_provider
        mock_embedding_provider.embed_texts.side_effect = Exception("Test error")

        demo_retrieval_tools()

        print_calls = [call.args[0] for call in mock_print.call_args_list]
        assert any("Error" in call for call in print_calls)


class TestRetrievalToolsIntegration:
    """Integration tests for retrieval tools."""

    def test_end_to_end_retrieval_flow(self, pure_mock_vector_store, mock_reranker, sample_documents):
        """Test complete retrieval flow from query to results."""
        # Setup mock search results
        search_results = [
            SearchResult(document=sample_documents[0], score=0.95, rank=1),
            SearchResult(document=sample_documents[1], score=0.85, rank=2)
        ]
        pure_mock_vector_store.search.return_value = search_results

        # Setup mock reranking
        reranked_results = [
            SearchResult(document=sample_documents[1], score=0.98, rank=1),
            SearchResult(document=sample_documents[0], score=0.92, rank=2)
        ]
        mock_reranker.rerank.return_value = reranked_results

        # Create tools and test flow
        tools = RAGRetrievalTools(pure_mock_vector_store, mock_reranker)

        # Test basic search
        basic_result = tools.search_knowledge_base("AI framework")
        assert len(basic_result.documents) == 2

        # Test advanced search with reranking
        advanced_query = RetrievalQuery(
            query="AI framework",
            max_results=2,
            use_reranking=True
        )
        advanced_result = tools.advanced_search(advanced_query)
        assert advanced_result.reranked is True
        assert len(advanced_result.documents) == 2

        # Test that reranking changed the order
        assert advanced_result.documents[0].content == sample_documents[1].content
        assert advanced_result.documents[1].content == sample_documents[0].content

    def test_agent_tool_integration(self, mock_agent, pure_mock_vector_store, sample_documents):
        """Test integration with PydanticAI agent tools."""
        # Register tools
        register_retrieval_tools(mock_agent, pure_mock_vector_store, include_advanced=True)

        # Verify tools were registered
        assert mock_agent.tool.call_count == 4

        # Get registered tool functions
        tool_calls = mock_agent.tool.call_args_list
        search_tool = tool_calls[0][0][0]  # First registered tool (search_knowledge_base)

        # Mock vector store behavior
        search_results = [
            SearchResult(document=sample_documents[0], score=0.95, rank=1)
        ]
        pure_mock_vector_store.search.return_value = search_results

        # Create mock run context
        mock_ctx = Mock()

        # Test tool execution
        result = search_tool(mock_ctx, "test query")

        assert isinstance(result, RetrievalResult)
        assert result.query == "test query"
        assert len(result.documents) == 1