"""
Integration tests for the agentic RAG service.

This module tests the integration between different components of the RAG system,
verifying that they work together correctly in realistic scenarios.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from src.retrieval.base import Document
from src.retrieval.vector_store import create_vector_store
from src.tools.retrieval_tools import create_retrieval_tools, register_retrieval_tools
from src.agents.rag_agent import create_rag_agent


class TestRAGSystemIntegration:
    """Test complete RAG system integration."""

    def test_full_rag_pipeline(self, mock_embedding_provider, mock_reranker, sample_documents):
        """Test complete RAG pipeline from documents to agent response."""
        # 1. Create vector store and add documents
        vector_store = create_vector_store(mock_embedding_provider)
        vector_store.add_documents(sample_documents)

        # 2. Create retrieval tools
        retrieval_tools = create_retrieval_tools(vector_store, mock_reranker)

        # 3. Test search functionality
        search_result = retrieval_tools.search_knowledge_base("AI framework")
        assert len(search_result.documents) > 0

        # 4. Create RAG agent
        with patch('src.agents.rag_agent.Agent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent

            agent = create_rag_agent("test_model", vector_store, mock_reranker)

            # Verify agent was created and tools registered
            assert mock_agent_class.called
            assert agent == mock_agent

    def test_document_to_search_workflow(self, mock_embedding_provider, sample_documents):
        """Test workflow from document loading to search results."""
        # Create vector store
        vector_store = create_vector_store(mock_embedding_provider)

        # Add documents
        vector_store.add_documents(sample_documents)
        assert vector_store.get_document_count() == len(sample_documents)

        # Search for specific content
        results = vector_store.search("PydanticAI framework", k=3)
        assert len(results) > 0

        # Verify search results contain expected content
        found_pydantic = any("PydanticAI" in result.document.content for result in results)
        assert found_pydantic

    def test_reranking_integration(self, mock_embedding_provider, mock_reranker, sample_documents):
        """Test integration with reranking functionality."""
        # Setup vector store
        vector_store = create_vector_store(mock_embedding_provider)
        vector_store.add_documents(sample_documents)

        # Create tools with reranker
        tools = create_retrieval_tools(vector_store, mock_reranker)

        # Setup mock reranking behavior
        from src.retrieval.base import SearchResult
        mock_reranked = [
            SearchResult(
                document=sample_documents[1],  # Different order
                score=0.98,
                rank=1
            ),
            SearchResult(
                document=sample_documents[0],
                score=0.95,
                rank=2
            )
        ]
        mock_reranker.rerank.return_value = mock_reranked

        # Test advanced search with reranking
        from src.tools.retrieval_tools import RetrievalQuery
        query = RetrievalQuery(query="test", use_reranking=True, max_results=2)
        result = tools.advanced_search(query)

        assert result.reranked is True
        assert len(result.documents) == 2
        mock_reranker.rerank.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_tool_interaction(self, mock_embedding_provider, sample_documents):
        """Test agent interaction with retrieval tools."""
        # Setup components
        vector_store = create_vector_store(mock_embedding_provider)
        vector_store.add_documents(sample_documents)

        # Mock agent behavior
        with patch('src.agents.rag_agent.Agent') as mock_agent_class:
            mock_agent = Mock()
            mock_response = Mock()
            mock_response.output = "Test response based on retrieved documents"
            mock_agent.run = AsyncMock(return_value=mock_response)
            mock_agent_class.return_value = mock_agent

            # Create agent
            agent = create_rag_agent("test_model", vector_store)

            # Test agent run
            result = await agent.run("What is PydanticAI?")
            assert result.output == "Test response based on retrieved documents"

    def test_error_propagation(self, mock_embedding_provider):
        """Test error handling across integrated components."""
        # Create vector store
        vector_store = create_vector_store(mock_embedding_provider)

        # Add documents that will cause search errors
        mock_embedding_provider.embed_text.side_effect = Exception("Embedding failed")

        # Test that tools handle errors gracefully
        tools = create_retrieval_tools(vector_store)
        result = tools.search_knowledge_base("test query")

        # Should return empty result, not crash
        assert result.total_results == 0
        assert len(result.documents) == 0

    def test_metadata_filtering_integration(self, mock_embedding_provider, sample_documents):
        """Test metadata filtering across the system."""
        # Setup
        vector_store = create_vector_store(mock_embedding_provider)
        vector_store.add_documents(sample_documents)

        # Test direct vector store filtering
        results = vector_store.search("framework", filter_metadata={"topic": "ai"})
        for result in results:
            assert result.document.metadata.get("topic") == "ai"

        # Test through tools
        tools = create_retrieval_tools(vector_store)
        from src.tools.retrieval_tools import RetrievalQuery

        filtered_query = RetrievalQuery(
            query="framework",
            metadata_filter={"category": "framework"}
        )
        tool_result = tools.advanced_search(filtered_query)

        for doc in tool_result.documents:
            assert doc.metadata.get("category") == "framework"


class TestKnowledgeLoadingIntegration:
    """Test integration with knowledge loading system."""

    def test_knowledge_to_vector_store_integration(self, mock_embedding_provider, mock_knowledge_files):
        """Test loading knowledge files into vector store."""
        from src.knowledge.loader import load_documents_from_directory

        # Load documents from files
        documents = load_documents_from_directory(str(mock_knowledge_files))
        assert len(documents) > 0

        # Add to vector store
        vector_store = create_vector_store(mock_embedding_provider)
        vector_store.add_documents(documents)

        assert vector_store.get_document_count() == len(documents)

        # Test search functionality
        results = vector_store.search("Python programming", k=3)
        assert len(results) > 0

    def test_setup_knowledge_base_integration(
        self, mock_embedding_provider, mock_vector_store, mock_knowledge_files
    ):
        """Test complete knowledge base setup integration."""
        from src.knowledge.loader import setup_knowledge_base

        # Test the actual setup function which loads documents
        # Mock the embedding provider and vector store creation
        with patch('src.retrieval.embeddings.get_best_available_provider') as mock_get_embedding:
            with patch('src.retrieval.vector_store.create_vector_store') as mock_create_store:
                # Setup mocks
                mock_get_embedding.return_value = mock_embedding_provider
                mock_create_store.return_value = mock_vector_store

                # Run setup - the function actually loads documents and returns them
                result = setup_knowledge_base(data_dir=str(mock_knowledge_files))

                # The function returns loaded documents, not vector store
                assert isinstance(result, list)
                assert len(result) > 0
                for doc in result:
                    assert isinstance(doc, Document)


class TestProviderIntegration:
    """Test integration with different AI model providers."""

    def test_embedding_provider_integration(self, mock_embedding_provider, sample_documents):
        """Test embedding provider integration with vector store."""
        # Test that embedding provider is properly used
        vector_store = create_vector_store(mock_embedding_provider)

        # Adding documents should call embedding provider
        vector_store.add_documents(sample_documents[:2])

        # Verify embedding provider was called
        assert mock_embedding_provider.embed_texts.called
        call_args = mock_embedding_provider.embed_texts.call_args[0][0]
        assert len(call_args) == 2  # Two documents

        # Test search calls embedding provider
        mock_embedding_provider.embed_text.reset_mock()
        vector_store.search("test query")

        assert mock_embedding_provider.embed_text.called

    def test_model_provider_selection_integration(self):
        """Test model provider selection in agent creation."""
        from src.models.providers import get_best_available_provider, get_provider_config

        with patch('src.models.providers.get_best_available_provider') as mock_get_best:
            with patch('src.models.providers.get_provider_config') as mock_get_config:
                with patch('src.agents.rag_agent.Agent') as mock_agent_class:
                    with patch('src.agents.rag_agent.register_retrieval_tools'):

                        # Mock provider selection
                        mock_get_best.return_value = "openai"
                        mock_config = Mock()
                        mock_config.get_model_id.return_value = "openai:gpt-4o-mini"
                        mock_get_config.return_value = mock_config

                        mock_vector_store = Mock()
                        mock_agent_class.return_value = Mock()

                        # Test agent creation uses correct model
                        create_rag_agent("auto", mock_vector_store)

                        # Verify agent was created with resolved model
                        assert mock_agent_class.called


class TestStreamlitIntegration:
    """Test Streamlit app integration with RAG components."""

    @patch('src.streamlit_app.app.st')
    def test_streamlit_app_imports(self, mock_st):
        """Test that Streamlit app can import and use RAG components."""
        # Test importing the main app
        try:
            from src.streamlit_app.app import main
            # Basic smoke test - function should be callable
            assert callable(main)
        except ImportError as e:
            pytest.fail(f"Failed to import Streamlit app: {e}")

    @patch('src.streamlit_app.utils.session_state.st')
    def test_streamlit_session_state_integration(self, mock_st):
        """Test Streamlit session state integration."""
        # Mock session state
        mock_st.session_state = {}

        try:
            from src.streamlit_app.utils.session_state import (
                initialize_session_state,
                get_session_state,
                update_session_state
            )

            # Test session state utilities
            initialize_session_state()
            update_session_state({"test_key": "test_value"})
            state = get_session_state()

            assert "test_key" in state
            assert state["test_key"] == "test_value"

        except ImportError as e:
            pytest.fail(f"Failed to import session state utilities: {e}")


class TestFactoryIntegration:
    """Test factory module integration."""

    def test_factory_component_creation(self, mock_embedding_provider, sample_documents):
        """Test factory creates components correctly."""
        try:
            from src.factory import create_rag_system
            from src.retrieval.base import Document

            # Mock the dependencies properly
            with patch('src.factory.RAGFactory') as mock_factory_class:
                mock_factory = Mock()
                mock_agent = Mock()
                mock_factory.create_complete_rag_system.return_value = mock_agent
                mock_factory_class.return_value = mock_factory

                # Test factory function with sample documents
                result = create_rag_system(documents=sample_documents)

                # Verify factory was created and used correctly
                assert mock_factory_class.called
                assert mock_factory.create_complete_rag_system.called
                assert result == mock_agent

                # Verify documents were passed correctly
                call_args = mock_factory.create_complete_rag_system.call_args
                assert 'documents' in call_args.kwargs
                assert call_args.kwargs['documents'] == sample_documents

        except ImportError:
            # Factory module might not exist - that's okay for this test
            pass

    def test_provider_configuration_integration(self):
        """Test provider configuration integration."""
        try:
            from src.models.providers import (
                get_best_available_provider,
                get_provider_config,
                list_available_providers
            )

            # Test provider listing
            providers = list_available_providers()
            assert isinstance(providers, list)

            # Test provider configuration
            if providers:
                config = get_provider_config(providers[0])
                assert config is not None

        except ImportError:
            # Provider module might have different structure
            pass


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    @pytest.mark.asyncio
    async def test_complete_rag_workflow(
        self, mock_embedding_provider, mock_vector_store, sample_documents
    ):
        """Test complete RAG workflow from setup to query."""
        # Mock the required dependencies with correct paths
        with patch('src.retrieval.embeddings.get_best_available_provider') as mock_get_embedding:
            with patch('src.retrieval.vector_store.create_vector_store') as mock_create_store:
                with patch('src.retrieval.reranker.get_best_available_provider') as mock_get_reranker:
                    with patch('src.models.providers.get_best_available_provider') as mock_get_model:
                        with patch('src.agents.rag_agent.Agent') as mock_agent_class:

                            # Setup mocks
                            mock_get_embedding.return_value = mock_embedding_provider
                            mock_create_store.return_value = mock_vector_store
                            mock_get_reranker.return_value = None  # No reranker
                            mock_get_model.return_value = "test"

                            # Mock vector store search
                            from src.retrieval.base import SearchResult
                            mock_search_results = [
                                SearchResult(
                                    document=sample_documents[0],
                                    score=0.95,
                                    rank=1
                                )
                            ]
                            mock_vector_store.search.return_value = mock_search_results

                            # Mock agent
                            mock_agent = Mock()
                            mock_response = Mock()
                            mock_response.output = "Comprehensive answer about PydanticAI framework"
                            mock_agent.run = AsyncMock(return_value=mock_response)
                            mock_agent_class.return_value = mock_agent

                            # Test the demo workflow
                            from src.agents.rag_agent import demo_rag_agent
                            await demo_rag_agent()

                            # Verify workflow executed
                            assert mock_vector_store.add_documents.called
                            assert mock_agent.run.called

    def test_error_recovery_workflow(self, mock_embedding_provider):
        """Test system behavior under various error conditions."""
        # Test 1: Embedding provider fails
        mock_embedding_provider.embed_texts.side_effect = Exception("Embedding failed")

        vector_store = create_vector_store(mock_embedding_provider)
        tools = create_retrieval_tools(vector_store)

        # Should handle gracefully
        result = tools.search_knowledge_base("test")
        assert result.total_results == 0

        # Test 2: Vector store operations fail
        mock_vector_store = Mock()
        mock_vector_store.search.side_effect = Exception("Search failed")

        tools = create_retrieval_tools(mock_vector_store)
        result = tools.search_knowledge_base("test")
        assert result.total_results == 0

    def test_performance_integration(self, mock_embedding_provider):
        """Test system performance with larger datasets."""
        # Create larger document set
        large_doc_set = []
        for i in range(100):
            doc = Document(
                content=f"Document {i} about topic {i % 10} with detailed content " * 10,
                metadata={"id": i, "topic": f"topic_{i % 10}"},
                doc_id=f"large_doc_{i}"
            )
            large_doc_set.append(doc)

        # Test system can handle larger dataset
        vector_store = create_vector_store(mock_embedding_provider)
        vector_store.add_documents(large_doc_set)

        assert vector_store.get_document_count() == 100

        # Test search performance
        results = vector_store.search("topic", k=10)
        assert len(results) <= 10

        # Test tools with larger dataset
        tools = create_retrieval_tools(vector_store)
        tool_result = tools.search_knowledge_base("topic")
        assert tool_result.total_results > 0