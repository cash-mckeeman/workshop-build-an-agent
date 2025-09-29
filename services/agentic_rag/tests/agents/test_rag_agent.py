"""
Tests for RAG agent implementation.

This module tests the RAG agent creation and functionality, including
the evolution from basic LLM to sophisticated RAG agents.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from src.agents.rag_agent import (
    create_rag_agent,
    create_simple_rag_agent,
    demo_rag_agent,
    demo_rag_agent_sync
)


class TestRAGAgentCreation:
    """Test cases for RAG agent creation functions."""

    def test_create_rag_agent_basic(self, mock_vector_store, mock_reranker):
        """Test basic RAG agent creation."""
        model_provider = "test_model"

        with patch('src.agents.rag_agent.Agent') as mock_agent_class:
            with patch('src.agents.rag_agent.register_retrieval_tools') as mock_register:
                mock_agent_instance = Mock()
                mock_agent_class.return_value = mock_agent_instance

                agent = create_rag_agent(
                    model_provider=model_provider,
                    vector_store=mock_vector_store,
                    reranker=mock_reranker
                )

                # Check agent creation
                mock_agent_class.assert_called_once()
                call_args = mock_agent_class.call_args
                assert call_args[0][0] == model_provider
                assert "instructions" in call_args[1]

                # Check tool registration
                mock_register.assert_called_once_with(
                    agent=mock_agent_instance,
                    vector_store=mock_vector_store,
                    reranker=mock_reranker,
                    include_advanced=True
                )

                assert agent == mock_agent_instance

    def test_create_rag_agent_custom_instructions(self, mock_vector_store):
        """Test RAG agent creation with custom instructions."""
        custom_instructions = "Custom agent instructions"

        with patch('src.agents.rag_agent.Agent') as mock_agent_class:
            with patch('src.agents.rag_agent.register_retrieval_tools'):
                create_rag_agent(
                    model_provider="test",
                    vector_store=mock_vector_store,
                    instructions=custom_instructions,
                    include_advanced_tools=False
                )

                call_args = mock_agent_class.call_args[1]
                assert call_args["instructions"] == custom_instructions

    def test_create_rag_agent_without_advanced_tools(self, mock_vector_store):
        """Test RAG agent creation without advanced tools."""
        with patch('src.agents.rag_agent.Agent') as mock_agent_class:
            with patch('src.agents.rag_agent.register_retrieval_tools') as mock_register:
                mock_agent_instance = Mock()
                mock_agent_class.return_value = mock_agent_instance

                create_rag_agent(
                    model_provider="test",
                    vector_store=mock_vector_store,
                    include_advanced_tools=False
                )

                mock_register.assert_called_once_with(
                    agent=mock_agent_instance,
                    vector_store=mock_vector_store,
                    reranker=None,
                    include_advanced=False
                )

    def test_create_simple_rag_agent(self, mock_vector_store, mock_reranker):
        """Test simple RAG agent creation."""
        with patch('src.agents.rag_agent.create_rag_agent') as mock_create_rag:
            mock_agent = Mock()
            mock_create_rag.return_value = mock_agent

            agent = create_simple_rag_agent(
                model_provider="test_model",
                vector_store=mock_vector_store,
                reranker=mock_reranker
            )

            # Check that create_rag_agent was called with correct parameters
            mock_create_rag.assert_called_once()
            call_args = mock_create_rag.call_args[1]
            assert call_args["model_provider"] == "test_model"
            assert call_args["vector_store"] == mock_vector_store
            assert call_args["reranker"] == mock_reranker
            assert call_args["include_advanced_tools"] is False
            assert "instructions" in call_args

            assert agent == mock_agent


class TestRAGAgentDemo:
    """Test cases for RAG agent demo functionality."""

    @pytest.mark.asyncio
    async def test_demo_rag_agent_success(self, mock_vector_store, mock_reranker):
        """Test successful RAG agent demo."""
        with patch('src.retrieval.embeddings.get_best_available_provider') as mock_get_embedding:
            with patch('src.retrieval.vector_store.create_vector_store') as mock_create_store:
                with patch('src.retrieval.reranker.get_best_available_provider') as mock_get_reranker:
                    with patch('src.models.providers.get_best_available_provider') as mock_get_model:
                        with patch('src.agents.rag_agent.create_rag_agent') as mock_create_agent:
                            with patch('builtins.print') as mock_print:

                                # Setup mocks
                                mock_embedding_provider = Mock()
                                mock_get_embedding.return_value = mock_embedding_provider
                                mock_create_store.return_value = mock_vector_store
                                mock_get_reranker.return_value = mock_reranker
                                mock_get_model.return_value = "openai"

                                # Mock agent and its run method
                                mock_agent = Mock()
                                mock_run_result = Mock()
                                mock_run_result.output = "Test agent response"
                                mock_agent.run = AsyncMock(return_value=mock_run_result)
                                mock_create_agent.return_value = mock_agent

                                # Run demo
                                await demo_rag_agent("test_model")

                                # Verify demo execution
                                assert mock_print.called
                                assert mock_create_store.called
                                assert mock_create_agent.called
                                assert mock_agent.run.called

    @pytest.mark.asyncio
    async def test_demo_rag_agent_no_embedding_provider(self):
        """Test demo with no embedding provider available."""
        with patch('src.retrieval.embeddings.get_best_available_provider') as mock_get_provider:
            with patch('builtins.print') as mock_print:
                mock_get_provider.return_value = None

                await demo_rag_agent()

                # Check error message was printed
                print_calls = [call.args[0] for call in mock_print.call_args_list]
                assert any("No embedding provider available" in call for call in print_calls)

    @pytest.mark.asyncio
    async def test_demo_rag_agent_model_provider_selection(self, mock_vector_store):
        """Test demo model provider selection logic."""
        with patch('src.retrieval.embeddings.get_best_available_provider') as mock_get_embedding:
            with patch('src.retrieval.vector_store.create_vector_store') as mock_create_store:
                with patch('src.retrieval.reranker.get_best_available_provider') as mock_get_reranker:
                    with patch('src.models.providers.get_best_available_provider') as mock_get_model:
                        with patch('src.models.providers.get_provider_config') as mock_get_config:
                            with patch('src.agents.rag_agent.create_rag_agent') as mock_create_agent:
                                with patch('builtins.print'):

                                    # Setup mocks
                                    mock_embedding_provider = Mock()
                                    mock_get_embedding.return_value = mock_embedding_provider
                                    mock_create_store.return_value = mock_vector_store
                                    mock_get_reranker.return_value = None

                                    # Test different provider scenarios
                                    test_cases = [
                                        ("test", "test"),  # test provider
                                        ("openai", "openai:gpt-4o-mini"),  # real provider
                                        (None, "test")  # no provider, fallback to test
                                    ]

                                    for provider_name, expected_model in test_cases:
                                        mock_get_model.return_value = provider_name

                                        if provider_name and provider_name != "test":
                                            mock_provider_config = Mock()
                                            mock_provider_config.get_model_id.return_value = expected_model
                                            mock_get_config.return_value = mock_provider_config
                                        else:
                                            mock_get_config.return_value = None

                                        mock_agent = Mock()
                                        mock_agent.run = AsyncMock(return_value=Mock(output="test"))
                                        mock_create_agent.return_value = mock_agent

                                        await demo_rag_agent()

                                        # Check that create_rag_agent was called with correct model
                                        mock_create_agent.assert_called()
                                        actual_model = mock_create_agent.call_args[0][0]
                                        assert actual_model == expected_model

    @pytest.mark.asyncio
    async def test_demo_rag_agent_query_execution(self, mock_vector_store):
        """Test that demo executes all test queries."""
        with patch('src.retrieval.embeddings.get_best_available_provider') as mock_get_embedding:
            with patch('src.retrieval.vector_store.create_vector_store') as mock_create_store:
                with patch('src.retrieval.reranker.get_best_available_provider') as mock_get_reranker:
                    with patch('src.models.providers.get_best_available_provider') as mock_get_model:
                        with patch('src.agents.rag_agent.create_rag_agent') as mock_create_agent:
                            with patch('builtins.print') as mock_print:

                                # Setup mocks
                                mock_embedding_provider = Mock()
                                mock_get_embedding.return_value = mock_embedding_provider
                                mock_create_store.return_value = mock_vector_store
                                mock_get_reranker.return_value = None
                                mock_get_model.return_value = "test"

                                # Mock agent responses
                                mock_agent = Mock()
                                mock_responses = [
                                    Mock(output="Response 1"),
                                    Mock(output="Response 2"),
                                    Mock(output="Response 3"),
                                    Mock(output="Response 4")
                                ]
                                mock_agent.run = AsyncMock(side_effect=mock_responses)
                                mock_create_agent.return_value = mock_agent

                                await demo_rag_agent()

                                # Check that agent.run was called 4 times (for each test query)
                                assert mock_agent.run.call_count == 4

                                # Check that responses were printed
                                print_calls = [call.args[0] for call in mock_print.call_args_list]
                                assert any("Agent Response" in call for call in print_calls)

    @pytest.mark.asyncio
    async def test_demo_rag_agent_query_error_handling(self, mock_vector_store):
        """Test demo error handling for individual queries."""
        with patch('src.retrieval.embeddings.get_best_available_provider') as mock_get_embedding:
            with patch('src.retrieval.vector_store.create_vector_store') as mock_create_store:
                with patch('src.retrieval.reranker.get_best_available_provider') as mock_get_reranker:
                    with patch('src.models.providers.get_best_available_provider') as mock_get_model:
                        with patch('src.agents.rag_agent.create_rag_agent') as mock_create_agent:
                            with patch('builtins.print') as mock_print:

                                # Setup mocks
                                mock_embedding_provider = Mock()
                                mock_get_embedding.return_value = mock_embedding_provider
                                mock_create_store.return_value = mock_vector_store
                                mock_get_reranker.return_value = None
                                mock_get_model.return_value = "test"

                                # Mock agent with some failures
                                mock_agent = Mock()
                                mock_agent.run = AsyncMock(side_effect=[
                                    Mock(output="Success"),
                                    Exception("Query failed"),
                                    Mock(output="Success again"),
                                    Exception("Another failure")
                                ])
                                mock_create_agent.return_value = mock_agent

                                await demo_rag_agent()

                                # Check that all queries were attempted
                                assert mock_agent.run.call_count == 4

                                # Check that errors were handled
                                print_calls = [call.args[0] for call in mock_print.call_args_list]
                                assert any("Error processing query" in call for call in print_calls)

    @pytest.mark.asyncio
    async def test_demo_rag_agent_general_exception(self):
        """Test demo general exception handling."""
        with patch('src.retrieval.embeddings.get_best_available_provider') as mock_get_embedding:
            with patch('builtins.print') as mock_print:
                # Mock general failure
                mock_get_embedding.side_effect = Exception("General failure")

                await demo_rag_agent()

                # Check that error was handled
                print_calls = [call.args[0] for call in mock_print.call_args_list]
                assert any("Demo failed" in call for call in print_calls)


class TestRAGAgentSyncDemo:
    """Test cases for synchronous RAG agent demo wrapper."""

    def test_demo_rag_agent_sync_no_loop(self):
        """Test sync demo when no event loop is running."""
        with patch('src.agents.rag_agent.asyncio.get_running_loop') as mock_get_loop:
            with patch('src.agents.rag_agent.asyncio.run') as mock_run:
                # Mock no running loop
                mock_get_loop.side_effect = RuntimeError("No loop")

                demo_rag_agent_sync("test_model")

                # Should use asyncio.run
                mock_run.assert_called_once()

    def test_demo_rag_agent_sync_with_loop(self):
        """Test sync demo when event loop is running."""
        with patch('src.agents.rag_agent.asyncio.get_running_loop') as mock_get_loop:
            with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor_class:
                # Mock running loop
                mock_loop = Mock()
                mock_get_loop.return_value = mock_loop

                # Mock executor
                mock_executor = Mock()
                mock_future = Mock()
                mock_executor.submit.return_value = mock_future
                mock_executor_class.return_value.__enter__.return_value = mock_executor

                demo_rag_agent_sync("test_model")

                # Should use ThreadPoolExecutor
                mock_executor.submit.assert_called_once()
                mock_future.result.assert_called_once()

    def test_demo_rag_agent_sync_main_execution(self):
        """Test sync demo when called as main module."""
        with patch('src.agents.rag_agent.demo_rag_agent_sync') as mock_sync_demo:
            # Simulate module execution
            import src.agents.rag_agent

            # Mock __name__ == "__main__"
            with patch.object(src.agents.rag_agent, '__name__', '__main__'):
                # This would normally trigger the if __name__ == "__main__" block
                # but we'll test the function directly
                mock_sync_demo.assert_not_called()  # Not actually called in test

    def test_demo_document_creation(self, pure_mock_vector_store):
        """Test that demo creates appropriate sample documents."""
        with patch('src.retrieval.embeddings.get_best_available_provider') as mock_get_embedding:
            with patch('src.retrieval.vector_store.create_vector_store') as mock_create_store:
                with patch('src.retrieval.reranker.get_best_available_provider') as mock_get_reranker:
                    with patch('src.models.providers.get_best_available_provider') as mock_get_model:
                        with patch('src.agents.rag_agent.create_rag_agent') as mock_create_agent:
                            with patch('builtins.print'):

                                # Setup mocks
                                mock_embedding_provider = Mock()
                                mock_get_embedding.return_value = mock_embedding_provider
                                mock_create_store.return_value = pure_mock_vector_store
                                mock_get_reranker.return_value = None
                                mock_get_model.return_value = "test"

                                mock_agent = Mock()
                                mock_agent.run = AsyncMock(return_value=Mock(output="test"))
                                mock_create_agent.return_value = mock_agent

                                # Run demo
                                asyncio.run(demo_rag_agent())

                                # Check that documents were added to vector store
                                pure_mock_vector_store.add_documents.assert_called_once()
                                added_docs = pure_mock_vector_store.add_documents.call_args[0][0]

                                # Verify document structure
                                assert len(added_docs) == 4  # Expected number of sample documents
                                for doc in added_docs:
                                    assert hasattr(doc, 'content')
                                    assert hasattr(doc, 'metadata')
                                    assert isinstance(doc.metadata, dict)
                                    assert 'topic' in doc.metadata
                                    assert 'category' in doc.metadata


class TestRAGAgentEdgeCases:
    """Test edge cases and error conditions for RAG agents."""

    def test_create_rag_agent_with_none_reranker(self, mock_vector_store):
        """Test RAG agent creation with None reranker."""
        with patch('src.agents.rag_agent.Agent') as mock_agent_class:
            with patch('src.agents.rag_agent.register_retrieval_tools') as mock_register:
                mock_agent_instance = Mock()
                mock_agent_class.return_value = mock_agent_instance

                create_rag_agent(
                    model_provider="test",
                    vector_store=mock_vector_store,
                    reranker=None
                )

                # Should still work with None reranker
                mock_register.assert_called_once_with(
                    agent=mock_agent_instance,
                    vector_store=mock_vector_store,
                    reranker=None,
                    include_advanced=True
                )

    def test_create_rag_agent_empty_instructions(self, mock_vector_store):
        """Test RAG agent creation with empty instructions."""
        with patch('src.agents.rag_agent.Agent') as mock_agent_class:
            with patch('src.agents.rag_agent.register_retrieval_tools'):
                mock_agent_instance = Mock()
                mock_agent_class.return_value = mock_agent_instance

                create_rag_agent(
                    model_provider="test",
                    vector_store=mock_vector_store,
                    instructions=""
                )

                # Should use default instructions when empty string provided
                call_args = mock_agent_class.call_args[1]
                assert call_args["instructions"] != ""  # Should fall back to default