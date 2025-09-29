"""
RAG Agent Implementation using PydanticAI

This module provides an agentic RAG implementation that demonstrates the evolution
from basic LLM inference to sophisticated retrieval-augmented generation with
decision-making capabilities.
"""

import asyncio
from typing import Optional
import logging

from pydantic_ai import Agent

from ..retrieval.vector_store import FAISSVectorStore
from ..retrieval.reranker import RerankerProvider
from ..tools.retrieval_tools import register_retrieval_tools

logger = logging.getLogger(__name__)


def create_rag_agent(
    model_provider: str,
    vector_store: FAISSVectorStore,
    reranker: Optional[RerankerProvider] = None,
    instructions: Optional[str] = None,
    include_advanced_tools: bool = True
) -> Agent[None, str]:
    """Create a RAG agent with retrieval capabilities.

    This demonstrates the evolution from basic agents to sophisticated RAG agents
    that can decide when and how to retrieve information to answer questions.

    Args:
        model_provider: Model identifier (e.g., "openai:gpt-4o-mini")
        vector_store: Vector store for document retrieval
        reranker: Optional reranker for improving retrieval quality
        instructions: Custom instructions for the agent
        include_advanced_tools: Whether to include advanced retrieval tools

    Returns:
        A configured PydanticAI RAG agent
    """
    default_instructions = """You are an intelligent RAG (Retrieval Augmented Generation) agent.

Your role is to help users by:
1. Understanding their questions and information needs
2. Deciding when to search for relevant information using your retrieval tools
3. Analyzing retrieved documents to find the most relevant information
4. Providing comprehensive, accurate answers based on the retrieved knowledge

Key guidelines:
- Always search the knowledge base when users ask questions that require specific information
- Use the search_knowledge_base tool for most queries
- Use advanced_search when you need more control (filtering, specific result counts)
- Cite your sources by mentioning relevant details from the retrieved documents
- If no relevant information is found, clearly state this and provide general guidance
- Be conversational and helpful while being accurate and factual

Remember: You have access to a knowledge base through your search tools. Use them actively to provide the best possible answers."""

    # Create the agent
    agent = Agent(
        model_provider,
        instructions=instructions or default_instructions
    )

    # Register retrieval tools
    register_retrieval_tools(
        agent=agent,
        vector_store=vector_store,
        reranker=reranker,
        include_advanced=include_advanced_tools
    )

    logger.info(f"Created RAG agent with {'advanced' if include_advanced_tools else 'basic'} tools")
    return agent


def create_simple_rag_agent(
    model_provider: str,
    vector_store: FAISSVectorStore,
    reranker: Optional[RerankerProvider] = None
) -> Agent[None, str]:
    """Create a simple RAG agent with basic retrieval capabilities.

    This is a streamlined version that includes only the essential search tool,
    perfect for demonstrations and basic use cases.

    Args:
        model_provider: Model identifier
        vector_store: Vector store for document retrieval
        reranker: Optional reranker for improving retrieval quality

    Returns:
        A configured simple RAG agent
    """
    instructions = """You are a helpful assistant with access to a knowledge base.

When users ask questions:
1. Search the knowledge base using your search_knowledge_base tool
2. Use the retrieved information to provide accurate, helpful answers
3. Always mention when your answer is based on the retrieved documents
4. If you can't find relevant information, say so clearly

Be conversational, helpful, and accurate in your responses."""

    return create_rag_agent(
        model_provider=model_provider,
        vector_store=vector_store,
        reranker=reranker,
        instructions=instructions,
        include_advanced_tools=False
    )


async def demo_rag_agent(model_provider: Optional[str] = None):
    """Demonstrate the RAG agent functionality.

    This shows the complete RAG workflow:
    1. Create vector store and load knowledge
    2. Create RAG agent with retrieval tools
    3. Ask questions that require knowledge retrieval
    4. Show how the agent uses tools to find and use information
    """
    print("🤖 RAG Agent Demo")
    print("=" * 40)

    try:
        # Create embedding provider
        from ..retrieval.embeddings import get_best_available_provider
        from ..retrieval.vector_store import create_vector_store
        from ..retrieval.reranker import get_best_available_provider as get_best_reranker
        from ..retrieval.base import Document

        embedding_provider = get_best_available_provider()
        if not embedding_provider:
            print("❌ No embedding provider available")
            return

        # Create vector store with sample knowledge
        vector_store = create_vector_store(embedding_provider)
        reranker = get_best_reranker()

        # Add sample knowledge base
        documents = [
            Document(
                content="""PydanticAI is a Python framework for building AI agents with type safety and structured outputs.
                It provides tools for creating agents that can use external tools, maintain state, and handle complex workflows.
                Key features include automatic type validation, tool registration, and integration with various LLM providers.""",
                metadata={"topic": "pydantic-ai", "category": "framework", "source": "documentation"}
            ),
            Document(
                content="""Retrieval Augmented Generation (RAG) is a technique that combines language models with external knowledge retrieval.
                The process involves: 1) Converting documents into vector embeddings, 2) Storing them in a vector database,
                3) Retrieving relevant documents for a query, 4) Using retrieved context to generate better responses.""",
                metadata={"topic": "rag", "category": "technique", "source": "guide"}
            ),
            Document(
                content="""Vector databases like FAISS enable efficient similarity search over high-dimensional vectors.
                They support various indexing methods including flat, IVF, and HNSW for different performance characteristics.
                FAISS is particularly good for exact and approximate nearest neighbor search in machine learning applications.""",
                metadata={"topic": "vector-db", "category": "database", "source": "technical"}
            ),
            Document(
                content="""Cross-encoder reranking models improve retrieval quality by scoring query-document pairs directly.
                Unlike bi-encoders that create separate embeddings, cross-encoders consider query and document together.
                Models like ms-marco-MiniLM-L-6-v2 provide good performance for reranking retrieved documents.""",
                metadata={"topic": "reranking", "category": "technique", "source": "research"}
            )
        ]

        vector_store.add_documents(documents)
        print(f"📚 Loaded {len(documents)} documents into knowledge base")

        # Create RAG agent
        if not model_provider:
            # Try to get best available model
            from ..models.providers import get_best_available_provider as get_best_model
            provider_name = get_best_model()
            if provider_name == "test":
                model_provider = "test"
            elif provider_name:
                from ..models.providers import get_provider_config
                provider_config = get_provider_config(provider_name)
                model_provider = provider_config.get_model_id()
            else:
                model_provider = "test"  # fallback to test

        agent = create_rag_agent(model_provider, vector_store, reranker)
        print(f"🤖 Created RAG agent with model: {model_provider}")
        print(f"🔄 Reranker: {'available' if reranker else 'not available'}")

        # Test queries
        test_queries = [
            "What is PydanticAI and what are its key features?",
            "How does RAG work? Explain the process step by step.",
            "What's the difference between bi-encoders and cross-encoders for reranking?",
            "Tell me about vector databases and FAISS indexing methods."
        ]

        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*50}")
            print(f"Query {i}: {query}")
            print("=" * 50)

            try:
                result = await agent.run(query)
                print(f"🔍 Agent Response:")
                print(result.output)

            except Exception as e:
                print(f"❌ Error processing query: {str(e)}")
                continue

        print(f"\n{'='*40}")
        print("✅ RAG Agent Demo Complete")

    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        logger.exception("RAG agent demo failed")


def demo_rag_agent_sync(model_provider: Optional[str] = None):
    """Synchronous wrapper for the RAG agent demo."""
    try:
        # Check if we're already in an event loop
        loop = asyncio.get_running_loop()
        # If we're in a loop, we need to use a different approach
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, demo_rag_agent(model_provider))
            future.result()
    except RuntimeError:
        # No event loop running, safe to use asyncio.run
        asyncio.run(demo_rag_agent(model_provider))


if __name__ == "__main__":
    demo_rag_agent_sync()