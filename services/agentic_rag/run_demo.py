"""
Demo script for the Agentic RAG service.

This script demonstrates the complete functionality of the agentic RAG system,
from basic embedding and retrieval to sophisticated agent-based question answering.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.factory import RAGFactory, AgentType
from src.retrieval.base import Document


def create_sample_knowledge_base():
    """Create a sample knowledge base for demonstration."""
    return [
        Document(
            content="""PydanticAI is a Python framework for building AI agents with type safety and structured outputs.
            It provides tools for creating agents that can use external tools, maintain state, and handle complex workflows.
            Key features include automatic type validation, tool registration, and integration with various LLM providers.
            PydanticAI makes it easy to build reliable, production-ready AI agents.""",
            metadata={"topic": "pydantic-ai", "category": "framework", "source": "documentation"}
        ),
        Document(
            content="""Retrieval Augmented Generation (RAG) is a technique that combines language models with external knowledge retrieval.
            The process involves: 1) Converting documents into vector embeddings, 2) Storing them in a vector database,
            3) Retrieving relevant documents for a query, 4) Using retrieved context to generate better responses.
            RAG enables language models to access up-to-date, domain-specific information beyond their training data.""",
            metadata={"topic": "rag", "category": "technique", "source": "guide"}
        ),
        Document(
            content="""Vector databases like FAISS enable efficient similarity search over high-dimensional vectors.
            They support various indexing methods including flat, IVF, and HNSW for different performance characteristics.
            FAISS is particularly good for exact and approximate nearest neighbor search in machine learning applications.
            It can handle millions of vectors efficiently and supports both CPU and GPU acceleration.""",
            metadata={"topic": "vector-db", "category": "database", "source": "technical"}
        ),
        Document(
            content="""Cross-encoder reranking models improve retrieval quality by scoring query-document pairs directly.
            Unlike bi-encoders that create separate embeddings, cross-encoders consider query and document together.
            Models like ms-marco-MiniLM-L-6-v2 provide good performance for reranking retrieved documents.
            Reranking typically improves the relevance of top results in RAG systems.""",
            metadata={"topic": "reranking", "category": "technique", "source": "research"}
        ),
        Document(
            content="""Agentic RAG systems use AI agents to make decisions about when and how to retrieve information.
            Unlike traditional RAG which always retrieves, agentic RAG can decide based on the query whether retrieval is needed.
            Agents can also choose different retrieval strategies, combine multiple sources, and iterate on their searches.
            This approach provides more flexible and intelligent information access.""",
            metadata={"topic": "agentic-rag", "category": "advanced", "source": "research"}
        )
    ]


async def demo_basic_rag():
    """Demonstrate basic RAG functionality."""
    print("🔍 Basic RAG Demo")
    print("=" * 50)

    try:
        # Create factory and knowledge base
        factory = RAGFactory()
        documents = create_sample_knowledge_base()

        # Create simple RAG agent
        agent = factory.create_complete_rag_system(
            documents=documents,
            agent_type=AgentType.SIMPLE_RAG
        )

        print(f"✅ Created RAG system with {len(documents)} documents")

        # Test queries
        queries = [
            "What is PydanticAI and what are its main features?",
            "How does RAG work? Explain the process.",
            "What's the difference between bi-encoders and cross-encoders?"
        ]

        for i, query in enumerate(queries, 1):
            print(f"\n📝 Query {i}: {query}")
            print("-" * 60)

            result = await agent.run(query)
            print(f"🤖 Response: {result.output}")

        # Show system info
        info = factory.get_system_info()
        print(f"\n📊 System Info:")
        print(f"   Embedding Provider: {info['embedding_provider']['class']}")
        print(f"   Reranker Available: {info['reranker']['available']}")

        for name, store_info in info['vector_stores'].items():
            print(f"   Vector Store '{name}': {store_info['document_count']} documents")

    except Exception as e:
        print(f"❌ Basic RAG demo failed: {str(e)}")


async def demo_advanced_rag():
    """Demonstrate advanced RAG features."""
    print("\n🧠 Advanced RAG Demo")
    print("=" * 50)

    try:
        # Create factory with advanced features
        factory = RAGFactory()
        documents = create_sample_knowledge_base()

        # Create full RAG agent with advanced tools
        agent = factory.create_complete_rag_system(
            documents=documents,
            agent_type=AgentType.RAG
        )

        print(f"✅ Created advanced RAG system")

        # Test complex queries that require advanced reasoning
        advanced_queries = [
            "Compare PydanticAI with traditional RAG approaches and explain how agentic RAG is different.",
            "I need to build a system that can search through technical documentation. What components should I use and why?",
            "What are the trade-offs between different vector database indexing methods?"
        ]

        for i, query in enumerate(advanced_queries, 1):
            print(f"\n📝 Advanced Query {i}: {query}")
            print("-" * 60)

            result = await agent.run(query)
            print(f"🤖 Response: {result.output}")

    except Exception as e:
        print(f"❌ Advanced RAG demo failed: {str(e)}")


def demo_components():
    """Demonstrate individual components."""
    print("\n🔧 Component Demos")
    print("=" * 50)

    try:
        # Import demo functions
        from src.retrieval.embeddings import demo_all_providers as demo_embeddings
        from src.retrieval.reranker import demo_all_providers as demo_rerankers
        from src.retrieval.vector_store import demo_vector_store
        from src.tools.retrieval_tools import demo_retrieval_tools

        print("🔤 Testing Embedding Providers:")
        demo_embeddings()

        print("\n🔄 Testing Reranking Providers:")
        demo_rerankers()

        print("\n🗃️ Testing Vector Store:")
        demo_vector_store()

        print("\n🔧 Testing Retrieval Tools:")
        demo_retrieval_tools()

    except Exception as e:
        print(f"❌ Component demos failed: {str(e)}")


async def main():
    """Run all demonstrations."""
    print("🚀 Agentic RAG Service Demo")
    print("=" * 60)
    print("This demo showcases the complete agentic RAG implementation")
    print("from embeddings and vector stores to intelligent agents.")
    print("=" * 60)

    # Run component demos first
    demo_components()

    # Run RAG demos
    await demo_basic_rag()
    await demo_advanced_rag()

    print("\n" + "=" * 60)
    print("✅ All demos completed successfully!")
    print("🎉 The agentic RAG service is ready for use!")


if __name__ == "__main__":
    # Handle event loop for async demos
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()