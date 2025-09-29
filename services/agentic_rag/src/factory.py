"""
Factory for Creating Agentic RAG Components

This module provides a centralized factory for creating and configuring
all components of the agentic RAG system, following the patterns from
the agent_intro service but extended for RAG-specific functionality.
"""

from typing import Optional, Dict, Any, List
from enum import Enum
from pathlib import Path
import logging

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from .models.providers import get_available_providers, get_provider_config, get_best_available_provider
from .retrieval.embeddings import create_embedding_provider, get_best_available_provider as get_best_embedding
from .retrieval.reranker import create_reranker_provider, get_best_available_provider as get_best_reranker
from .retrieval.vector_store import create_vector_store
from .retrieval.base import Document
from .agents.rag_agent import create_rag_agent, create_simple_rag_agent
from .agents.it_helpdesk_agent import create_it_helpdesk_agent

logger = logging.getLogger(__name__)


class AgentType(str, Enum):
    """Types of RAG agents that can be created."""
    RAG = "rag"
    SIMPLE_RAG = "simple_rag"
    IT_HELPDESK = "it_helpdesk"


class RAGFactory:
    """Factory for creating configured RAG agents and components."""

    def __init__(self, store_path: Optional[str] = None):
        """Initialize the RAG factory.

        Args:
            store_path: Optional path for persistent storage of vector stores
        """
        self.store_path = Path(store_path) if store_path else None
        self._embedding_provider = None
        self._reranker = None
        self._vector_stores: Dict[str, Any] = {}

    def _create_ollama_model(self, model_name: str, base_url: str) -> OpenAIChatModel:
        """Create an Ollama model using OpenAI-compatible API.

        Args:
            model_name: Name of the Ollama model
            base_url: Base URL for Ollama server

        Returns:
            Configured OpenAIChatModel for Ollama
        """
        # Ensure base_url has /v1 suffix for OpenAI compatibility
        if not base_url.endswith('/v1'):
            base_url = f"{base_url}/v1"

        # Remove :latest tag if present, as Ollama API doesn't need it
        if model_name.endswith(':latest'):
            model_name = model_name.replace(':latest', '')

        return OpenAIChatModel(
            model_name=model_name,
            provider=OpenAIProvider(base_url=base_url)
        )

    def get_embedding_provider(
        self,
        provider_type: Optional[str] = None,
        model_key: Optional[str] = None,
        **kwargs
    ):
        """Get or create an embedding provider.

        Args:
            provider_type: Type of provider ("local" or "openai")
            model_key: Model key for the provider
            **kwargs: Additional arguments

        Returns:
            Configured embedding provider
        """
        if self._embedding_provider is None:
            if provider_type is None:
                # Auto-select best available
                self._embedding_provider = get_best_embedding()
            else:
                # Use specific provider
                if model_key is None:
                    model_key = "all-MiniLM-L6-v2" if provider_type == "local" else "ada-002"
                self._embedding_provider = create_embedding_provider(
                    provider_type, model_key, **kwargs
                )

        if self._embedding_provider is None:
            raise ValueError("No embedding provider available")

        return self._embedding_provider

    def get_reranker(
        self,
        provider_type: Optional[str] = None,
        model_key: Optional[str] = None,
        **kwargs
    ):
        """Get or create a reranker.

        Args:
            provider_type: Type of provider ("local")
            model_key: Model key for the provider
            **kwargs: Additional arguments

        Returns:
            Configured reranker or None if not available
        """
        if self._reranker is None:
            if provider_type is None:
                # Auto-select best available
                self._reranker = get_best_reranker()
            else:
                # Use specific provider
                if model_key is None:
                    model_key = "ms-marco-MiniLM-L-6-v2"
                try:
                    self._reranker = create_reranker_provider(
                        provider_type, model_key, **kwargs
                    )
                except Exception as e:
                    logger.warning(f"Failed to create reranker: {str(e)}")
                    self._reranker = None

        return self._reranker

    def create_vector_store(
        self,
        name: str = "default",
        index_type: str = "flat",
        embedding_provider=None
    ):
        """Create or get a vector store.

        Args:
            name: Name for the vector store
            index_type: Type of FAISS index
            embedding_provider: Embedding provider to use

        Returns:
            Configured vector store
        """
        if name in self._vector_stores:
            return self._vector_stores[name]

        if embedding_provider is None:
            embedding_provider = self.get_embedding_provider()

        store_path = None
        if self.store_path:
            store_path = str(self.store_path / f"{name}_vector_store")

        vector_store = create_vector_store(
            embedding_provider=embedding_provider,
            index_type=index_type,
            store_path=store_path
        )

        self._vector_stores[name] = vector_store
        logger.info(f"Created vector store '{name}' with {index_type} index")
        return vector_store

    def create_agent(
        self,
        agent_type: AgentType,
        vector_store_name: str = "default",
        model_provider: Optional[str] = None,
        **kwargs
    ) -> Agent[None, str]:
        """Create a RAG agent of the specified type.

        Args:
            agent_type: Type of agent to create
            vector_store_name: Name of vector store to use
            model_provider: Model provider to use
            **kwargs: Additional arguments for agent creation

        Returns:
            Configured RAG agent
        """
        # Get model provider
        if model_provider is None:
            provider_name = get_best_available_provider()
            if not provider_name:
                raise ValueError("No model providers available")
            if provider_name == "test":
                model_provider = "test"
            else:
                provider_config = get_provider_config(provider_name)
                model_provider = provider_config.get_model_id()

        # Handle Ollama models specially
        if isinstance(model_provider, str) and model_provider.startswith("ollama:"):
            model_name = model_provider.split(":", 1)[1]
            provider_config = get_provider_config("ollama")
            base_url = provider_config.base_url or "http://localhost:11434"
            model_provider = self._create_ollama_model(model_name, base_url)

        # Get vector store
        vector_store = self.get_vector_store(vector_store_name)
        if vector_store is None:
            vector_store = self.create_vector_store(vector_store_name)

        # Get reranker
        reranker = self.get_reranker()

        # Create agent based on type
        if agent_type == AgentType.RAG:
            return create_rag_agent(
                model_provider=model_provider,
                vector_store=vector_store,
                reranker=reranker,
                **kwargs
            )
        elif agent_type == AgentType.SIMPLE_RAG:
            return create_simple_rag_agent(
                model_provider=model_provider,
                vector_store=vector_store,
                reranker=reranker
            )
        elif agent_type == AgentType.IT_HELPDESK:
            return create_it_helpdesk_agent(
                model_provider=model_provider,
                vector_store=vector_store,
                reranker=reranker,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    def get_vector_store(self, name: str = "default"):
        """Get an existing vector store by name.

        Args:
            name: Name of the vector store

        Returns:
            Vector store if it exists, None otherwise
        """
        return self._vector_stores.get(name)

    def load_documents(
        self,
        documents: List[Document],
        vector_store_name: str = "default"
    ):
        """Load documents into a vector store.

        Args:
            documents: List of documents to load
            vector_store_name: Name of vector store to use
        """
        vector_store = self.get_vector_store(vector_store_name)
        if vector_store is None:
            vector_store = self.create_vector_store(vector_store_name)

        vector_store.add_documents(documents)
        logger.info(f"Loaded {len(documents)} documents into '{vector_store_name}'")

    def create_complete_rag_system(
        self,
        documents: List[Document],
        agent_type: AgentType = AgentType.RAG,
        vector_store_name: str = "default"
    ) -> Agent[None, str]:
        """Create a complete RAG system with documents and agent.

        Args:
            documents: Documents to load into the knowledge base
            agent_type: Type of agent to create
            vector_store_name: Name for the vector store

        Returns:
            Configured RAG agent with loaded knowledge base
        """
        # Load documents
        self.load_documents(documents, vector_store_name)

        # Create agent
        agent = self.create_agent(agent_type, vector_store_name)

        logger.info(f"Created complete RAG system with {len(documents)} documents")
        return agent

    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the RAG system status.

        Returns:
            Dictionary with system information
        """
        embedding_provider = self.get_embedding_provider()
        reranker = self.get_reranker()

        info = {
            "embedding_provider": {
                "available": embedding_provider is not None,
                "class": embedding_provider.__class__.__name__ if embedding_provider else None,
                "model": embedding_provider.model_name if embedding_provider else None
            },
            "reranker": {
                "available": reranker is not None,
                "class": reranker.__class__.__name__ if reranker else None,
                "model": reranker.model_name if reranker else None
            },
            "vector_stores": {
                name: store.get_info() for name, store in self._vector_stores.items()
            },
            "model_providers": list(get_available_providers().keys())
        }

        return info


# Convenience functions
def create_rag_system(
    documents: List[Document],
    agent_type: str = "rag",
    store_path: Optional[str] = None
) -> Agent[None, str]:
    """Create a complete RAG system with documents and agent.

    Args:
        documents: Documents to load
        agent_type: Type of agent ("rag", "simple_rag", "it_helpdesk")
        store_path: Optional path for persistent storage

    Returns:
        Configured RAG agent
    """
    factory = RAGFactory(store_path)
    return factory.create_complete_rag_system(
        documents=documents,
        agent_type=AgentType(agent_type)
    )


def create_it_helpdesk_system(
    it_documents: List[Document],
    store_path: Optional[str] = None
) -> Agent[None, str]:
    """Create a complete IT helpdesk system.

    Args:
        it_documents: IT documentation and procedures
        store_path: Optional path for persistent storage

    Returns:
        Configured IT helpdesk agent
    """
    return create_rag_system(
        documents=it_documents,
        agent_type="it_helpdesk",
        store_path=store_path
    )


# Demo function
def demo_rag_factory():
    """Demonstrate the RAG factory capabilities."""
    print("🏭 RAG Factory Demo")
    print("=" * 40)

    try:
        # Create factory
        factory = RAGFactory()

        # Show system info
        info = factory.get_system_info()
        print("📊 System Status:")
        print(f"   Embedding provider: {info['embedding_provider']['available']}")
        print(f"   Reranker: {info['reranker']['available']}")
        print(f"   Model providers: {len(info['model_providers'])}")

        # Create sample documents
        sample_docs = [
            Document(
                content="RAG combines retrieval and generation for better AI responses.",
                metadata={"topic": "rag", "type": "definition"}
            ),
            Document(
                content="Vector stores enable fast similarity search over embeddings.",
                metadata={"topic": "vector-db", "type": "explanation"}
            )
        ]

        # Create complete RAG system
        agent = factory.create_complete_rag_system(
            documents=sample_docs,
            agent_type=AgentType.SIMPLE_RAG
        )

        print(f"\n✅ Created RAG system with {len(sample_docs)} documents")
        print("🤖 Agent ready for queries!")

        # Show final system info
        final_info = factory.get_system_info()
        for name, store_info in final_info['vector_stores'].items():
            print(f"   Vector store '{name}': {store_info['document_count']} documents")

    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        logger.exception("RAG factory demo failed")


if __name__ == "__main__":
    demo_rag_factory()