"""
RAG retrieval tools for PydanticAI agents.

This module provides pydantic-ai tools that allow agents to perform retrieval
operations, following the patterns established in the agent_intro service.
The tools use structured inputs and outputs for reliable agent interactions.
"""

from typing import Annotated, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from pydantic_ai import RunContext
import logging

from ..retrieval.vector_store import FAISSVectorStore
from ..retrieval.base import Document, SearchResult
from ..retrieval.reranker import RerankerProvider

logger = logging.getLogger(__name__)


class RetrievalQuery(BaseModel):
    """Structured input for retrieval queries."""
    query: str = Field(description="The search query to find relevant documents")
    max_results: int = Field(default=5, ge=1, le=20, description="Maximum number of results to return")
    use_reranking: bool = Field(default=True, description="Whether to apply reranking for better relevance")
    metadata_filter: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata filters")


class DocumentResult(BaseModel):
    """Structured representation of a retrieved document."""
    content: str = Field(description="The document content")
    score: float = Field(description="Relevance score for the document")
    rank: int = Field(description="Rank in the search results")
    metadata: Dict[str, Any] = Field(description="Document metadata")
    doc_id: Optional[str] = Field(description="Document identifier")


class RetrievalResult(BaseModel):
    """Structured output for retrieval operations."""
    query: str = Field(description="The original search query")
    documents: List[DocumentResult] = Field(description="Retrieved documents")
    total_results: int = Field(description="Total number of results found")
    reranked: bool = Field(description="Whether reranking was applied")


class RAGRetrievalTools:
    """RAG retrieval tools for PydanticAI agents.

    This class provides tools that agents can use to search and retrieve
    relevant documents from the knowledge base.
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        reranker: Optional[RerankerProvider] = None
    ):
        """Initialize the RAG retrieval tools.

        Args:
            vector_store: Vector store for document retrieval
            reranker: Optional reranker for improving results
        """
        self.vector_store = vector_store
        self.reranker = reranker

    def search_knowledge_base(
        self,
        query: Annotated[str, "The search query to find relevant documents"]
    ) -> RetrievalResult:
        """Search the knowledge base for relevant documents.

        This is a simple search tool that provides a clean interface for agents
        to retrieve documents without complex parameters.

        Args:
            query: The search query

        Returns:
            RetrievalResult with the found documents
        """
        return self.advanced_search(RetrievalQuery(query=query))

    def advanced_search(
        self,
        query_params: Annotated[RetrievalQuery, "Structured query parameters for document retrieval"]
    ) -> RetrievalResult:
        """Perform advanced search with configurable parameters.

        This tool allows agents to perform more sophisticated searches with
        control over result count, reranking, and metadata filtering.

        Args:
            query_params: Structured query parameters

        Returns:
            RetrievalResult with the found documents
        """
        try:
            # Perform initial vector search
            search_results = self.vector_store.search(
                query=query_params.query,
                k=query_params.max_results * 2 if query_params.use_reranking else query_params.max_results,
                filter_metadata=query_params.metadata_filter
            )

            # Apply reranking if requested and available
            reranked = False
            if query_params.use_reranking and self.reranker and search_results:
                try:
                    documents = [result.document for result in search_results]
                    reranked_results = self.reranker.rerank(
                        query=query_params.query,
                        documents=documents,
                        top_k=query_params.max_results
                    )
                    search_results = reranked_results
                    reranked = True
                    logger.debug(f"Applied reranking to {len(search_results)} results")
                except Exception as e:
                    logger.warning(f"Reranking failed, using original results: {str(e)}")

            # Limit results if not reranked
            if not reranked:
                search_results = search_results[:query_params.max_results]

            # Convert to structured output
            document_results = []
            for result in search_results:
                document_results.append(DocumentResult(
                    content=result.document.content,
                    score=result.score,
                    rank=result.rank,
                    metadata=result.document.metadata,
                    doc_id=result.document.doc_id
                ))

            return RetrievalResult(
                query=query_params.query,
                documents=document_results,
                total_results=len(document_results),
                reranked=reranked
            )

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            # Return empty result on error
            return RetrievalResult(
                query=query_params.query,
                documents=[],
                total_results=0,
                reranked=False
            )

    def get_document_by_id(
        self,
        doc_id: Annotated[str, "The document ID to retrieve"]
    ) -> Optional[DocumentResult]:
        """Retrieve a specific document by its ID.

        Args:
            doc_id: Document identifier

        Returns:
            DocumentResult if found, None otherwise
        """
        try:
            # Search through documents for the ID
            for i, doc in enumerate(self.vector_store._documents):
                if doc.doc_id == doc_id:
                    return DocumentResult(
                        content=doc.content,
                        score=1.0,  # Perfect match for direct retrieval
                        rank=1,
                        metadata=doc.metadata,
                        doc_id=doc.doc_id
                    )
            return None

        except Exception as e:
            logger.error(f"Document retrieval failed: {str(e)}")
            return None

    def get_knowledge_base_info(self) -> Dict[str, Any]:
        """Get information about the knowledge base.

        Returns:
            Dictionary with knowledge base statistics
        """
        try:
            info = self.vector_store.get_info()

            # Add additional statistics
            topics = set()
            categories = set()
            for metadata in self.vector_store._metadata:
                if "topic" in metadata:
                    topics.add(metadata["topic"])
                if "category" in metadata:
                    categories.add(metadata["category"])

            info.update({
                "available_topics": list(topics),
                "available_categories": list(categories),
                "reranker_available": self.reranker is not None
            })

            return info

        except Exception as e:
            logger.error(f"Failed to get knowledge base info: {str(e)}")
            return {"error": str(e)}


def register_retrieval_tools(
    agent,
    vector_store: FAISSVectorStore,
    reranker: Optional[RerankerProvider] = None,
    include_advanced: bool = True
) -> None:
    """Register retrieval tools with a PydanticAI agent.

    Args:
        agent: PydanticAI agent to register tools with
        vector_store: Vector store for document retrieval
        reranker: Optional reranker for improving results
        include_advanced: Whether to include advanced search tools
    """
    # Create closure to capture dependencies
    def create_search_tool():
        def search_knowledge_base(ctx: RunContext, query: Annotated[str, "The search query to find relevant documents"]) -> RetrievalResult:
            """Search the knowledge base for relevant documents."""
            tools = RAGRetrievalTools(vector_store, reranker)
            return tools.search_knowledge_base(query)
        return search_knowledge_base

    def create_advanced_search_tool():
        def advanced_search(ctx: RunContext, query_params: Annotated[RetrievalQuery, "Structured query parameters for document retrieval"]) -> RetrievalResult:
            """Perform advanced search with configurable parameters."""
            tools = RAGRetrievalTools(vector_store, reranker)
            return tools.advanced_search(query_params)
        return advanced_search

    def create_get_document_tool():
        def get_document_by_id(ctx: RunContext, doc_id: Annotated[str, "The document ID to retrieve"]) -> Optional[DocumentResult]:
            """Retrieve a specific document by its ID."""
            tools = RAGRetrievalTools(vector_store, reranker)
            return tools.get_document_by_id(doc_id)
        return get_document_by_id

    def create_info_tool():
        def get_knowledge_base_info(ctx: RunContext) -> Dict[str, Any]:
            """Get information about the knowledge base."""
            tools = RAGRetrievalTools(vector_store, reranker)
            return tools.get_knowledge_base_info()
        return get_knowledge_base_info

    # Register basic search tool
    agent.tool(create_search_tool())

    # Register advanced tools if requested
    if include_advanced:
        agent.tool(create_advanced_search_tool())
        agent.tool(create_get_document_tool())
        agent.tool(create_info_tool())

    logger.info(f"Registered {'advanced' if include_advanced else 'basic'} retrieval tools")


# Convenience function for creating a complete retrieval tool set
def create_retrieval_tools(
    vector_store: FAISSVectorStore,
    reranker: Optional[RerankerProvider] = None
) -> RAGRetrievalTools:
    """Create a complete set of retrieval tools.

    Args:
        vector_store: Vector store for document retrieval
        reranker: Optional reranker for improving results

    Returns:
        Configured RAGRetrievalTools instance
    """
    return RAGRetrievalTools(vector_store, reranker)


# Demo function for testing
def demo_retrieval_tools():
    """Demonstrate the RAG retrieval tools."""
    print("🔧 RAG Retrieval Tools Demo")
    print("=" * 40)

    try:
        # Create embedding provider and vector store
        from ..retrieval.embeddings import get_best_available_provider
        from ..retrieval.vector_store import create_vector_store
        from ..retrieval.reranker import get_best_available_provider as get_best_reranker

        embedding_provider = get_best_available_provider()
        if not embedding_provider:
            print("❌ No embedding provider available")
            return

        vector_store = create_vector_store(embedding_provider)
        reranker = get_best_reranker()

        # Add test documents
        documents = [
            Document(
                content="PydanticAI is a framework for building AI agents with type safety.",
                metadata={"topic": "ai", "category": "framework"}
            ),
            Document(
                content="Python is excellent for data science and machine learning projects.",
                metadata={"topic": "programming", "category": "language"}
            ),
            Document(
                content="Vector databases enable semantic search and retrieval augmented generation.",
                metadata={"topic": "database", "category": "technology"}
            )
        ]

        vector_store.add_documents(documents)
        print(f"✅ Created vector store with {len(documents)} documents")

        # Create tools
        tools = create_retrieval_tools(vector_store, reranker)
        print(f"🔧 Created retrieval tools (reranker: {'available' if reranker else 'not available'})")

        # Test basic search
        query = "AI frameworks for building agents"
        result = tools.search_knowledge_base(query)

        print(f"\n🔍 Basic search for: '{query}'")
        print(f"   Found {result.total_results} documents")
        print(f"   Reranked: {result.reranked}")

        for doc in result.documents:
            print(f"   Score: {doc.score:.3f} - {doc.content[:60]}...")

        # Test advanced search
        advanced_query = RetrievalQuery(
            query="Python programming",
            max_results=2,
            use_reranking=True,
            metadata_filter={"topic": "programming"}
        )

        advanced_result = tools.advanced_search(advanced_query)
        print(f"\n🎯 Advanced search (filtered): {advanced_result.total_results} results")

        # Test knowledge base info
        info = tools.get_knowledge_base_info()
        print(f"\n📊 Knowledge base info:")
        print(f"   Documents: {info.get('document_count', 0)}")
        print(f"   Topics: {info.get('available_topics', [])}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    demo_retrieval_tools()