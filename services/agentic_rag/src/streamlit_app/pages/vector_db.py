"""
Vector Database Page for Agentic RAG Service

This page demonstrates vector database functionality and document embeddings.
"""

import streamlit as st
import os
from pathlib import Path
from typing import List, Dict, Any
import logging

from ..utils.session_state import (
    initialize_session_state,
    mark_setup_complete,
    get_system_status,
    update_session_state
)
from ..utils.formatters import (
    format_document_result,
    format_metrics_display,
    format_progress_bar,
    format_error_message,
    format_success_message
)

logger = logging.getLogger(__name__)


def render():
    """Render the vector database page."""
    st.title("🗃️ Vector Database")
    st.markdown("**Understanding Document Embeddings and Semantic Search**")

    # Overview section
    st.header("🎯 What is a Vector Database?")

    st.markdown("""
    A **Vector Database** is the foundation of any RAG system. It allows us to:

    1. **Convert text to numbers** (embeddings) that capture semantic meaning
    2. **Store documents** in a searchable format
    3. **Find similar content** using mathematical similarity
    4. **Retrieve relevant information** for answering questions
    """)

    # Conceptual explanation
    with st.expander("🧠 How Embeddings Work", expanded=True):
        st.markdown("""
        **Traditional Search vs. Vector Search:**

        - **Keyword Search**: "password reset" only finds exact matches
        - **Vector Search**: "password reset" also finds "forgot password", "login issues", "account recovery"

        **The Magic of Embeddings:**
        - Similar concepts cluster together in high-dimensional space
        - "cat" and "kitten" have similar vectors
        - "password" and "authentication" are mathematically close
        """)

        # Simple visualization
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Keyword Matching**")
            st.code("""
Query: "reset password"
Documents:
❌ "forgot my login" (no match)
✅ "password reset guide" (exact match)
❌ "account recovery" (no match)
            """)

        with col2:
            st.markdown("**Vector Similarity**")
            st.code("""
Query: "reset password"
Documents:
✅ "forgot my login" (0.85 similar)
✅ "password reset guide" (0.95 similar)
✅ "account recovery" (0.78 similar)
            """)

    st.markdown("---")

    # System status check
    system_status = get_system_status()

    # Provider setup section
    st.header("⚙️ Setup Embedding Provider")

    embedding_status = system_status.get("embeddings_ready", False)

    if embedding_status:
        st.success("✅ Embedding provider is ready!")
        provider_info = st.session_state.get("embedding_provider")
        if provider_info:
            st.info(f"Using: {provider_info.get('class', 'Unknown')} with model {provider_info.get('model', 'Unknown')}")
    else:
        st.warning("⚠️ Embedding provider not configured yet")

        if st.button("🔧 Initialize Embedding Provider", type="primary"):
            with st.spinner("Setting up embedding provider..."):
                try:
                    # Import here to avoid circular imports
                    from ...factory import RAGFactory

                    # Create factory directly without session state caching for now
                    # This avoids the session state initialization issue
                    factory = RAGFactory()

                    # Get embedding provider
                    embedding_provider = factory.get_embedding_provider()

                    # Store both factory and provider info in session state
                    update_session_state({
                        "rag_factory": factory,
                        "embedding_provider": {
                            "class": embedding_provider.__class__.__name__,
                            "model": getattr(embedding_provider, 'model_name', 'Unknown')
                        }
                    })

                    mark_setup_complete("embeddings")
                    format_success_message("Embedding provider initialized!")
                    st.rerun()

                except Exception as e:
                    format_error_message("Failed to initialize embedding provider", str(e))
                    logger.exception("Failed to initialize embedding provider")

    st.markdown("---")

    # Document loading section
    st.header("📚 Load Documents")

    documents_loaded = system_status.get("documents_loaded", False)

    if documents_loaded:
        doc_count = st.session_state.get("document_count", 0)
        st.success(f"✅ {doc_count} documents loaded in vector database!")

        if st.button("🔄 Reload Documents"):
            # Reset documents
            update_session_state({
                "documents_loaded": False,
                "document_count": 0,
                "vector_store": None
            })
            st.rerun()

    else:
        st.info("📂 Load IT help desk documents into the vector database")

        # Show available documents
        data_path = Path(__file__).parent.parent.parent.parent / "data"

        if data_path.exists():
            docs = list(data_path.glob("*.md"))
            if docs:
                st.markdown(f"**Found {len(docs)} documents:**")
                for doc in docs[:5]:  # Show first 5
                    st.markdown(f"- {doc.name}")
                if len(docs) > 5:
                    st.markdown(f"- ... and {len(docs) - 5} more")

        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("📥 Load IT Documents", type="primary", disabled=not embedding_status):
                if not embedding_status:
                    st.error("Please initialize embedding provider first!")
                else:
                    load_documents()

        with col2:
            if st.button("📝 Load Custom Text", disabled=not embedding_status):
                if not embedding_status:
                    st.error("Please initialize embedding provider first!")
                else:
                    show_custom_text_loader()

    st.markdown("---")

    # Vector search demo
    if documents_loaded and embedding_status:
        st.header("🔍 Interactive Vector Search")

        st.markdown("""
        Now that we have documents loaded, let's test the vector search functionality!
        Try different queries to see how semantic search works.
        """)

        # Query input
        query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., 'How do I reset my password?', 'VPN connection issues', 'install software'",
            help="Try different phrasings to see how vector search understands meaning"
        )

        col1, col2 = st.columns([1, 3])

        with col1:
            max_results = st.slider("Max results:", 1, 10, 3)
            show_scores = st.checkbox("Show similarity scores", value=True)

        if query and st.button("🔍 Search", type="primary"):
            perform_vector_search(query, max_results, show_scores)

    elif documents_loaded and not embedding_status:
        st.warning("⚠️ Please initialize embedding provider to use vector search")
    elif not documents_loaded:
        st.info("ℹ️ Load documents first to enable vector search")

    # Educational section about vector databases
    st.markdown("---")

    with st.expander("📖 Learn More About Vector Databases"):
        st.markdown("""
        **Key Concepts:**

        1. **Embeddings**: Dense vector representations of text that capture semantic meaning
        2. **FAISS**: Facebook AI Similarity Search - fast similarity search library
        3. **Cosine Similarity**: Mathematical measure of how similar two vectors are
        4. **Chunking**: Breaking documents into smaller pieces for better retrieval

        **Real-World Applications:**
        - Document search and discovery
        - Recommendation systems
        - Semantic similarity matching
        - Content moderation
        - Scientific literature search

        **Popular Vector Databases:**
        - FAISS (local, what we're using)
        - Pinecone (cloud service)
        - Weaviate (open source)
        - Qdrant (open source)
        - Chroma (developer-friendly)
        """)

    # Next steps
    if documents_loaded and embedding_status:
        st.markdown("---")
        st.success("🎉 Vector database is ready! Next, explore how retrieval chains work.")

        if st.button("🔍 Continue to Retrieval Demo", type="primary"):
            # Navigate to next page
            st.session_state.current_page_key = "🔍 Retrieval Demo"
            st.rerun()


def load_documents():
    """Load IT documents into the vector database."""
    try:
        with st.spinner("Loading documents into vector database..."):
            # Import knowledge loader
            from ...knowledge.loader import load_it_knowledge_base
            from ...factory import RAGFactory

            # Get or create factory
            if "rag_factory" not in st.session_state or st.session_state.rag_factory is None:
                st.session_state.rag_factory = RAGFactory()

            factory = st.session_state.rag_factory

            # Load documents
            documents = load_it_knowledge_base()

            # Create vector store and load documents
            vector_store = factory.create_vector_store("it_helpdesk")
            factory.load_documents(documents, "it_helpdesk")

            # Update session state
            update_session_state({
                "documents_loaded": True,
                "document_count": len(documents),
                "vector_store": vector_store
            })

            mark_setup_complete("documents")
            mark_setup_complete("vector_store")

            format_success_message(
                f"Successfully loaded {len(documents)} documents!",
                "Documents are now searchable in the vector database."
            )

            # Show some stats
            show_document_stats(documents)

    except Exception as e:
        format_error_message("Failed to load documents", str(e))
        logger.exception("Failed to load documents")


def show_custom_text_loader():
    """Show interface for loading custom text."""
    st.subheader("📝 Add Custom Text")

    custom_text = st.text_area(
        "Enter your text:",
        placeholder="Paste any text you want to add to the knowledge base...",
        height=150
    )

    if custom_text and st.button("➕ Add to Vector Database"):
        try:
            with st.spinner("Adding custom text..."):
                from ...retrieval.base import Document
                from ...factory import RAGFactory

                # Create document
                doc = Document(
                    content=custom_text,
                    metadata={"source": "custom", "type": "user_provided"}
                )

                # Get factory
                if "rag_factory" not in st.session_state or st.session_state.rag_factory is None:
                    st.session_state.rag_factory = RAGFactory()

                factory = st.session_state.rag_factory

                # Add to vector store
                if st.session_state.get("vector_store"):
                    factory.load_documents([doc], "it_helpdesk")
                else:
                    vector_store = factory.create_vector_store("it_helpdesk")
                    factory.load_documents([doc], "it_helpdesk")
                    st.session_state.vector_store = vector_store

                # Update count
                current_count = st.session_state.get("document_count", 0)
                update_session_state({
                    "documents_loaded": True,
                    "document_count": current_count + 1
                })

                format_success_message("Custom text added to vector database!")

        except Exception as e:
            format_error_message("Failed to add custom text", str(e))


def show_document_stats(documents):
    """Show statistics about loaded documents."""
    st.subheader("📊 Document Statistics")

    # Calculate stats
    total_chars = sum(len(doc.content) for doc in documents)
    avg_length = total_chars / len(documents) if documents else 0

    # Count by topic/source
    topics = {}
    for doc in documents:
        topic = doc.metadata.get("topic", "unknown")
        topics[topic] = topics.get(topic, 0) + 1

    # Display metrics
    metrics = {
        "Total Documents": len(documents),
        "Average Length": f"{avg_length:.0f} chars",
        "Total Characters": f"{total_chars:,}",
        "Unique Topics": len(topics)
    }

    format_metrics_display(metrics)

    # Show topic breakdown
    if topics:
        st.markdown("**Documents by Topic:**")
        for topic, count in sorted(topics.items()):
            st.markdown(f"- {topic}: {count} documents")


def perform_vector_search(query: str, max_results: int, show_scores: bool):
    """Perform vector search and display results."""
    try:
        with st.spinner("Searching vector database..."):
            vector_store = st.session_state.get("vector_store")

            if not vector_store:
                st.error("Vector store not available")
                return

            # Perform search
            results = vector_store.search(query, k=max_results)

            st.subheader(f"🔍 Search Results for: '{query}'")

            if not results:
                st.warning("No results found. Try a different query.")
                return

            # Display results
            for i, result in enumerate(results, 1):
                doc = result.document
                score = result.score
                with st.expander(f"Result {i}" + (f" (Score: {score:.3f})" if show_scores else ""), expanded=(i == 1)):

                    # Document content
                    st.markdown("**Content:**")
                    st.markdown(doc.content[:500] + ("..." if len(doc.content) > 500 else ""))

                    # Metadata
                    if doc.metadata:
                        st.markdown("**Metadata:**")
                        for key, value in doc.metadata.items():
                            st.markdown(f"- **{key}**: {value}")

                    # Similarity explanation
                    if show_scores:
                        if score > 0.8:
                            similarity_text = "🟢 Very High Similarity"
                        elif score > 0.6:
                            similarity_text = "🟡 Good Similarity"
                        elif score > 0.4:
                            similarity_text = "🟠 Moderate Similarity"
                        else:
                            similarity_text = "🔴 Low Similarity"

                        st.markdown(f"**Similarity:** {similarity_text} ({score:.3f})")

            # Search tips
            with st.expander("💡 Search Tips"):
                st.markdown("""
                **Try these queries to explore vector search:**
                - "password reset" vs "forgot login" vs "account recovery"
                - "VPN connection" vs "remote access" vs "network issues"
                - "install software" vs "download application" vs "setup program"

                **Notice how:**
                - Different words with similar meanings return similar results
                - The system understands context and intent
                - Similarity scores help rank relevance
                """)

    except Exception as e:
        format_error_message("Search failed", str(e))
        logger.exception("Vector search failed")


if __name__ == "__main__":
    render()