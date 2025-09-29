"""
Retrieval Demo Page for Agentic RAG Service

This page demonstrates retrieval functionality and search capabilities.
"""

import streamlit as st
from typing import List, Dict, Any, Optional
import logging

from ...retrieval.base import SearchResult

from ..utils.session_state import (
    get_system_status,
    update_session_state,
    mark_setup_complete
)
from ..utils.formatters import (
    format_error_message,
    format_success_message,
    format_metrics_display,
    format_info_box
)

logger = logging.getLogger(__name__)


def render():
    """Render the retrieval demo page."""
    st.title("🔍 Retrieval Demo")
    st.markdown("**Interactive Document Retrieval and Search**")

    # Check prerequisites
    system_status = get_system_status()
    embedding_ready = system_status.get("embeddings_ready", False)
    documents_loaded = system_status.get("documents_loaded", False)

    if not embedding_ready or not documents_loaded:
        st.warning("⚠️ Please complete the Vector Database setup first!")

        missing_steps = []
        if not embedding_ready:
            missing_steps.append("Initialize embedding provider")
        if not documents_loaded:
            missing_steps.append("Load documents")

        st.markdown("**Missing steps:**")
        for step in missing_steps:
            st.markdown(f"- {step}")

        if st.button("🗃️ Go to Vector Database", type="primary"):
            st.session_state.current_page_key = "🗃️ Vector Database"
            st.rerun()
        return

    # Introduction
    st.header("🎯 Advanced Retrieval Techniques")

    st.markdown("""
    Beyond basic vector search, modern RAG systems use sophisticated retrieval techniques:

    1. **Multi-step Retrieval**: First retrieve, then rerank for better relevance
    2. **Hybrid Search**: Combine semantic search with keyword matching
    3. **Context Filtering**: Use metadata to narrow search scope
    4. **Result Fusion**: Merge results from multiple search strategies
    """)

    # Retrieval comparison demo
    st.header("⚖️ Retrieval Comparison")

    query = st.text_input(
        "Enter your search query:",
        placeholder="e.g., 'How do I reset my password?', 'VPN connection problems'",
        help="We'll show you different retrieval approaches with the same query"
    )

    if query:
        col1, col2 = st.columns(2)

        with col1:
            max_results = st.slider("Max results:", 1, 10, 5)

        with col2:
            use_reranking = st.checkbox("Enable reranking", value=True,
                                       help="Reranking improves result quality but takes longer")

        if st.button("🔍 Compare Retrieval Methods", type="primary"):
            compare_retrieval_methods(query, max_results, use_reranking)

    st.markdown("---")

    # Reranking demonstration
    st.header("🏆 Reranking Deep Dive")

    st.markdown("""
    **Reranking** is a crucial step that improves retrieval quality:

    - **First Stage**: Fast vector search finds candidate documents
    - **Second Stage**: Slower but more accurate model reranks candidates
    - **Result**: Better relevance ranking with acceptable performance
    """)

    if st.button("📊 Show Reranking Impact"):
        demonstrate_reranking_impact()

    # Advanced retrieval options
    with st.expander("🔧 Advanced Retrieval Settings", expanded=False):
        st.markdown("**Experiment with different retrieval parameters:**")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Search Parameters")
            similarity_threshold = st.slider(
                "Similarity threshold:",
                0.0, 1.0, 0.3, 0.1,
                help="Minimum similarity score to include results"
            )

            search_type = st.selectbox(
                "Search type:",
                ["similarity", "mmr", "similarity_score_threshold"],
                help="Different search algorithms"
            )

        with col2:
            st.subheader("Filtering Options")
            topic_filter = st.selectbox(
                "Filter by topic:",
                ["All"] + get_available_topics(),
                help="Narrow search to specific topics"
            )

            source_filter = st.selectbox(
                "Filter by source:",
                ["All"] + get_available_sources(),
                help="Narrow search to specific document sources"
            )

        if st.button("🎯 Advanced Search"):
            perform_advanced_search(
                query, max_results, similarity_threshold,
                search_type, topic_filter, source_filter
            )

    st.markdown("---")

    # Educational section
    st.header("📚 Understanding Retrieval Quality")

    quality_tabs = st.tabs([
        "Precision vs Recall",
        "Embedding Quality",
        "Chunking Strategy",
        "Metadata Importance"
    ])

    with quality_tabs[0]:
        st.markdown("""
        **Precision vs Recall Trade-off:**

        - **Precision**: What percentage of returned documents are relevant?
        - **Recall**: What percentage of relevant documents were found?
        - **Challenge**: Optimizing both simultaneously is difficult

        **In Practice:**
        - More results → Higher recall, potentially lower precision
        - Stricter filtering → Higher precision, potentially lower recall
        - Reranking helps improve precision without hurting recall
        """)

        if st.button("📈 Analyze Query Performance"):
            analyze_query_performance()

    with quality_tabs[1]:
        st.markdown("""
        **Embedding Model Impact:**

        Different embedding models excel at different tasks:
        - **General purpose**: Good for most content
        - **Domain-specific**: Better for specialized content
        - **Multilingual**: Handle multiple languages
        - **Code-aware**: Understand programming concepts

        **Key Factors:**
        - Model size vs speed trade-off
        - Training data domain match
        - Embedding dimensionality
        """)

    with quality_tabs[2]:
        st.markdown("""
        **Chunking Strategy Impact:**

        How you split documents affects retrieval quality:
        - **Chunk size**: Larger chunks have more context but less precision
        - **Overlap**: Prevents information from being split across chunks
        - **Boundaries**: Smart splitting at sentences/paragraphs works better

        **Best Practices:**
        - 100-800 characters per chunk for most content
        - 20-100 character overlap
        - Preserve paragraph boundaries when possible
        """)

    with quality_tabs[3]:
        st.markdown("""
        **Metadata for Better Retrieval:**

        Rich metadata enables:
        - **Filtering**: Only search relevant document types
        - **Boosting**: Prefer certain sources or topics
        - **Context**: Provide additional information to users

        **Useful Metadata:**
        - Document type, topic, author
        - Creation/modification dates
        - Confidence scores, review status
        - Relationships to other documents
        """)

    # Performance tips
    with st.expander("⚡ Performance Optimization Tips"):
        st.markdown("""
        **Speed up retrieval:**
        1. **Index Optimization**: Use appropriate FAISS index type
        2. **Embedding Caching**: Cache embeddings for repeated queries
        3. **Async Processing**: Use async/await for concurrent operations
        4. **Batch Processing**: Process multiple queries together
        5. **Result Caching**: Cache popular query results

        **Quality improvements:**
        1. **Query Expansion**: Add synonyms and related terms
        2. **Negative Sampling**: Learn what not to retrieve
        3. **User Feedback**: Improve based on user interactions
        4. **A/B Testing**: Compare different retrieval strategies
        """)

    # Next steps
    if system_status.get("vector_store_ready"):
        st.markdown("---")
        st.success("🎉 Retrieval system is working! Next, let's set up the complete RAG agent.")

        if st.button("⚙️ Continue to Agent Setup", type="primary"):
            st.session_state.current_page_key = "⚙️ Agent Setup"
            st.rerun()


def compare_retrieval_methods(query: str, max_results: int, use_reranking: bool):
    """Compare different retrieval methods side by side."""
    try:
        vector_store = st.session_state.get("vector_store")
        if not vector_store:
            st.error("Vector store not available")
            return

        st.subheader(f"🔍 Comparing retrieval for: '{query}'")

        # Basic vector search
        with st.spinner("Running basic vector search..."):
            basic_results = vector_store.search(query, k=max_results)

        # Reranked results (if enabled)
        reranked_results = None
        if use_reranking:
            try:
                with st.spinner("Running reranked search..."):
                    reranked_results = perform_reranked_search(query, max_results)
            except Exception as e:
                st.warning(f"Reranking failed: {str(e)}")

        # Display comparison
        if reranked_results:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🔍 Basic Vector Search")
                display_search_results(basic_results, "basic")

            with col2:
                st.markdown("### 🏆 Reranked Results")
                display_search_results(reranked_results, "reranked")
        else:
            st.markdown("### 🔍 Vector Search Results")
            display_search_results(basic_results, "basic")

        # Analysis
        analyze_retrieval_results(basic_results, reranked_results)

    except Exception as e:
        format_error_message("Retrieval comparison failed", str(e))
        logger.exception("Retrieval comparison failed")


def perform_reranked_search(query: str, max_results: int):
    """Perform search with reranking."""
    vector_store = st.session_state.get("vector_store")

    # Get more candidates for reranking
    candidates = vector_store.search(query, k=max_results * 2)

    # Simulate reranking (in real implementation, would use actual reranker)
    # For demo purposes, we'll adjust scores based on query-document similarity
    reranked = []
    for result in candidates:
        doc = result.document
        score = result.score

        # Simple reranking simulation: boost scores for documents with query terms
        boost = 0
        query_terms = query.lower().split()
        doc_text = doc.content.lower()

        for term in query_terms:
            if term in doc_text:
                boost += 0.1

        adjusted_score = min(1.0, score + boost)
        reranked.append(SearchResult(document=doc, score=adjusted_score))

    # Sort by adjusted score and return top results
    reranked.sort(key=lambda x: x.score, reverse=True)
    return reranked[:max_results]


def display_search_results(results: List, result_type: str):
    """Display search results in a formatted way."""
    if not results:
        st.warning("No results found")
        return

    for i, item in enumerate(results, 1):
        # Handle both SearchResult objects and tuples (for reranked results)
        if hasattr(item, 'document'):
            # SearchResult object
            doc = item.document
            score = item.score
        else:
            # Tuple (doc, score) for reranked results
            doc, score = item

        with st.expander(f"Result {i} (Score: {score:.3f})", expanded=(i == 1)):
            # Content preview
            content = doc.content
            preview = content[:200] + "..." if len(content) > 200 else content
            st.markdown(f"**Content:** {preview}")

            # Metadata
            if hasattr(doc, 'metadata') and doc.metadata:
                st.markdown("**Metadata:**")
                for key, value in doc.metadata.items():
                    st.markdown(f"- {key}: {value}")

            # Score interpretation
            if score > 0.8:
                st.success("🟢 Highly relevant")
            elif score > 0.6:
                st.info("🟡 Moderately relevant")
            else:
                st.warning("🟠 Low relevance")


def analyze_retrieval_results(basic_results, reranked_results):
    """Analyze and compare retrieval results."""
    st.subheader("📊 Retrieval Analysis")

    if basic_results:
        basic_avg_score = sum(result.score for result in basic_results) / len(basic_results)

        metrics = {"Basic Search Avg Score": f"{basic_avg_score:.3f}"}

        if reranked_results:
            reranked_avg_score = sum(result.score for result in reranked_results) / len(reranked_results)
            metrics["Reranked Avg Score"] = f"{reranked_avg_score:.3f}"
            metrics["Score Improvement"] = f"{reranked_avg_score - basic_avg_score:+.3f}"

        format_metrics_display(metrics)


def demonstrate_reranking_impact():
    """Show the impact of reranking on search results."""
    st.subheader("🏆 Reranking Impact Analysis")

    # Sample queries that show reranking benefits
    sample_queries = [
        "password reset procedure",
        "VPN connection troubleshooting",
        "software installation issues",
        "network access problems"
    ]

    selected_query = st.selectbox("Choose a sample query:", sample_queries)

    if st.button("🔍 Analyze Reranking Impact"):
        try:
            vector_store = st.session_state.get("vector_store")
            if not vector_store:
                st.error("Vector store not available")
                return

            # Get results with and without reranking
            basic_results = vector_store.search(selected_query, k=5)
            reranked_results = perform_reranked_search(selected_query, 5)

            # Show side-by-side comparison
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Before Reranking**")
                for i, result in enumerate(basic_results[:3], 1):
                    st.markdown(f"{i}. Score: {result.score:.3f}")
                    st.markdown(f"   {result.document.content[:100]}...")

            with col2:
                st.markdown("**After Reranking**")
                for i, result in enumerate(reranked_results[:3], 1):
                    st.markdown(f"{i}. Score: {result.score:.3f}")
                    st.markdown(f"   {result.document.content[:100]}...")

        except Exception as e:
            format_error_message("Reranking analysis failed", str(e))


def get_available_topics() -> List[str]:
    """Get list of available document topics."""
    try:
        vector_store = st.session_state.get("vector_store")
        if not vector_store:
            return []

        # This would typically query the vector store for unique metadata values
        # For demo purposes, return common IT topics
        return [
            "password-reset",
            "vpn-access",
            "software-installation",
            "network-troubleshooting",
            "security-policies",
            "hardware-support"
        ]
    except:
        return []


def get_available_sources() -> List[str]:
    """Get list of available document sources."""
    try:
        # Return common document sources
        return [
            "IT Help Desk",
            "Security Team",
            "Network Admin",
            "Software Team",
            "Hardware Support"
        ]
    except:
        return []


def perform_advanced_search(query: str, max_results: int, similarity_threshold: float,
                           search_type: str, topic_filter: str, source_filter: str):
    """Perform advanced search with filtering and custom parameters."""
    try:
        vector_store = st.session_state.get("vector_store")
        if not vector_store:
            st.error("Vector store not available")
            return

        st.subheader(f"🎯 Advanced Search Results: '{query}'")

        # Build search parameters
        search_kwargs = {"k": max_results}

        if search_type == "similarity_score_threshold":
            search_kwargs["score_threshold"] = similarity_threshold

        # Apply filters (simplified for demo)
        filter_criteria = {}
        if topic_filter != "All":
            filter_criteria["topic"] = topic_filter
        if source_filter != "All":
            filter_criteria["source"] = source_filter

        if filter_criteria:
            search_kwargs["filter"] = filter_criteria

        # Perform search
        with st.spinner("Performing advanced search..."):
            results = vector_store.search(query, **search_kwargs)

        # Display results
        if results:
            st.success(f"Found {len(results)} results matching criteria")
            display_search_results(results, "advanced")

            # Show applied filters
            if filter_criteria:
                st.info(f"**Applied filters:** {filter_criteria}")
        else:
            st.warning("No results found with the specified criteria. Try relaxing the filters.")

    except Exception as e:
        format_error_message("Advanced search failed", str(e))


def analyze_query_performance():
    """Analyze the performance of different queries."""
    st.subheader("📈 Query Performance Analysis")

    # Sample queries with different characteristics
    test_queries = [
        ("Specific query", "How do I reset my network password?"),
        ("General query", "password help"),
        ("Complex query", "I'm having trouble connecting to the VPN and getting network timeouts"),
        ("Ambiguous query", "login issues")
    ]

    try:
        vector_store = st.session_state.get("vector_store")
        if not vector_store:
            st.error("Vector store not available")
            return

        results_data = []

        with st.spinner("Analyzing query performance..."):
            for query_type, query in test_queries:
                results = vector_store.search(query, k=5)

                if results:
                    avg_score = sum(result.score for result in results) / len(results)
                    max_score = max(result.score for result in results)
                    min_score = min(result.score for result in results)
                else:
                    avg_score = max_score = min_score = 0

                results_data.append({
                    "Query Type": query_type,
                    "Query": query,
                    "Results Found": len(results),
                    "Avg Score": f"{avg_score:.3f}",
                    "Max Score": f"{max_score:.3f}",
                    "Min Score": f"{min_score:.3f}"
                })

        # Display results table
        st.table(results_data)

        # Analysis insights
        st.markdown("""
        **Performance Insights:**
        - **Specific queries** typically get higher scores due to exact term matches
        - **General queries** may return more diverse results with lower precision
        - **Complex queries** can benefit from query preprocessing and expansion
        - **Ambiguous queries** often need clarification or context from users
        """)

    except Exception as e:
        format_error_message("Performance analysis failed", str(e))


if __name__ == "__main__":
    render()