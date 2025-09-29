"""
Formatting Utilities for Streamlit App

This module provides formatting utilities for displaying information
in the Streamlit interface.
"""

from typing import Dict, Any, List, Optional
import streamlit as st


def format_system_status(system_info: Dict[str, Any]) -> str:
    """Format system status information for display.

    Args:
        system_info: System information dictionary

    Returns:
        Formatted status string
    """
    status_parts = []

    # Embedding provider status
    embedding_info = system_info.get("embedding_provider", {})
    if embedding_info.get("available"):
        status_parts.append(f"✅ Embeddings: {embedding_info.get('class', 'Unknown')}")
    else:
        status_parts.append("❌ Embeddings: Not available")

    # Reranker status
    reranker_info = system_info.get("reranker", {})
    if reranker_info.get("available"):
        status_parts.append(f"✅ Reranker: Available")
    else:
        status_parts.append("⚠️ Reranker: Not available")

    # Vector stores
    vector_stores = system_info.get("vector_stores", {})
    if vector_stores:
        total_docs = sum(store.get("document_count", 0) for store in vector_stores.values())
        status_parts.append(f"📚 Documents: {total_docs}")
    else:
        status_parts.append("📚 Documents: 0")

    # Model providers
    providers = system_info.get("model_providers", [])
    if providers:
        status_parts.append(f"🤖 Models: {len(providers)} available")
    else:
        status_parts.append("🤖 Models: None available")

    return "\n".join(status_parts)


def format_document_result(doc_result: Dict[str, Any]) -> str:
    """Format a document result for display.

    Args:
        doc_result: Document result dictionary

    Returns:
        Formatted document string
    """
    content = doc_result.get("content", "")
    score = doc_result.get("score", 0.0)
    metadata = doc_result.get("metadata", {})

    # Truncate content for display
    display_content = content[:200] + "..." if len(content) > 200 else content

    # Format metadata
    source = metadata.get("source", "Unknown")
    topic = metadata.get("topic", "general")

    return f"""
**Score: {score:.3f}** | **Topic: {topic}**
**Source:** {source}

{display_content}
"""


def format_retrieval_result(result: Dict[str, Any]) -> str:
    """Format a retrieval result for display.

    Args:
        result: Retrieval result dictionary

    Returns:
        Formatted result string
    """
    query = result.get("query", "")
    documents = result.get("documents", [])
    total_results = result.get("total_results", 0)
    reranked = result.get("reranked", False)

    header = f"""
**Query:** {query}
**Results:** {total_results} documents found
**Reranked:** {'Yes' if reranked else 'No'}
"""

    if not documents:
        return header + "\n\nNo documents found."

    doc_sections = []
    for i, doc in enumerate(documents, 1):
        doc_text = format_document_result(doc)
        doc_sections.append(f"### Document {i}\n{doc_text}")

    return header + "\n\n" + "\n\n---\n\n".join(doc_sections)


def format_chat_message(role: str, content: str) -> None:
    """Display a chat message with appropriate styling.

    Args:
        role: Message role ("user" or "assistant")
        content: Message content
    """
    if role == "user":
        st.markdown(f"**You:** {content}")
    else:
        st.markdown(f"**🤖 Assistant:** {content}")


def format_progress_bar(completed_steps: Dict[str, bool]) -> None:
    """Display a progress bar for setup steps.

    Args:
        completed_steps: Dictionary of step completion status
    """
    total_steps = len(completed_steps)
    completed_count = sum(completed_steps.values())
    progress = completed_count / total_steps if total_steps > 0 else 0

    st.progress(progress)
    st.caption(f"Setup Progress: {completed_count}/{total_steps} steps completed")

    # Show individual step status
    for step, completed in completed_steps.items():
        icon = "✅" if completed else "⏳"
        st.write(f"{icon} {step.replace('_', ' ').title()}")


def format_metrics_display(metrics: Dict[str, Any]) -> None:
    """Display metrics in a formatted way.

    Args:
        metrics: Dictionary of metrics to display
    """
    # Create columns for metrics
    cols = st.columns(len(metrics))

    for i, (metric_name, metric_value) in enumerate(metrics.items()):
        with cols[i]:
            if isinstance(metric_value, (int, float)):
                st.metric(metric_name.replace("_", " ").title(), metric_value)
            else:
                st.metric(metric_name.replace("_", " ").title(), str(metric_value))


def format_error_message(error: str, details: Optional[str] = None) -> None:
    """Display an error message with optional details.

    Args:
        error: Main error message
        details: Optional detailed error information
    """
    st.error(f"❌ {error}")

    if details:
        with st.expander("Error Details"):
            st.code(details)


def format_success_message(message: str, details: Optional[str] = None) -> None:
    """Display a success message with optional details.

    Args:
        message: Main success message
        details: Optional additional information
    """
    st.success(f"✅ {message}")

    if details:
        st.info(details)


def format_info_box(title: str, content: str, type: str = "info") -> None:
    """Display an information box.

    Args:
        title: Box title
        content: Box content
        type: Box type ("info", "warning", "success", "error")
    """
    formatted_content = f"**{title}**\n\n{content}"

    if type == "info":
        st.info(formatted_content)
    elif type == "warning":
        st.warning(formatted_content)
    elif type == "success":
        st.success(formatted_content)
    elif type == "error":
        st.error(formatted_content)
    else:
        st.markdown(formatted_content)


def format_code_block(code: str, language: str = "python") -> None:
    """Display a code block with syntax highlighting.

    Args:
        code: Code to display
        language: Programming language for syntax highlighting
    """
    st.code(code, language=language)


def format_json_display(data: Dict[str, Any], title: Optional[str] = None) -> None:
    """Display JSON data in a formatted way.

    Args:
        data: Data to display
        title: Optional title for the display
    """
    if title:
        st.subheader(title)

    st.json(data)


def create_download_button(data: str, filename: str, label: str, mime_type: str = "text/plain") -> None:
    """Create a download button for data.

    Args:
        data: Data to download
        filename: Suggested filename
        label: Button label
        mime_type: MIME type of the data
    """
    st.download_button(
        label=label,
        data=data,
        file_name=filename,
        mime=mime_type
    )