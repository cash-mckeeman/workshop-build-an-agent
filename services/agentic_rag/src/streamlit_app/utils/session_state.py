"""
Session State Management for Streamlit App

This module manages the Streamlit session state for the Agentic RAG application.
"""

import streamlit as st
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def initialize_session_state():
    """Initialize the Streamlit session state with default values."""
    defaults = {
        # System components
        "rag_factory": None,
        "vector_store": None,
        "embedding_provider": None,
        "reranker": None,
        "agent": None,

        # System information
        "system_info": {},
        "documents_loaded": False,
        "document_count": 0,

        # Configuration
        "chunk_size": 800,
        "chunk_overlap": 100,
        "max_results": 5,
        "model_provider": None,

        # UI state
        "current_page": "intro",
        "show_advanced": False,
        "demo_query": "",
        "chat_history": [],

        # Progress tracking
        "setup_completed": {
            "embeddings": False,
            "vector_store": False,
            "documents": False,
            "agent": False
        }
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_session_state() -> Dict[str, Any]:
    """Get the current session state as a dictionary."""
    return dict(st.session_state)


def update_session_state(updates: Dict[str, Any]):
    """Update multiple session state values.

    Args:
        updates: Dictionary of key-value pairs to update
    """
    for key, value in updates.items():
        st.session_state[key] = value


def clear_session_state():
    """Clear all session state (useful for reset)."""
    keys_to_keep = ["current_page"]  # Keep page navigation
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]
    initialize_session_state()


def get_system_status() -> Dict[str, bool]:
    """Get the current system setup status.

    Returns:
        Dictionary indicating which components are ready
    """
    return {
        "embeddings_ready": st.session_state.get("embedding_provider") is not None,
        "vector_store_ready": st.session_state.get("vector_store") is not None,
        "documents_loaded": st.session_state.get("documents_loaded", False),
        "agent_ready": st.session_state.get("agent") is not None,
        "system_ready": all([
            st.session_state.get("embedding_provider") is not None,
            st.session_state.get("vector_store") is not None,
            st.session_state.get("documents_loaded", False),
            st.session_state.get("agent") is not None
        ])
    }


def is_component_ready(component: str) -> bool:
    """Check if a specific component is ready.

    Args:
        component: Component name to check

    Returns:
        True if component is ready, False otherwise
    """
    status = get_system_status()
    return status.get(f"{component}_ready", False)


def mark_setup_complete(component: str):
    """Mark a setup step as complete.

    Args:
        component: Component that was set up
    """
    if "setup_completed" not in st.session_state:
        st.session_state.setup_completed = {}

    st.session_state.setup_completed[component] = True


def get_setup_progress() -> Dict[str, bool]:
    """Get the setup progress.

    Returns:
        Dictionary of setup completion status
    """
    return st.session_state.get("setup_completed", {})


def add_chat_message(role: str, content: str):
    """Add a message to the chat history.

    Args:
        role: Message role ("user" or "assistant")
        content: Message content
    """
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    import time
    st.session_state.chat_history.append({
        "role": role,
        "content": content,
        "timestamp": time.time()  # Simple timestamp
    })


def clear_chat_history():
    """Clear the chat history."""
    st.session_state.chat_history = []


def get_chat_history() -> list:
    """Get the current chat history."""
    return st.session_state.get("chat_history", [])