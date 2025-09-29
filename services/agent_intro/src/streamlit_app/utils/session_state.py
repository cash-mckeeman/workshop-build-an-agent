"""
Session state management utilities for the Streamlit app.
"""

import streamlit as st
from typing import Any, Dict, List, Optional


def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "current_page" not in st.session_state:
        st.session_state.current_page = "welcome"

    if "provider_health_checked" not in st.session_state:
        st.session_state.provider_health_checked = False

    if "health_results" not in st.session_state:
        st.session_state.health_results = {}

    if "tutorial_progress" not in st.session_state:
        st.session_state.tutorial_progress = {
            "welcome": False,
            "model_setup": False,
            "tools": False,
            "memory": False,
            "routing": False,
            "complete_agent": False
        }


def add_message(role: str, content: str):
    """Add a message to the chat history."""
    st.session_state.messages.append({"role": role, "content": content})


def clear_messages():
    """Clear all messages from chat history."""
    st.session_state.messages = []


def get_messages() -> List[Dict[str, str]]:
    """Get all messages from chat history."""
    return st.session_state.messages


def mark_progress(page: str, completed: bool = True):
    """Mark tutorial progress for a specific page."""
    if page in st.session_state.tutorial_progress:
        st.session_state.tutorial_progress[page] = completed


def get_progress(page: str) -> bool:
    """Get tutorial progress for a specific page."""
    return st.session_state.tutorial_progress.get(page, False)


def get_overall_progress() -> Dict[str, bool]:
    """Get overall tutorial progress."""
    return st.session_state.tutorial_progress.copy()


def set_current_page(page: str):
    """Set the current page."""
    st.session_state.current_page = page


def get_current_page() -> str:
    """Get the current page."""
    return st.session_state.current_page


def store_health_results(results: Dict[str, Any]):
    """Store provider health check results."""
    st.session_state.health_results = results
    st.session_state.provider_health_checked = True


def get_health_results() -> Dict[str, Any]:
    """Get stored provider health check results."""
    return st.session_state.health_results


def is_health_checked() -> bool:
    """Check if provider health has been checked."""
    return st.session_state.provider_health_checked