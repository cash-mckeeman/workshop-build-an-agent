"""
Main Streamlit Application for Agentic RAG Service

This module provides the main Streamlit interface for exploring and
demonstrating the Agentic RAG implementation.
"""

import streamlit as st
import logging
from typing import Optional

# Import page modules
from .pages import intro, vector_db, retrieval, agent_setup, live_demo
from .utils.session_state import initialize_session_state, get_session_state
from .utils.formatters import format_system_status

logger = logging.getLogger(__name__)


def render_sidebar():
    """Render the navigation sidebar."""
    st.sidebar.title("🤖 Agentic RAG")
    st.sidebar.markdown("---")

    # Navigation
    pages = {
        "🏠 Introduction": "intro",
        "🗃️ Vector Database": "vector_db",
        "🔍 Retrieval Demo": "retrieval",
        "⚙️ Agent Setup": "agent_setup",
        "🚀 Live Demo": "live_demo"
    }

    # Get current page from session state or default to intro
    current_page_key = st.session_state.get("current_page_key", "🏠 Introduction")

    # Find the index of the current page
    page_list = list(pages.keys())
    try:
        current_index = page_list.index(current_page_key)
    except ValueError:
        current_index = 0
        current_page_key = page_list[0]

    selected_page = st.sidebar.radio(
        "Navigate to:",
        page_list,
        index=current_index,
        key="page_selection"
    )

    # Update current page in session state if it changed
    if selected_page != current_page_key:
        st.session_state.current_page_key = selected_page

    st.sidebar.markdown("---")

    # System status
    st.sidebar.subheader("📊 System Status")
    system_info = get_session_state().get("system_info", {})
    if system_info:
        status_text = format_system_status(system_info)
        st.sidebar.markdown(status_text)
    else:
        st.sidebar.info("System not initialized")

    # Help section
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ Help")
    st.sidebar.markdown("""
    **Getting Started:**
    1. Start with the Introduction
    2. Explore the Vector Database
    3. Test Retrieval functionality
    4. Configure your Agent
    5. Try the Live Demo

    **Tips:**
    - Each page builds on the previous one
    - Check the System Status for issues
    - Use the Live Demo to test your setup
    """)

    return pages[selected_page]


def main():
    """Main application entry point."""
    try:
        # Initialize session state
        initialize_session_state()

        # Render sidebar and get selected page
        selected_page = render_sidebar()

        # Render main content based on selected page
        if selected_page == "intro":
            intro.render()
        elif selected_page == "vector_db":
            vector_db.render()
        elif selected_page == "retrieval":
            retrieval.render()
        elif selected_page == "agent_setup":
            agent_setup.render()
        elif selected_page == "live_demo":
            live_demo.render()
        else:
            st.error(f"Unknown page: {selected_page}")

    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.exception("Application error in main()")

        # Show error details in development
        if st.checkbox("Show error details"):
            st.exception(e)


if __name__ == "__main__":
    main()