"""
Main Streamlit application for Agent Introduction Tutorial.

This refactored application follows the planned architecture with separate
pages, components, and utilities for better organization and maintainability.
"""

import sys
from pathlib import Path

import streamlit as st

# Add the src directory to Python path
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from src.factory import get_recommended_setup
from src.models import check_all_providers
from src.streamlit_app.pages.basic_agent import show_basic_agent_page
from src.streamlit_app.pages.complete_agent import show_complete_agent_page
from src.streamlit_app.pages.memory import show_memory_page
from src.streamlit_app.pages.model_setup import show_model_setup_page
from src.streamlit_app.pages.routing import show_routing_page
from src.streamlit_app.pages.tools import show_tools_page

# Import page modules
from src.streamlit_app.pages.welcome import show_welcome_page
from src.streamlit_app.utils.formatters import (
    format_provider_status,
    format_tutorial_progress,
)

# Import utilities
from src.streamlit_app.utils.session_state import (
    get_overall_progress,
    initialize_session_state,
)


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Agent Introduction Tutorial",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state
    initialize_session_state()

    # Header
    st.title("🤖 Introduction to AI Agents")
    st.markdown("""
    Welcome to the interactive tutorial on **AI Agents**! This hands-on experience will teach you
    about the four core components that make agents intelligent and capable.
    """)

    # Sidebar configuration
    render_sidebar()

    # Main content area with page navigation
    render_main_content()


def render_sidebar():
    """Render the sidebar with configuration and navigation."""
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Provider status
        render_provider_status()

        st.markdown("---")

        # Tutorial progress
        render_tutorial_progress()

        st.markdown("---")

        # Page navigation
        render_page_navigation()


def render_provider_status():
    """Render provider configuration status."""
    setup = get_recommended_setup()

    if setup["status"] == "ready":
        st.success(f"✅ Ready with {setup['recommended_provider']}")
        st.info(f"Model: {setup['recommended_model']}")
    else:
        st.error("❌ No providers available")
        st.warning("Please configure API keys or start Ollama")

        with st.expander("Setup Instructions"):
            st.markdown("""
            **Option 1: Cloud Providers (Recommended)**
            - Set `OPENAI_API_KEY` environment variable
            - Or set `ANTHROPIC_API_KEY` for Claude
            - Or set `HUGGINGFACE_API_KEY` for open source

            **Option 2: Local Models**
            - Install Ollama: `curl -fsSL https://ollama.ai/install.sh | sh`
            - Start server: `ollama serve`
            - Pull a model: `ollama pull llama3.2`
            """)

    # Provider health check
    if st.button("🔍 Check Provider Health"):
        with st.spinner("Checking providers..."):
            health_results = check_all_providers()

            st.subheader("Provider Status")
            for name, result in health_results.items():
                status_msg = format_provider_status(
                    name, result.status.value, result.message
                )
                if result.status.value == "healthy":
                    st.success(status_msg)
                else:
                    st.error(status_msg)


def render_tutorial_progress():
    """Render tutorial progress tracker."""
    st.subheader("📊 Tutorial Progress")

    progress = get_overall_progress()
    progress_text = format_tutorial_progress(progress)
    st.markdown(progress_text)

    # Individual page progress
    page_status = [
        ("👋 Welcome", progress.get("welcome", False)),
        ("⚙️ Model Setup", progress.get("model_setup", False)),
        ("🤖 Basic Agent", progress.get("basic_agent", False)),
        ("🛠️ Tools", progress.get("tools", False)),
        ("💾 Memory", progress.get("memory", False)),
        ("🚦 Routing", progress.get("routing", False)),
        ("🎯 Complete Agent", progress.get("complete_agent", False)),
    ]

    for page_name, completed in page_status:
        if completed:
            st.markdown(f"✅ {page_name}")
        else:
            st.markdown(f"⏳ {page_name}")


def render_page_navigation():
    """Render page navigation."""
    st.subheader("📖 Tutorial Pages")

    # Page selection
    page_options = [
        ("Welcome", "👋"),
        ("Model Setup", "⚙️"),
        ("Basic Agent", "🤖"),
        ("Tools", "🛠️"),
        ("Memory", "💾"),
        ("Routing", "🚦"),
        ("Complete Agent", "🎯"),
    ]

    # Use selectbox for page navigation
    selected_page = st.selectbox(
        "Choose a page:",
        [f"{icon} {name}" for name, icon in page_options],
        index=0,
        key="page_selector",
    )

    # Store selected page in session state
    page_name = selected_page.split(" ", 1)[1]  # Remove icon
    st.session_state.current_page = page_name.lower().replace(" ", "_")


def render_main_content():
    """Render the main content area based on selected page."""
    current_page = st.session_state.get("current_page", "welcome")

    # Route to appropriate page
    if current_page == "welcome":
        show_welcome_page()
    elif current_page == "model_setup":
        show_model_setup_page()
    elif current_page == "basic_agent":
        show_basic_agent_page()
    elif current_page == "tools":
        show_tools_page()
    elif current_page == "memory":
        show_memory_page()
    elif current_page == "routing":
        show_routing_page()
    elif current_page == "complete_agent":
        show_complete_agent_page()
    else:
        # Fallback to welcome page
        show_welcome_page()


if __name__ == "__main__":
    main()
