"""
Streamlit Entry Point for Agentic RAG Service

This script provides the main entry point for the Streamlit web interface,
which demonstrates the educational Agentic RAG implementation.
"""

import sys
import streamlit as st
from pathlib import Path

# Add src to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

# Import after path setup
from src.streamlit_app.app import main as run_app

if __name__ == "__main__":
    # Configure Streamlit page
    st.set_page_config(
        page_title="Agentic RAG Service",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Run the main app
    run_app()