"""
Welcome page for the Agent Introduction Tutorial.
"""

import streamlit as st

from src.streamlit_app.components.concept_explanation import (
    show_agent_component_overview,
    show_agent_vs_chatbot,
    show_tutorial_roadmap,
)
from src.streamlit_app.utils.session_state import mark_progress


def show_welcome_page():
    """Display the welcome page content."""
    st.header("👋 Welcome to AI Agents!")

    st.markdown("""
    Welcome to the interactive tutorial on **AI Agents**! This hands-on experience will teach you
    about the four core components that make agents intelligent and capable.

    By the end of this tutorial, you'll understand how to build agents that can:
    - **Calculate** actual math problems (not just estimate)
    - **Search** the web for current information
    - **Control** other software and systems
    - **Remember** context across long conversations
    - **Chain together** multiple actions to solve complex problems
    """)

    # Agent vs Chatbot comparison
    show_agent_vs_chatbot()

    st.markdown("---")

    # Component overview
    show_agent_component_overview()

    st.markdown("---")

    # Tutorial roadmap
    show_tutorial_roadmap()

    st.markdown("---")

    # What makes this tutorial special
    st.markdown("## ✨ What Makes This Tutorial Special")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🎮 Interactive Learning
        - **Hands-on demos** with real agents
        - **Live code examples** you can modify
        - **Step-by-step building** from simple to complex
        - **Visual explanations** of how components work
        """)

    with col2:
        st.markdown("""
        ### 🔧 Multiple Model Support
        - **Cloud providers**: OpenAI, Anthropic, HuggingFace
        - **Local models**: Ollama, vLLM
        - **Easy switching** between providers
        - **Health monitoring** for all models
        """)

    # Getting started section
    st.markdown("---")
    st.markdown("## 🚀 Ready to Get Started?")

    st.markdown("""
    **Prerequisites:**
    - Basic Python knowledge (helpful but not required)
    - Curiosity about AI and automation
    - 15-30 minutes to complete the tutorial

    **No installation required!** This tutorial runs entirely in your browser.
    """)

    # Progress tracking
    col1, col2, col3 = st.columns([2, 1, 2])

    with col2:
        if st.button("✅ I'm Ready to Start!", key="ready_to_start", type="primary"):
            mark_progress("welcome", True)
            st.success(
                "Great! Head to the **Model Setup** page to configure your AI provider."
            )
            st.info("💡 Use the sidebar to navigate between tutorial pages.")

    # Tips for success
    st.markdown("---")
    st.markdown("## 💡 Tips for Success")

    tips = [
        "**Follow the pages in order** - each builds on the previous one",
        "**Experiment freely** - try modifying the examples",
        "**Ask questions** - use the chat interfaces to test your understanding",
        "**Take your time** - understanding beats speed",
        "**Have fun** - building agents is exciting!",
    ]

    for tip in tips:
        st.markdown(f"- {tip}")

    # What you'll build
    st.markdown("---")
    st.markdown("## 🏗️ What You'll Build")

    st.markdown("""
    Throughout this tutorial, you'll build increasingly sophisticated agents:

    1. **Basic Agent** - Simple chat capabilities
    2. **Math Agent** - Can perform calculations
    3. **Tool Agent** - Multiple capabilities (math, text, time)
    4. **Memory Agent** - Remembers conversation history
    5. **Complete Agent** - All components working together

    Each agent demonstrates key concepts and builds toward a full understanding
    of how modern AI agents work.
    """)

    # Final encouragement
    st.markdown("---")
    st.info("""
    🎯 **Learning Goal**: By the end of this tutorial, you'll understand the four core
    components of AI agents and how they work together to create intelligent, capable systems.

    Ready when you are! Click the button above and let's start building agents! 🚀
    """)


if __name__ == "__main__":
    show_welcome_page()
