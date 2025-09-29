"""
Streamlit Tutorial Application for Agent Introduction

This interactive application walks users through the four core components of AI agents:
1. Model - The LLM that powers the agent
2. Tools - Functions the agent can call
3. Memory - Conversation history and context
4. Routing - Logic to decide what to do next

Features:
- Local mode (Ollama) and Cloud mode (OpenAI, Anthropic, etc.)
- Progressive tutorial: Basic → Math → Tools
- Interactive agent playground
- Real-time provider health monitoring
"""

import streamlit as st
import sys
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from factory import (
    AgentFactory,
    get_recommended_setup,
    create_tutorial_agent
)
from models import (
    get_available_providers,
    check_all_providers
)
from models.health import print_health_status


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Agent Introduction Tutorial",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Header
    st.title("🤖 Introduction to AI Agents")
    st.markdown("""
    Welcome to the interactive tutorial on **AI Agents**! This hands-on experience will teach you
    about the four core components that make agents intelligent and capable.
    """)

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Provider selection
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

        # Mode selection
        mode = st.radio(
            "Tutorial Mode",
            ["🎓 Beginner", "🔬 Intermediate", "🚀 Advanced"],
            help="Choose your learning level"
        )

        # Provider health status
        if st.button("🔍 Check Provider Health"):
            with st.spinner("Checking providers..."):
                health_results = check_all_providers()

                st.subheader("Provider Status")
                for name, result in health_results.items():
                    if result.status.value == "healthy":
                        st.success(f"✅ {name.upper()}")
                    else:
                        st.error(f"❌ {name.upper()}: {result.message}")

    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs([
        "📚 Tutorial",
        "🎮 Playground",
        "🧮 Math Demo",
        "🔧 Tools Demo"
    ])

    with tab1:
        tutorial_section()

    with tab2:
        playground_section()

    with tab3:
        math_demo_section()

    with tab4:
        tools_demo_section()


def tutorial_section():
    """Educational content section."""
    st.header("📚 Learn About AI Agents")

    st.markdown("""
    ## What is an AI Agent?

    An AI agent is a system that can perceive its environment, make decisions, and take actions
    to achieve specific goals. Think of it as an AI that can not only chat with you but also
    **do things** in the real world.

    ### The Four Core Components

    Every effective AI agent is built on these four fundamental components:
    """)

    # Component explanations
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧠 1. Model")
        st.markdown("""
        **The "brain" of your agent**
        - Large Language Model (LLM) that processes and generates text
        - Examples: GPT-4, Claude, Llama
        - Handles reasoning, understanding, and decision-making
        """)

        st.subheader("🛠️ 2. Tools")
        st.markdown("""
        **Functions the agent can execute**
        - Mathematical calculations
        - Web searches, API calls
        - File operations, database queries
        - Any Python function you define!
        """)

    with col2:
        st.subheader("🧠 3. Memory")
        st.markdown("""
        **Conversation history and context**
        - Remembers previous interactions
        - Maintains conversation flow
        - Can reference earlier information
        - Essential for multi-turn conversations
        """)

        st.subheader("🚦 4. Routing")
        st.markdown("""
        **Decision-making logic**
        - Determines which tool to use when
        - Manages conversation flow
        - Handles error cases and retries
        - Orchestrates the entire interaction
        """)

    st.markdown("""
    ---
    ## Why This Matters

    Traditional chatbots can only respond with text. AI agents can:
    - **Calculate** actual math problems (not just estimate)
    - **Search** the web for current information
    - **Control** other software and systems
    - **Remember** context across long conversations
    - **Chain together** multiple actions to solve complex problems

    Let's see this in action! 👇
    """)


def playground_section():
    """Interactive agent playground."""
    st.header("🎮 Agent Playground")

    setup = get_recommended_setup()

    if setup["status"] != "ready":
        st.error("Please configure a provider first (see sidebar)")
        return

    st.markdown("""
    This is your sandbox to experiment with different types of agents. Try asking questions
    and see how the different components work together!
    """)

    # Agent type selection
    agent_type = st.selectbox(
        "Choose Agent Type",
        ["Basic Agent (MODEL only)", "Math Agent (MODEL + TOOLS)", "Tool Agent (MODEL + TOOLS + more)"],
        help="See how adding components changes capabilities"
    )

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask your agent anything..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Create appropriate agent
                    if "Basic" in agent_type:
                        agent = create_tutorial_agent("basic")
                    elif "Math" in agent_type:
                        agent = create_tutorial_agent("math")
                    else:
                        agent = create_tutorial_agent("tools")

                    # Get response
                    result = agent.run_sync(prompt)
                    response = result.output

                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()


def math_demo_section():
    """Recreation of the original tutorial math demo."""
    st.header("🧮 Math Agent Demo")

    st.markdown("""
    This recreates the exact workflow from the original `intro_to_agents.ipynb` notebook.
    We'll create an agent with an `add()` tool and ask "What is 3 plus 12?"
    """)

    setup = get_recommended_setup()

    if setup["status"] != "ready":
        st.error("Please configure a provider first (see sidebar)")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Without Tools (Basic Agent)")
        st.code("""
# Basic agent - MODEL only
agent = Agent("model", instructions="You are helpful")
result = agent.run_sync("What is 3 plus 12?")
        """)

        if st.button("🚫 Try Basic Agent", key="basic_math"):
            with st.spinner("Running basic agent..."):
                try:
                    agent = create_tutorial_agent("basic")
                    result = agent.run_sync("What is 3 plus 12?")
                    st.write("**Response:**", result.output)
                    st.warning("⚠️ Cannot provide exact calculation - no tools available!")
                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        st.subheader("With Tools (Math Agent)")
        st.code("""
# Math agent - MODEL + TOOLS
@agent.tool
def add(a: int, b: int) -> int:
    return a + b

result = agent.run_sync("What is 3 plus 12?")
        """)

        if st.button("✅ Try Math Agent", key="math_agent"):
            with st.spinner("Running math agent..."):
                try:
                    agent = create_tutorial_agent("math")
                    result = agent.run_sync("What is 3 plus 12?")
                    st.write("**Response:**", result.output)
                    st.success("✅ Exact calculation using the add() tool!")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("""
    ---
    ### What Just Happened?

    1. **MODEL**: Both agents understood the question
    2. **TOOLS**: Only the math agent had access to the `add()` function
    3. **MEMORY**: Both agents maintained conversation context
    4. **ROUTING**: The math agent automatically decided to use the tool

    This demonstrates why **TOOLS** are crucial for agents that need to interact with the real world!
    """)


def tools_demo_section():
    """Advanced tools demonstration."""
    st.header("🔧 Tools Demonstration")

    setup = get_recommended_setup()

    if setup["status"] != "ready":
        st.error("Please configure a provider first (see sidebar)")
        return

    st.markdown("""
    Let's explore various types of tools that agents can use. Each tool gives the agent
    a new capability!
    """)

    # Tool examples
    st.subheader("Available Tools")

    tool_examples = {
        "📊 Math Tools": [
            "What is 15 multiplied by 8?",
            "Calculate (5 + 3) × 2",
            "What's 2 to the power of 10?"
        ],
        "📝 Text Tools": [
            "How many words are in 'Hello beautiful world'?",
            "What is 'hello' spelled backwards?",
            "Is 'racecar' a palindrome?"
        ],
        "🕒 Utility Tools": [
            "What time is it right now?",
            "Tell me a random interesting fact",
            "Generate a random number between 1 and 100"
        ]
    }

    for category, examples in tool_examples.items():
        with st.expander(category):
            for example in examples:
                if st.button(f"Try: '{example}'", key=f"tool_{example}"):
                    with st.spinner("Running tool agent..."):
                        try:
                            agent = create_tutorial_agent("tools")
                            result = agent.run_sync(example)
                            st.success(f"**Result:** {result.output}")
                        except Exception as e:
                            st.error(f"Error: {e}")

    # Custom question
    st.subheader("🎯 Try Your Own Questions")
    custom_question = st.text_input(
        "Ask the tool agent anything:",
        placeholder="e.g., 'What time is it and give me a fun fact?'"
    )

    if custom_question and st.button("🚀 Ask Tool Agent"):
        with st.spinner("Processing..."):
            try:
                agent = create_tutorial_agent("tools")
                result = agent.run_sync(custom_question)
                st.success(f"**Response:** {result.output}")
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("""
    ---
    ### Tool Combination

    Notice how the agent can use **multiple tools** in a single response! Try asking:
    - "What time is it and what's 5 × 7?"
    - "Tell me a fact and count words in it"
    - "What's 'hello' backwards and is it a palindrome?"

    This shows the **ROUTING** component deciding which tools to use and when.
    """)


if __name__ == "__main__":
    main()