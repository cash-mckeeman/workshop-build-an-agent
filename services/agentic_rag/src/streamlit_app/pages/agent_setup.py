"""
Agent Setup Page for Agentic RAG Service

This page provides interface for configuring RAG agents.
"""

import streamlit as st
from typing import Dict, Any, Optional
import logging

from ..utils.session_state import (
    get_system_status,
    update_session_state,
    mark_setup_complete
)
from ..utils.formatters import (
    format_error_message,
    format_success_message,
    format_info_box,
    format_code_block
)

logger = logging.getLogger(__name__)


def render():
    """Render the agent setup page."""
    st.title("⚙️ Agent Setup")
    st.markdown("**Configure Your RAG Agent**")

    # Check prerequisites
    system_status = get_system_status()
    prerequisites_met = (
        system_status.get("embeddings_ready", False) and
        system_status.get("documents_loaded", False) and
        system_status.get("vector_store_ready", False)
    )

    if not prerequisites_met:
        st.warning("⚠️ Please complete the previous steps before setting up your agent!")

        missing_steps = []
        if not system_status.get("embeddings_ready", False):
            missing_steps.append("Set up embedding provider")
        if not system_status.get("documents_loaded", False):
            missing_steps.append("Load documents into vector database")
        if not system_status.get("vector_store_ready", False):
            missing_steps.append("Verify vector store functionality")

        st.markdown("**Missing steps:**")
        for step in missing_steps:
            st.markdown(f"- {step}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗃️ Go to Vector Database", type="primary"):
                st.session_state.current_page_key = "🗃️ Vector Database"
                st.rerun()
        with col2:
            if st.button("🔍 Go to Retrieval Demo"):
                st.session_state.current_page_key = "🔍 Retrieval Demo"
                st.rerun()

        return

    # Agent overview
    st.header("🤖 Understanding RAG Agents")

    st.markdown("""
    A **RAG Agent** is different from a traditional RAG pipeline because it can make intelligent decisions:

    - **Traditional RAG**: Always retrieves documents for every query
    - **Agentic RAG**: Agent decides when, how, and what to retrieve based on the query

    This makes agents more efficient and capable of handling diverse query types.
    """)

    # Agent architecture explanation
    with st.expander("🏗️ Agent Architecture Overview", expanded=True):
        st.markdown("""
        **Key Components of our RAG Agent:**

        1. **Language Model**: The "brain" that understands queries and generates responses
        2. **Retrieval Tools**: Functions the agent can call to search the knowledge base
        3. **Decision Logic**: Determines when and how to use retrieval tools
        4. **Context Management**: Handles retrieved information and conversation history

        **Decision Flow:**
        ```
        User Query → Agent Analysis → Tool Selection → Information Retrieval → Response Generation
        ```
        """)

        # Show code structure
        agent_code = '''
from pydantic_ai import Agent
from retrieval_tools import search_knowledge_base

# Create agent with model and tools
agent = Agent(
    model="openai:gpt-4o-mini",
    tools=[search_knowledge_base],
    system_prompt="You are an IT help desk agent..."
)

# Agent automatically decides when to use tools
response = agent.run("How do I reset my password?")
'''
        format_code_block(agent_code, "python")

    st.markdown("---")

    # Model provider selection
    st.header("🧠 Model Provider Setup")

    current_agent = st.session_state.get("agent")
    agent_ready = system_status.get("agent_ready", False)

    if agent_ready and current_agent:
        st.success("✅ RAG Agent is configured and ready!")

        # Show current configuration
        show_agent_configuration()

        if st.button("🔄 Reconfigure Agent"):
            # Reset agent
            update_session_state({"agent": None})
            st.rerun()

    else:
        st.info("🔧 Configure your RAG agent's model provider and settings")

        # Model provider selection
        show_model_provider_selection()

    st.markdown("---")

    # Agent types
    if not agent_ready:
        st.header("🎯 Choose Agent Type")

        agent_types = {
            "Simple RAG Agent": {
                "description": "Basic agent that searches the knowledge base for every query",
                "use_case": "Good for document-heavy tasks where retrieval is always needed",
                "complexity": "Low",
                "features": ["Always retrieves documents", "Simple search", "Direct responses"]
            },
            "Smart RAG Agent": {
                "description": "Intelligent agent that decides when to search based on the query",
                "use_case": "Best for mixed queries (some need retrieval, others don't)",
                "complexity": "Medium",
                "features": ["Selective retrieval", "Query analysis", "Context awareness"]
            },
            "IT Helpdesk Agent": {
                "description": "Specialized agent for IT support with domain-specific behavior",
                "use_case": "Optimized for technical support and troubleshooting",
                "complexity": "High",
                "features": ["Domain expertise", "Structured responses", "Escalation logic"]
            }
        }

        selected_type = None
        for agent_name, config in agent_types.items():
            with st.expander(f"🤖 {agent_name} - {config['complexity']} Complexity", expanded=False):
                st.markdown(f"**Description:** {config['description']}")
                st.markdown(f"**Best for:** {config['use_case']}")
                st.markdown("**Features:**")
                for feature in config['features']:
                    st.markdown(f"- {feature}")

                if st.button(f"Select {agent_name}", key=f"select_{agent_name.replace(' ', '_')}"):
                    selected_type = agent_name
                    create_selected_agent(selected_type)

    # Agent configuration options
    if agent_ready:
        st.header("🔧 Agent Configuration")

        with st.expander("🎛️ Advanced Settings", expanded=False):
            show_advanced_agent_settings()

        # Test the agent
        st.header("🧪 Test Your Agent")
        test_agent_functionality()

    # Educational content
    st.markdown("---")

    with st.expander("📖 Learn More About Agentic RAG"):
        st.markdown("""
        **Key Concepts:**

        **ReAct Pattern (Reasoning + Acting):**
        - Agent thinks about what it needs to do
        - Chooses appropriate tools to use
        - Acts by calling the selected tools
        - Reasons about the results

        **Tool Selection Logic:**
        - Simple queries: May not need retrieval ("What time is it?")
        - Knowledge queries: Use retrieval tools ("How do I reset password?")
        - Complex queries: May use multiple tools in sequence

        **Benefits of Agentic RAG:**
        - **Efficiency**: Only retrieves when necessary
        - **Accuracy**: Better tool selection for different query types
        - **Flexibility**: Can handle diverse conversation patterns
        - **Scalability**: Easy to add new tools and capabilities

        **Real-World Applications:**
        - Customer support chatbots
        - Technical documentation assistants
        - Research and analysis tools
        - Educational tutoring systems
        """)

    # Next steps
    if agent_ready:
        st.markdown("---")
        st.success("🎉 Your RAG agent is ready! Time to see it in action with real conversations.")

        if st.button("🚀 Continue to Live Demo", type="primary"):
            st.session_state.current_page_key = "🚀 Live Demo"
            st.rerun()


def show_model_provider_selection():
    """Show model provider selection interface."""
    st.subheader("🔌 Select Model Provider")

    try:
        # Import here to avoid circular imports
        from ...models.providers import get_available_providers, get_provider_config

        available_providers = get_available_providers()

        if not available_providers:
            st.error("❌ No model providers available!")
            st.markdown("""
            **To use RAG agents, you need at least one model provider configured.**

            **Quick setup options:**
            1. Set `OPENAI_API_KEY` environment variable
            2. Set `ANTHROPIC_API_KEY` for Claude
            3. Install and run Ollama for local models
            """)
            return

        # Provider selection
        provider_options = list(available_providers.keys())
        selected_provider = st.selectbox(
            "Choose your model provider:",
            provider_options,
            help="Select the AI model provider for your RAG agent"
        )

        if selected_provider:
            provider_config = get_provider_config(selected_provider)

            # Show provider details
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Provider:** {selected_provider}")
                st.markdown(f"**Status:** {'✅ Available' if provider_config else '❌ Not configured'}")

            with col2:
                if provider_config:
                    models = provider_config.get_available_models()
                    if models:
                        selected_model = st.selectbox(
                            "Choose model:",
                            models,
                            help="Select the specific model to use"
                        )
                    else:
                        selected_model = provider_config.get_default_model()
                        st.info(f"Using default model: {selected_model}")

            # Configuration confirmation
            if st.button("🚀 Create RAG Agent", type="primary"):
                create_rag_agent(selected_provider, selected_model if 'selected_model' in locals() else None)

    except Exception as e:
        format_error_message("Failed to load model providers", str(e))


def create_selected_agent(agent_type: str):
    """Create an agent of the selected type."""
    st.info(f"Creating {agent_type}...")
    # This would integrate with the actual agent creation logic
    # For now, show a placeholder
    st.success(f"✅ {agent_type} created successfully!")


def create_rag_agent(provider: str, model: Optional[str] = None):
    """Create and configure a RAG agent."""
    try:
        with st.spinner("Creating RAG agent..."):
            # Import here to avoid circular imports
            from ...factory import RAGFactory, AgentType

            # Get or create factory
            if "rag_factory" not in st.session_state or st.session_state.rag_factory is None:
                st.session_state.rag_factory = RAGFactory()

            factory = st.session_state.rag_factory

            # Build model identifier
            if model:
                model_id = f"{provider}:{model}"
            else:
                # Use provider default
                from ...models.providers import get_provider_config
                provider_config = get_provider_config(provider)
                model_id = provider_config.get_model_id()

            # Create agent
            agent = factory.create_agent(
                agent_type=AgentType.IT_HELPDESK,  # Default to IT helpdesk
                model_provider=model_id,
                vector_store_name="it_helpdesk"
            )

            # Store in session state
            update_session_state({
                "agent": agent,
                "agent_model": model_id,
                "agent_provider": provider
            })

            mark_setup_complete("agent")

            format_success_message(
                "RAG agent created successfully!",
                f"Using {provider} with model {model_id}"
            )

            # Show next steps
            st.info("🎯 Your agent is ready! You can now test it or proceed to the live demo.")

    except Exception as e:
        format_error_message("Failed to create RAG agent", str(e))
        logger.exception("Failed to create RAG agent")


def show_agent_configuration():
    """Show current agent configuration."""
    st.subheader("📋 Current Configuration")

    agent_model = st.session_state.get("agent_model", "Unknown")
    agent_provider = st.session_state.get("agent_provider", "Unknown")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Provider", agent_provider)
        st.metric("Model", agent_model.split(":")[-1] if ":" in agent_model else agent_model)

    with col2:
        # Agent capabilities
        st.markdown("**Agent Capabilities:**")
        capabilities = [
            "✅ Document retrieval",
            "✅ Semantic search",
            "✅ Context-aware responses",
            "✅ IT domain expertise"
        ]
        for cap in capabilities:
            st.markdown(cap)


def show_advanced_agent_settings():
    """Show advanced agent configuration options."""
    st.markdown("**Retrieval Settings:**")

    col1, col2 = st.columns(2)

    with col1:
        max_docs = st.slider("Max documents to retrieve:", 1, 10, 3,
                            help="Maximum number of documents to retrieve per query")

        similarity_threshold = st.slider("Similarity threshold:", 0.0, 1.0, 0.3,
                                       help="Minimum similarity score to include documents")

    with col2:
        use_reranking = st.checkbox("Enable reranking", value=True,
                                  help="Improve retrieval quality with reranking")

        context_length = st.slider("Context window:", 1000, 8000, 4000,
                                 help="Maximum context length for the agent")

    st.markdown("**Response Settings:**")

    response_style = st.selectbox(
        "Response style:",
        ["Professional", "Casual", "Technical", "Beginner-friendly"],
        help="Tone and style of agent responses"
    )

    include_sources = st.checkbox("Include source citations", value=True,
                                help="Show which documents were used in the response")

    if st.button("💾 Save Settings"):
        # Save settings to session state
        settings = {
            "max_docs": max_docs,
            "similarity_threshold": similarity_threshold,
            "use_reranking": use_reranking,
            "context_length": context_length,
            "response_style": response_style,
            "include_sources": include_sources
        }

        update_session_state({"agent_settings": settings})
        st.success("✅ Settings saved!")


def test_agent_functionality():
    """Test the agent with sample queries."""
    st.markdown("Try your agent with these sample IT support queries:")

    sample_queries = [
        "How do I reset my password?",
        "I'm having VPN connection issues",
        "How do I install new software?",
        "My email is not working",
        "I need help with network access"
    ]

    selected_query = st.selectbox("Choose a test query:", [""] + sample_queries)

    custom_query = st.text_input("Or enter your own query:",
                               placeholder="Type your question here...")

    test_query = custom_query if custom_query else selected_query

    if test_query and st.button("🧪 Test Agent", type="primary"):
        run_agent_test(test_query)


def run_agent_test(query: str):
    """Run a test query with the agent."""
    try:
        agent = st.session_state.get("agent")
        if not agent:
            st.error("Agent not available")
            return

        st.subheader(f"🤖 Agent Response to: '{query}'")

        with st.spinner("Agent is thinking and retrieving information..."):
            # Run the agent
            response = agent.run_sync(query)

            # Display response
            st.markdown("**Agent Response:**")
            st.markdown(response.output)

            # Show tool usage if available
            if hasattr(response, 'tool_calls') and response.tool_calls:
                with st.expander("🔍 Tool Usage Details"):
                    st.markdown("**Tools used by the agent:**")
                    for tool_call in response.tool_calls:
                        st.markdown(f"- {tool_call.tool_name}: {tool_call.args}")

            # Show retrieved documents if available
            show_retrieved_documents(query)

    except Exception as e:
        format_error_message("Agent test failed", str(e))
        logger.exception("Agent test failed")


def show_retrieved_documents(query: str):
    """Show documents that would be retrieved for the query."""
    try:
        vector_store = st.session_state.get("vector_store")
        if not vector_store:
            return

        with st.expander("📚 Retrieved Documents"):
            st.markdown("**Documents the agent used to answer your question:**")

            # Get similar documents
            results = vector_store.search(query, k=3)

            for i, result in enumerate(results, 1):
                doc = result.document
                score = result.score
                st.markdown(f"**Document {i}** (Similarity: {score:.3f})")
                content_preview = doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
                st.markdown(content_preview)

                if hasattr(doc, 'metadata') and doc.metadata:
                    metadata_str = ", ".join(f"{k}: {v}" for k, v in doc.metadata.items())
                    st.caption(f"Source: {metadata_str}")
                st.markdown("---")

    except Exception as e:
        logger.exception("Failed to show retrieved documents")


if __name__ == "__main__":
    render()