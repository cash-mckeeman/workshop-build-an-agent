"""
Live Demo Page for Agentic RAG Service

This page provides an interactive chat interface with the RAG agent.
"""

import streamlit as st
from typing import Dict, Any, List
import logging
import time

from ..utils.session_state import (
    get_system_status,
    add_chat_message,
    get_chat_history,
    clear_chat_history
)
from ..utils.formatters import (
    format_chat_message,
    format_error_message,
    format_success_message
)

logger = logging.getLogger(__name__)


def render():
    """Render the live demo page."""
    st.title("🚀 Live Demo")
    st.markdown("**Interactive Chat with Your RAG Agent**")

    # Check prerequisites
    system_status = get_system_status()
    agent_ready = system_status.get("agent_ready", False)

    if not agent_ready:
        st.warning("⚠️ Please complete the agent setup first!")

        missing_steps = []
        if not system_status.get("embeddings_ready", False):
            missing_steps.append("Set up embedding provider")
        if not system_status.get("documents_loaded", False):
            missing_steps.append("Load documents")
        if not system_status.get("vector_store_ready", False):
            missing_steps.append("Configure vector store")
        if not system_status.get("agent_ready", False):
            missing_steps.append("Create RAG agent")

        st.markdown("**Missing steps:**")
        for step in missing_steps:
            st.markdown(f"- {step}")

        if st.button("⚙️ Go to Agent Setup", type="primary"):
            st.session_state.current_page_key = "⚙️ Agent Setup"
            st.rerun()
        return

    # Welcome message
    st.success("🎉 Your RAG agent is ready! Start chatting to see agentic RAG in action.")

    # Demo instructions
    with st.expander("💡 How to Use This Demo", expanded=True):
        st.markdown("""
        **Try these types of queries to see different agent behaviors:**

        **📚 Knowledge-based queries (will use retrieval):**
        - "How do I reset my password?"
        - "What's the VPN setup procedure?"
        - "I need help installing software"

        **💬 General conversation (may not need retrieval):**
        - "Hello, can you help me?"
        - "What can you assist me with?"
        - "Thank you for your help"

        **🔍 Complex queries (may use multiple tools):**
        - "I'm having network issues and can't access email"
        - "Help me troubleshoot VPN connection problems"

        **Watch for:**
        - 🔍 When the agent searches the knowledge base
        - 📋 Source citations in responses
        - 🧠 How the agent decides when to retrieve information
        """)

    # Chat interface
    render_chat_interface()

    # Agent insights
    st.markdown("---")
    render_agent_insights()

    # Performance metrics
    render_performance_metrics()


def render_chat_interface():
    """Render the main chat interface."""
    st.header("💬 Chat with Your IT Support Agent")

    # Chat history display
    chat_container = st.container()

    with chat_container:
        chat_history = get_chat_history()

        if not chat_history:
            # Show welcome message
            st.markdown("""
            🤖 **IT Support Agent**: Hello! I'm your AI-powered IT support assistant.
            I can help you with:
            - Password resets and account issues
            - VPN and network connectivity
            - Software installation and troubleshooting
            - General IT support questions

            How can I assist you today?
            """)
        else:
            # Display chat history
            for message in chat_history:
                role = message["role"]
                content = message["content"]

                if role == "user":
                    st.markdown(f"👤 **You**: {content}")
                else:
                    st.markdown(f"🤖 **Agent**: {content}")

                # Add some spacing
                st.markdown("<br>", unsafe_allow_html=True)

    # Input area
    render_chat_input()

    # Chat controls
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("🗑️ Clear Chat"):
            clear_chat_history()
            st.rerun()

    with col2:
        if st.button("💾 Save Conversation"):
            save_conversation()

    with col3:
        if st.button("📊 Show Analytics"):
            show_conversation_analytics()


def render_chat_input():
    """Render the chat input area."""
    # Sample questions for quick testing
    st.markdown("**Quick test questions:**")

    sample_questions = [
        "How do I reset my password?",
        "I'm having VPN connection issues",
        "How do I install new software?",
        "What are the wifi setup instructions?",
        "I need help with email configuration"
    ]

    cols = st.columns(len(sample_questions))
    for i, question in enumerate(sample_questions):
        with cols[i]:
            if st.button(f"💡 {question[:20]}...", key=f"sample_{i}",
                        help=question, use_container_width=True):
                process_user_message(question)

    st.markdown("**Or type your own question:**")

    # Text input
    user_input = st.text_input(
        "Your question:",
        placeholder="Type your IT support question here...",
        key="user_input",
        label_visibility="collapsed"
    )

    # Send button
    col1, col2 = st.columns([4, 1])

    with col2:
        send_clicked = st.button("🚀 Send", type="primary", use_container_width=True)

    if (send_clicked and user_input) or (user_input and st.session_state.get("send_message")):
        # Reset the send flag
        st.session_state.send_message = False
        process_user_message(user_input)

    # Handle enter key (simulation)
    if user_input and user_input != st.session_state.get("last_input", ""):
        st.session_state.last_input = user_input


def process_user_message(message: str):
    """Process a user message and get agent response."""
    try:
        # Add user message to history
        add_chat_message("user", message)

        # Get agent
        agent = st.session_state.get("agent")
        if not agent:
            st.error("Agent not available")
            return

        # Show thinking indicator
        with st.spinner("🤖 Agent is thinking and searching..."):
            # Record start time for metrics
            start_time = time.time()

            # Run the agent
            response = agent.run_sync(message)

            # Record response time
            response_time = time.time() - start_time

            # Store metrics
            store_interaction_metrics(message, response, response_time)

            # Debug the response object
            logger.info(f"Response type: {type(response)}")
            logger.info(f"Response attributes: {dir(response)}")
            logger.info(f"Response output: {response.output}")
            if hasattr(response, 'data'):
                logger.info(f"Response data: {response.data}")

            # Add agent response to history
            response_text = response.data if hasattr(response, 'data') else str(response.output)
            add_chat_message("assistant", response_text)

            # Show tool details
            show_tool_details(response)

            # Show retrieval details if available
            show_retrieval_details(message, response)

        # Refresh the page to show new messages
        st.rerun()

    except Exception as e:
        format_error_message("Failed to process message", str(e))
        logger.exception("Failed to process user message")


def show_tool_details(response):
    """Show detailed information about tool calls and returns."""
    try:
        if not hasattr(response, 'all_messages'):
            return

        from ...utils.tool_helpers import extract_tool_calls_from_messages

        messages = response.all_messages()
        tool_calls = extract_tool_calls_from_messages(messages)

        if tool_calls:
            with st.expander("🔧 Tool Details", expanded=False):
                st.markdown("**Tools used by the agent:**")

                for i, tool_call in enumerate(tool_calls, 1):
                    st.markdown(f"**Tool {i}: {tool_call.tool_name or 'Unknown'}**")

                    if tool_call.arguments:
                        st.markdown("*Arguments:*")
                        st.json(tool_call.arguments)

                    if tool_call.result:
                        st.markdown("*Result:*")
                        # Truncate long results for display
                        result_text = tool_call.result
                        if len(result_text) > 500:
                            result_text = result_text[:500] + "..."
                        st.code(result_text, language="json")

                    if tool_call.has_matching_return:
                        st.success("✅ Tool call completed successfully")
                    else:
                        st.warning("⚠️ Tool return not found")

                    st.markdown("---")

    except Exception as e:
        logger.exception("Failed to show tool details")


def show_retrieval_details(query: str, response):
    """Show details about what the agent retrieved."""
    with st.expander("🔍 See What the Agent Retrieved", expanded=False):
        try:
            # Check if agent used retrieval tools
            used_retrieval = hasattr(response, 'tool_calls') and any(
                'search' in call.tool_name.lower() for call in response.tool_calls
            )

            if used_retrieval:
                st.success("✅ Agent used knowledge base search")

                # Show retrieved documents
                vector_store = st.session_state.get("vector_store")
                if vector_store:
                    results = vector_store.search(query, k=3)

                    st.markdown("**Top retrieved documents:**")
                    for i, result in enumerate(results, 1):
                        doc = result.document
                        score = result.score
                        with st.container():
                            st.markdown(f"**Document {i}** (Relevance: {score:.3f})")
                            content_preview = doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
                            st.markdown(content_preview)

                            if hasattr(doc, 'metadata') and doc.metadata:
                                metadata_str = ", ".join(f"{k}: {v}" for k, v in doc.metadata.items())
                                st.caption(f"📄 Source: {metadata_str}")
                            st.markdown("---")
            else:
                st.info("ℹ️ Agent answered without needing to search the knowledge base")
                st.markdown("This query was handled using the agent's general knowledge.")

        except Exception as e:
            st.error(f"Could not show retrieval details: {e}")


def render_agent_insights():
    """Render insights about agent behavior."""
    st.header("🧠 Agent Insights")

    # Agent configuration
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("⚙️ Current Configuration")
        agent_model = st.session_state.get("agent_model", "Unknown")
        agent_provider = st.session_state.get("agent_provider", "Unknown")

        st.markdown(f"**Provider:** {agent_provider}")
        st.markdown(f"**Model:** {agent_model.split(':')[-1] if ':' in agent_model else agent_model}")

        doc_count = st.session_state.get("document_count", 0)
        st.markdown(f"**Knowledge Base:** {doc_count} documents")

    with col2:
        st.subheader("🎯 Agent Capabilities")
        capabilities = [
            "✅ Semantic document search",
            "✅ Context-aware responses",
            "✅ Selective tool usage",
            "✅ Source attribution",
            "✅ IT domain expertise"
        ]
        for capability in capabilities:
            st.markdown(capability)

    # How it works
    with st.expander("🔬 How Agentic RAG Works"):
        st.markdown("""
        **Step-by-step process:**

        1. **Query Analysis**: Agent analyzes your question to understand intent
        2. **Tool Decision**: Agent decides if it needs to search the knowledge base
        3. **Information Retrieval**: If needed, searches for relevant documents
        4. **Context Integration**: Combines retrieved information with general knowledge
        5. **Response Generation**: Creates a comprehensive answer with sources

        **Key Differences from Traditional RAG:**
        - 🎯 **Selective**: Only searches when necessary
        - 🧠 **Intelligent**: Understands query types and responds appropriately
        - 🔄 **Adaptive**: Can use multiple tools in sequence if needed
        - 📋 **Transparent**: Shows you what information was used
        """)


def render_performance_metrics():
    """Render performance metrics for the session."""
    st.header("📊 Performance Metrics")

    metrics = st.session_state.get("interaction_metrics", {})

    if not metrics:
        st.info("💡 Start chatting to see performance metrics!")
        return

    # Overall stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_queries = metrics.get("total_queries", 0)
        st.metric("Total Queries", total_queries)

    with col2:
        retrieval_queries = metrics.get("retrieval_queries", 0)
        st.metric("Used Knowledge Base", retrieval_queries)

    with col3:
        avg_response_time = metrics.get("avg_response_time", 0)
        st.metric("Avg Response Time", f"{avg_response_time:.2f}s")

    with col4:
        if total_queries > 0:
            retrieval_rate = (retrieval_queries / total_queries) * 100
            st.metric("Retrieval Rate", f"{retrieval_rate:.0f}%")

    # Detailed metrics
    with st.expander("📈 Detailed Metrics"):
        query_types = metrics.get("query_types", {})

        if query_types:
            st.markdown("**Query Types:**")
            for query_type, count in query_types.items():
                st.markdown(f"- {query_type}: {count}")

        response_times = metrics.get("response_times", [])
        if response_times:
            st.markdown("**Response Time Distribution:**")
            st.line_chart(response_times)


def store_interaction_metrics(query: str, response, response_time: float):
    """Store metrics for the current interaction."""
    if "interaction_metrics" not in st.session_state:
        st.session_state.interaction_metrics = {
            "total_queries": 0,
            "retrieval_queries": 0,
            "response_times": [],
            "avg_response_time": 0,
            "query_types": {}
        }

    metrics = st.session_state.interaction_metrics

    # Update total queries
    metrics["total_queries"] += 1

    # Check if retrieval was used
    used_retrieval = hasattr(response, 'tool_calls') and any(
        'search' in call.tool_name.lower() for call in response.tool_calls
    )

    if used_retrieval:
        metrics["retrieval_queries"] += 1

    # Update response times
    metrics["response_times"].append(response_time)
    metrics["avg_response_time"] = sum(metrics["response_times"]) / len(metrics["response_times"])

    # Categorize query type (simple heuristic)
    query_lower = query.lower()
    if any(word in query_lower for word in ["how", "what", "why", "when", "where"]):
        query_type = "Question"
    elif any(word in query_lower for word in ["help", "issue", "problem", "trouble"]):
        query_type = "Support Request"
    elif any(word in query_lower for word in ["hello", "hi", "thanks", "thank you"]):
        query_type = "Greeting/Courtesy"
    else:
        query_type = "Other"

    metrics["query_types"][query_type] = metrics["query_types"].get(query_type, 0) + 1

    # Update session state
    st.session_state.interaction_metrics = metrics


def save_conversation():
    """Save the current conversation."""
    chat_history = get_chat_history()

    if not chat_history:
        st.warning("No conversation to save!")
        return

    # Create conversation text
    conversation_text = "# IT Support Conversation\n\n"
    for message in chat_history:
        role = "User" if message["role"] == "user" else "Agent"
        conversation_text += f"**{role}:** {message['content']}\n\n"

    # Add timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conversation_text += f"\n---\n*Conversation saved on {timestamp}*"

    # Provide download
    st.download_button(
        label="💾 Download Conversation",
        data=conversation_text,
        file_name=f"rag_conversation_{timestamp.replace(':', '-').replace(' ', '_')}.md",
        mime="text/markdown"
    )

    st.success("✅ Conversation prepared for download!")


def show_conversation_analytics():
    """Show analytics about the conversation."""
    chat_history = get_chat_history()

    if not chat_history:
        st.info("No conversation data available yet.")
        return

    st.subheader("📊 Conversation Analytics")

    # Basic stats
    user_messages = [msg for msg in chat_history if msg["role"] == "user"]
    agent_messages = [msg for msg in chat_history if msg["role"] == "assistant"]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Messages", len(chat_history))

    with col2:
        st.metric("User Messages", len(user_messages))

    with col3:
        st.metric("Agent Responses", len(agent_messages))

    # Message length analysis
    if user_messages:
        user_lengths = [len(msg["content"]) for msg in user_messages]
        avg_user_length = sum(user_lengths) / len(user_lengths)

        agent_lengths = [len(msg["content"]) for msg in agent_messages]
        avg_agent_length = sum(agent_lengths) / len(agent_lengths)

        st.markdown("**Message Length Analysis:**")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Avg User Message Length", f"{avg_user_length:.0f} chars")

        with col2:
            st.metric("Avg Agent Response Length", f"{avg_agent_length:.0f} chars")


if __name__ == "__main__":
    render()