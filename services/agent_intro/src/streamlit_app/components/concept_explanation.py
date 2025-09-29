"""
Concept explanation components for the Streamlit app.
"""

import streamlit as st
from typing import Any, Dict, List, Optional
from streamlit_app.utils.formatters import format_component_explanation


def render_concept_card(title: str, description: str, examples: List[str] = None, icon: str = "📚"):
    """Render expandable concept explanation cards."""
    with st.expander(f"{icon} {title}", expanded=False):
        st.markdown(description)

        if examples:
            st.markdown("**Examples:**")
            for example in examples:
                st.markdown(f"- {example}")


def show_agent_component_overview():
    """Display overview of all four agent components."""
    st.markdown("## The Four Core Components of AI Agents")

    components = [
        {
            "title": "🧠 Model",
            "description": """
            The **"brain"** of your agent - the Large Language Model (LLM) that processes and generates text.

            **Key Features:**
            - Understands natural language input
            - Performs reasoning and decision-making
            - Generates human-like responses
            - Can be local (Ollama, vLLM) or cloud-based (OpenAI, Anthropic)
            """,
            "examples": [
                "GPT-4 for advanced reasoning",
                "Claude for detailed analysis",
                "Llama for local deployment",
                "Mistral for efficiency"
            ]
        },
        {
            "title": "🛠️ Tools",
            "description": """
            **Functions** the agent can execute to interact with the real world.

            **Key Features:**
            - Extend agent capabilities beyond text generation
            - Can be any Python function
            - Automatically integrated with the agent
            - Enable real-world interactions
            """,
            "examples": [
                "Mathematical calculations",
                "Web API calls",
                "Database queries",
                "File operations",
                "System commands"
            ]
        },
        {
            "title": "💾 Memory",
            "description": """
            **Conversation history** and context management system.

            **Key Features:**
            - Maintains conversation continuity
            - Stores previous interactions
            - Enables multi-turn conversations
            - Can include context from external sources
            """,
            "examples": [
                "Chat message history",
                "Previous tool call results",
                "User preferences",
                "Session context",
                "Long-term knowledge"
            ]
        },
        {
            "title": "🚦 Routing",
            "description": """
            **Decision-making logic** that determines what the agent should do next.

            **Key Features:**
            - Decides which tools to use when
            - Manages conversation flow
            - Handles error cases and retries
            - Orchestrates the entire interaction
            """,
            "examples": [
                "Tool selection logic",
                "Conversation flow control",
                "Error handling strategies",
                "Response formatting",
                "Context switching"
            ]
        }
    ]

    for component in components:
        render_concept_card(
            component["title"],
            component["description"],
            component["examples"]
        )


def show_agent_vs_chatbot():
    """Compare agents vs traditional chatbots."""
    st.markdown("## 🤖 Agents vs Traditional Chatbots")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 💬 Traditional Chatbot
        **Can only respond with text**

        **Limitations:**
        - Text responses only
        - No real-world actions
        - Limited problem-solving
        - No tool integration
        - Basic conversation flow

        **Example Interaction:**
        > User: "What's 127 × 89?"
        >
        > Bot: "I estimate it's around 11,000"
        """)

    with col2:
        st.markdown("""
        ### 🤖 AI Agent
        **Can take actions in the real world**

        **Capabilities:**
        - Execute functions and tools
        - Perform calculations exactly
        - Access external systems
        - Chain multiple actions
        - Complex problem-solving

        **Example Interaction:**
        > User: "What's 127 × 89?"
        >
        > Agent: *[calls multiply(127, 89)]*
        >
        > "The exact answer is 11,303"
        """)

    st.markdown("---")
    st.info("💡 **The key difference:** Agents can **DO** things, not just **SAY** things!")


def show_component_interaction():
    """Show how the four components work together."""
    st.markdown("## 🔄 How Components Work Together")

    # Create a flow diagram
    steps = [
        {
            "component": "🧠 Model",
            "action": "Understands user input",
            "example": "User asks: 'What is 5 + 3?'"
        },
        {
            "component": "🚦 Routing",
            "action": "Decides what to do",
            "example": "Determines: need to use math tool"
        },
        {
            "component": "🛠️ Tools",
            "action": "Executes the action",
            "example": "Calls: add(5, 3) → returns 8"
        },
        {
            "component": "💾 Memory",
            "action": "Stores the interaction",
            "example": "Saves: Q: '5+3?' A: '8'"
        },
        {
            "component": "🧠 Model",
            "action": "Generates response",
            "example": "Says: 'The answer is 8'"
        }
    ]

    for i, step in enumerate(steps, 1):
        col1, col2, col3 = st.columns([1, 2, 3])

        with col1:
            st.markdown(f"**Step {i}**")

        with col2:
            st.markdown(f"{step['component']}")

        with col3:
            st.markdown(f"{step['action']}: *{step['example']}*")

        if i < len(steps):
            st.markdown("↓")

    st.success("🎉 **Result:** The agent successfully answered the question using all four components!")


def show_progressive_complexity():
    """Show how agents can be built with increasing complexity."""
    st.markdown("## 📈 Progressive Agent Development")

    levels = [
        {
            "level": "Level 1: Basic Agent",
            "components": ["🧠 Model"],
            "description": "Just a language model - can chat but can't take actions",
            "example": "A simple chatbot that only responds with text"
        },
        {
            "level": "Level 2: Tool-Enabled Agent",
            "components": ["🧠 Model", "🛠️ Tools"],
            "description": "Can use functions to perform calculations and actions",
            "example": "Can do math, make API calls, manipulate data"
        },
        {
            "level": "Level 3: Memory-Aware Agent",
            "components": ["🧠 Model", "🛠️ Tools", "💾 Memory"],
            "description": "Remembers context across conversations",
            "example": "Can reference previous interactions and maintain state"
        },
        {
            "level": "Level 4: Complete Agent",
            "components": ["🧠 Model", "🛠️ Tools", "💾 Memory", "🚦 Routing"],
            "description": "Full decision-making capabilities with sophisticated flow control",
            "example": "Can handle complex multi-step tasks and error recovery"
        }
    ]

    for level in levels:
        with st.expander(level["level"], expanded=False):
            st.markdown(f"**Components:** {' + '.join(level['components'])}")
            st.markdown(f"**Description:** {level['description']}")
            st.markdown(f"**Example:** {level['example']}")


def show_real_world_examples():
    """Show real-world examples of agent applications."""
    st.markdown("## 🌍 Real-World Agent Applications")

    examples = [
        {
            "title": "🏥 Medical Assistant",
            "description": "Helps doctors with diagnosis and treatment recommendations",
            "tools": ["Medical database lookup", "Symptom analysis", "Drug interaction checker"],
            "scenario": "Doctor inputs symptoms → Agent searches medical literature → Suggests potential diagnoses"
        },
        {
            "title": "📊 Data Analyst",
            "description": "Automatically analyzes data and generates insights",
            "tools": ["SQL queries", "Data visualization", "Statistical analysis"],
            "scenario": "User uploads data → Agent explores patterns → Creates charts and reports"
        },
        {
            "title": "🛒 Shopping Assistant",
            "description": "Helps users find and purchase products online",
            "tools": ["Price comparison", "Review analysis", "Inventory checking"],
            "scenario": "User describes needs → Agent searches products → Compares options and makes recommendations"
        },
        {
            "title": "📝 Content Creator",
            "description": "Creates and manages content across multiple platforms",
            "tools": ["Text generation", "Image creation", "Social media posting"],
            "scenario": "User provides topic → Agent creates content → Publishes across platforms"
        }
    ]

    for example in examples:
        with st.expander(f"{example['title']}", expanded=False):
            st.markdown(f"**Purpose:** {example['description']}")
            st.markdown("**Tools Used:**")
            for tool in example['tools']:
                st.markdown(f"- {tool}")
            st.markdown(f"**Example Workflow:** {example['scenario']}")


def show_tutorial_roadmap():
    """Show the tutorial learning roadmap."""
    st.markdown("## 🗺️ Your Learning Journey")

    pages = [
        {"name": "Welcome", "icon": "👋", "description": "Introduction and overview"},
        {"name": "Model Setup", "icon": "⚙️", "description": "Configure your AI model provider"},
        {"name": "Tools", "icon": "🛠️", "description": "Learn to create and use tools"},
        {"name": "Memory", "icon": "💾", "description": "Understand conversation memory"},
        {"name": "Routing", "icon": "🚦", "description": "Master decision-making logic"},
        {"name": "Complete Agent", "icon": "🤖", "description": "Build a full-featured agent"}
    ]

    st.markdown("**Tutorial Pages:**")
    for i, page in enumerate(pages, 1):
        st.markdown(f"{i}. {page['icon']} **{page['name']}** - {page['description']}")

    st.info("💡 Each page builds on the previous one. Complete them in order for the best learning experience!")