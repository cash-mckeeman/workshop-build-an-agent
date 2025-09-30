"""
Tools page for learning about agent tools and function calling.
"""

import streamlit as st

from src.streamlit_app.components.code_display import (
    display_interactive_code,
)
from src.streamlit_app.components.interactive_demo import (
    agent_comparison_demo,
    tool_execution_sandbox,
)
from src.streamlit_app.utils.session_state import mark_progress


def show_tools_page():
    """Display the tools page content."""
    from src.factory import create_tutorial_agent, get_recommended_setup

    st.header("🛠️ Tools & Function Calling")

    st.markdown("""
    **Tools** are what transform a chatbot into an agent! They give your agent the ability to
    perform actions in the real world, not just generate text responses.
    """)

    # Check provider status
    setup = get_recommended_setup()

    if setup["status"] != "ready":
        st.error("⚠️ Please configure a provider first (go to Model Setup page)")
        return

    # What are tools?
    st.markdown("## 🎯 What Are Tools?")

    st.markdown("""
    Tools are **Python functions** that your agent can call to perform specific tasks.
    Instead of just guessing at math problems, your agent can use an `add()` function
    to get exact results!
    """)

    # The famous example from the tutorial
    st.markdown("### 📚 The Classic Example")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Without Tools (Basic Chatbot)**")
        basic_example = """
# Just a language model
agent = Agent("openai:gpt-4o-mini")

# User asks: "What is 3 plus 12?"
# Agent guesses: "Around 15" ❌
"""
        st.code(basic_example, language="python")

        if st.button("🚫 Try Basic Agent", key="basic_demo"):
            with st.spinner("Running basic agent..."):
                try:
                    agent = create_tutorial_agent("basic")
                    result = agent.run_sync("What is 3 plus 12?")
                    st.write("**Response:**", result.output)
                    st.warning(
                        "⚠️ Cannot provide exact calculation - no tools available!"
                    )
                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        st.markdown("**With Tools (Smart Agent)**")
        tools_example = '''
# Agent with math tools
agent = Agent("openai:gpt-4o-mini")

@agent.tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

# User asks: "What is 3 plus 12?"
# Agent calls add(3, 12) → 15 ✅
'''
        st.code(tools_example, language="python")

        if st.button("✅ Try Math Agent", key="math_demo"):
            with st.spinner("Running math agent..."):
                try:
                    agent = create_tutorial_agent("math")
                    result = agent.run_sync("What is 3 plus 12?")
                    st.write("**Response:**", result.output)
                    st.success("✅ Exact calculation using the add() tool!")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("---")

    # How tools work
    st.markdown("## ⚙️ How Tools Work")

    st.markdown("""
    When you add a tool to an agent, several things happen automatically:

    1. **Function Registration**: The agent learns about your function
    2. **Schema Generation**: Creates a description of inputs/outputs
    3. **Decision Making**: Agent decides when to use each tool
    4. **Execution**: Agent calls the function with appropriate arguments
    5. **Result Integration**: Tool output is incorporated into the response
    """)

    # Tool creation step by step
    st.markdown("### 🔨 Creating Your First Tool")

    tool_steps = [
        {
            "title": "Step 1: Write a Python Function",
            "code": '''
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b
''',
            "explanation": "Start with a regular Python function with type hints and a docstring.",
        },
        {
            "title": "Step 2: Register with Agent",
            "code": '''
from pydantic_ai import Agent

agent = Agent("openai:gpt-4o-mini")

@agent.tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b
''',
            "explanation": "Use the @agent.tool decorator to register your function.",
        },
        {
            "title": "Step 3: Agent Uses Tool Automatically",
            "code": """
# Agent automatically calls tools when needed
result = agent.run_sync("What is 5 + 3?")
print(result.output)
# Output: "The answer is 8" (after calling add(5, 3))
""",
            "explanation": "The agent automatically decides when and how to use your tools!",
        },
    ]

    for i, step in enumerate(tool_steps):
        with st.expander(step["title"], expanded=(i == 0)):
            st.markdown(step["explanation"])
            st.code(step["code"], language="python")

    st.markdown("---")

    # Tool examples
    st.markdown("## 🧰 Common Tool Types")

    tool_categories = [
        {
            "name": "📊 Math Tools",
            "description": "Perform calculations",
            "examples": ["add()", "multiply()", "power()", "factorial()"],
            "use_cases": [
                "Financial calculations",
                "Engineering computations",
                "Data analysis",
            ],
        },
        {
            "name": "🌐 Web Tools",
            "description": "Access the internet",
            "examples": ["search_web()", "fetch_url()", "check_status()"],
            "use_cases": ["Research", "Monitoring", "Data gathering"],
        },
        {
            "name": "📁 File Tools",
            "description": "Work with files",
            "examples": ["read_file()", "write_file()", "list_files()"],
            "use_cases": [
                "Document processing",
                "Data import/export",
                "File management",
            ],
        },
        {
            "name": "🗄️ Database Tools",
            "description": "Query databases",
            "examples": ["run_query()", "get_user()", "update_record()"],
            "use_cases": ["Data retrieval", "Report generation", "User management"],
        },
        {
            "name": "🎨 Creative Tools",
            "description": "Generate content",
            "examples": ["create_image()", "generate_music()", "write_poem()"],
            "use_cases": ["Content creation", "Art generation", "Creative assistance"],
        },
    ]

    for category in tool_categories:
        with st.expander(
            f"{category['name']}: {category['description']}", expanded=False
        ):
            st.markdown(f"**Examples:** {', '.join(category['examples'])}")
            st.markdown(f"**Use Cases:** {', '.join(category['use_cases'])}")

    st.markdown("---")

    # Interactive tool sandbox
    st.markdown("## 🧪 Interactive Tool Sandbox")

    tool_execution_sandbox()

    st.markdown("---")

    # Agent comparison demo
    st.markdown("## 🔍 Agent Comparison Demo")

    agent_comparison_demo()

    st.markdown("---")

    # Advanced tool concepts
    st.markdown("## 🚀 Advanced Tool Concepts")

    advanced_concepts = [
        {
            "title": "Type Safety with Pydantic",
            "description": "Tools automatically validate inputs and outputs",
            "code": '''
from typing import Annotated

@agent.tool
def divide(
    a: Annotated[float, "First number"],
    b: Annotated[float, "Second number (cannot be zero)"]
) -> float:
    """Divide two numbers."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
''',
        },
        {
            "title": "Multiple Tool Usage",
            "description": "Agents can chain multiple tools together",
            "code": """
# Agent can use multiple tools in one response
result = agent.run_sync("What's 5 + 3 multiplied by 2?")
# Agent calls: add(5, 3) → 8, then multiply(8, 2) → 16
""",
        },
        {
            "title": "Error Handling",
            "description": "Tools can handle errors gracefully",
            "code": '''
@agent.tool
def safe_divide(a: float, b: float) -> str:
    """Safely divide two numbers."""
    try:
        if b == 0:
            return "Error: Cannot divide by zero"
        return f"Result: {a / b}"
    except Exception as e:
        return f"Error: {str(e)}"
''',
        },
    ]

    for concept in advanced_concepts:
        with st.expander(concept["title"], expanded=False):
            st.markdown(concept["description"])
            st.code(concept["code"], language="python")

    st.markdown("---")

    # Tool best practices
    st.markdown("## 💡 Tool Best Practices")

    best_practices = [
        "**Clear naming**: Use descriptive function names like `calculate_tax()` not `calc()`",
        "**Type hints**: Always include type annotations for inputs and outputs",
        "**Docstrings**: Write clear descriptions of what the tool does",
        "**Error handling**: Handle edge cases and provide helpful error messages",
        "**Single responsibility**: Each tool should do one thing well",
        "**Input validation**: Validate inputs to prevent errors",
        "**Documentation**: Include examples of how to use the tool",
    ]

    for practice in best_practices:
        st.markdown(f"- {practice}")

    # Real-world example
    st.markdown("---")
    st.markdown("## 🌍 Real-World Example")

    st.markdown("Let's look at a practical tool for weather information:")

    weather_tool_code = '''
import requests
from typing import Optional

@agent.tool
def get_weather(city: str, country: Optional[str] = None) -> str:
    """Get current weather information for a city.

    Args:
        city: Name of the city
        country: Optional country code (e.g., 'US', 'UK')

    Returns:
        Weather description as a string
    """
    try:
        # This would call a real weather API
        if country:
            location = f"{city}, {country}"
        else:
            location = city

        # Simulated API response
        return f"Weather in {location}: 72°F, sunny with light clouds"

    except Exception as e:
        return f"Could not get weather for {location}: {str(e)}"

# Usage example:
# User: "What's the weather in San Francisco?"
# Agent calls: get_weather("San Francisco")
# Agent responds: "The weather in San Francisco is 72°F, sunny with light clouds"
'''

    display_interactive_code(
        weather_tool_code,
        "This weather tool demonstrates real-world agent capabilities:",
        allow_edit=False,
    )

    # Next steps
    st.markdown("---")
    st.markdown("## 🎯 Key Takeaways")

    st.markdown("""
    **What you've learned:**
    - ✅ Tools transform chatbots into capable agents
    - ✅ Tools are just Python functions with decorators
    - ✅ Agents automatically decide when to use tools
    - ✅ Type hints and docstrings are important
    - ✅ Tools can be chained together for complex tasks

    **The difference tools make:**
    - 🚫 Without tools: "I think 3 + 12 is about 15"
    - ✅ With tools: "I'll calculate that for you. 3 + 12 = 15"

    Tools are what make agents **do** things instead of just **say** things!
    """)

    # Progress button
    col1, col2, col3 = st.columns([2, 1, 2])

    with col2:
        if st.button("✅ I Understand Tools!", key="understand_tools", type="primary"):
            mark_progress("tools", True)
            st.success(
                "Excellent! Next, learn how agents remember conversations with **Memory**!"
            )

    st.markdown("---")

    st.info("""
    🚀 **Next up:** Learn about **Memory** - how agents remember previous conversations
    and maintain context across multiple interactions!
    """)


if __name__ == "__main__":
    show_tools_page()
