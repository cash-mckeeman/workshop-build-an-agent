"""
Memory page for learning about agent memory, conversation history, and structured outputs.
"""

import streamlit as st

from src.streamlit_app.components.code_display import (
    display_interactive_code,
)
from src.streamlit_app.utils.session_state import mark_progress


def show_memory_page():
    """Display the memory page content."""
    from src.factory import create_tutorial_agent, get_recommended_setup

    st.header("🧠 Memory & Structured Outputs")

    st.markdown("""
    **Memory** is what transforms agents from simple Q&A bots into intelligent assistants that can:
    - **Remember** previous conversations
    - **Build context** over multiple interactions
    - **Provide structured data** in specific formats
    - **Maintain state** across complex workflows

    This is the third core component that makes agents truly powerful!
    """)

    # Check provider status
    setup = get_recommended_setup()

    if setup["status"] != "ready":
        st.error("⚠️ Please configure a provider first (go to Model Setup page)")
        return

    # What is memory?
    st.markdown("## 🎯 What is Agent Memory?")

    st.markdown("""
    Agent memory isn't like human memory - it's more like a conversation transcript that the agent can read and reference.
    Every time you chat with an agent, it sees the entire conversation history.
    """)

    # Memory vs no memory comparison
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Without Memory (Each message is isolated)**")
        no_memory_example = """
User: "My name is Alex"
Agent: "Nice to meet you!"

User: "What's my name?"
Agent: "I don't know your name."
❌ Agent forgets everything
"""
        st.code(no_memory_example, language="text")

    with col2:
        st.markdown("**With Memory (Conversation context)**")
        memory_example = """
User: "My name is Alex"
Agent: "Nice to meet you, Alex!"

User: "What's my name?"
Agent: "Your name is Alex!"
✅ Agent remembers the conversation
"""
        st.code(memory_example, language="text")

    st.markdown("---")

    # How memory works in PydanticAI
    st.markdown("## ⚙️ How Memory Works in PydanticAI")

    st.markdown("""
    PydanticAI handles memory automatically! Every conversation maintains a history of messages.
    """)

    memory_steps = [
        {
            "title": "Step 1: Start a Conversation",
            "code": """
from pydantic_ai import Agent

agent = Agent("openai:gpt-4o-mini")

# First message starts the conversation
result = agent.run_sync("Hi, I'm working on a Python project")
print(result.output)
# "Hello! I'd be happy to help with your Python project."
""",
            "explanation": "The first message creates a new conversation with message history.",
        },
        {
            "title": "Step 2: Continue the Conversation",
            "code": """
# Agent remembers the previous context
result = agent.run_sync("Can you help me with functions?")
print(result.output)
# "Of course! Since you're working on a Python project,
#  let me explain Python functions..."

# Still remembers
result = agent.run_sync("Show me an example")
print(result.output)
# "Here's a Python function example for your project..."
""",
            "explanation": "Each new message adds to the conversation history that the agent can see.",
        },
        {
            "title": "Step 3: Access Conversation History",
            "code": """
# You can see the full conversation
for message in result.all_messages():
    print(f"{message.role}: {message.content}")

# Output:
# user: Hi, I'm working on a Python project
# assistant: Hello! I'd be happy to help...
# user: Can you help me with functions?
# assistant: Of course! Since you're working...
""",
            "explanation": "PydanticAI keeps track of the entire conversation automatically.",
        },
    ]

    for i, step in enumerate(memory_steps):
        with st.expander(step["title"], expanded=(i == 0)):
            st.markdown(step["explanation"])
            st.code(step["code"], language="python")

    st.markdown("---")

    # Interactive memory demo
    st.markdown("## 🎮 Memory Demo: Teaching Assistant")

    st.markdown("""
    Let's test memory with a teaching assistant agent. Notice how it remembers your learning preferences and builds on previous topics!
    """)

    try:
        # Create a basic agent for memory demonstration
        memory_agent = create_tutorial_agent("basic")

        if "memory_agent_messages" not in st.session_state:
            st.session_state.memory_agent_messages = []

        # Chat interface
        st.markdown("**Try these conversation starters to test memory:**")

        suggestions = [
            "I'm learning Python and prefer visual examples",
            "What are variables?",
            "Can you build on that with a more complex example?",
            "Remind me what we discussed about variables",
        ]

        st.markdown("**Suggested messages:**")
        for suggestion in suggestions:
            if st.button(
                f'💬 "{suggestion}"', key=f"memory_suggestion_{hash(suggestion)}"
            ):
                st.session_state.memory_agent_messages.append(
                    {"role": "user", "content": suggestion}
                )

                with st.spinner("Agent is thinking..."):
                    try:
                        # Build conversation context for the agent
                        context = ""
                        for msg in st.session_state.memory_agent_messages[
                            :-1
                        ]:  # All but the last message
                            context += f"{msg['role']}: {msg['content']}\n"

                        full_prompt = f"Previous conversation:\n{context}\nCurrent message: {suggestion}"
                        result = memory_agent.run_sync(full_prompt)

                        st.session_state.memory_agent_messages.append(
                            {"role": "agent", "content": result.output}
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

        # Display chat history
        if st.session_state.memory_agent_messages:
            st.markdown("**Conversation:**")
            for message in st.session_state.memory_agent_messages:
                if message["role"] == "user":
                    st.markdown(f"**You:** {message['content']}")
                else:
                    st.markdown(f"**🧠 Memory Agent:** {message['content']}")

        # Manual input
        st.markdown("**Or ask your own question:**")
        user_input = st.text_input(
            "Continue the conversation:", key="memory_agent_input"
        )

        if st.button("Send", key="memory_agent_send") and user_input:
            st.session_state.memory_agent_messages.append(
                {"role": "user", "content": user_input}
            )

            with st.spinner("Agent is thinking..."):
                try:
                    # Build conversation context
                    context = ""
                    for msg in st.session_state.memory_agent_messages[:-1]:
                        context += f"{msg['role']}: {msg['content']}\n"

                    full_prompt = f"Previous conversation:\n{context}\nCurrent message: {user_input}"
                    result = memory_agent.run_sync(full_prompt)

                    st.session_state.memory_agent_messages.append(
                        {"role": "agent", "content": result.output}
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

        # Clear chat button
        if st.button("🗑️ Clear Conversation", key="clear_memory_chat"):
            st.session_state.memory_agent_messages = []
            st.rerun()

    except Exception as e:
        st.error(f"Could not create memory agent: {e}")

    st.markdown("---")

    # Structured outputs
    st.markdown("## 📊 Structured Outputs with Pydantic")

    st.markdown("""
    Memory isn't just about conversation history - agents can also return **structured data**
    instead of just text. This is perfect for building applications that need specific data formats.
    """)

    # Pydantic model example
    structured_example = '''
from pydantic import BaseModel
from pydantic_ai import Agent
from typing import List

# Define the structure you want
class MovieRecommendation(BaseModel):
    title: str
    genre: str
    year: int
    rating: float
    reason: str

class MovieList(BaseModel):
    recommendations: List[MovieRecommendation]
    total_count: int

# Create agent that returns structured data
movie_agent = Agent(
    "openai:gpt-4o-mini",
    result_type=MovieList,
    instructions="""
    You are a movie recommendation expert. When users ask for movies,
    return a structured list with exactly the fields specified.
    """
)

# Get structured data instead of text
result = movie_agent.run_sync("Recommend 3 sci-fi movies from the 2020s")

# Access the structured data
movies = result.output
print(f"Found {movies.total_count} recommendations:")
for movie in movies.recommendations:
    print(f"- {movie.title} ({movie.year}): {movie.rating}/10")
    print(f"  Genre: {movie.genre}")
    print(f"  Why: {movie.reason}")
'''

    display_interactive_code(
        structured_example,
        "This agent returns structured data that your code can directly use!",
        allow_edit=False,
    )

    st.markdown("---")

    # Structured output demo
    st.markdown("## 🎬 Try Structured Output Demo")

    st.markdown("""
    This demo shows an agent returning structured data instead of plain text.
    Perfect for building applications that need specific data formats!
    """)

    if st.button("🎬 Get Movie Recommendations", key="movie_demo"):
        with st.spinner("Getting structured movie data..."):
            try:
                # Simulate structured output (in real implementation, this would use result_type)

                # Mock structured response
                mock_response = {
                    "recommendations": [
                        {
                            "title": "Dune",
                            "genre": "Science Fiction",
                            "year": 2021,
                            "rating": 8.0,
                            "reason": "Epic space opera with stunning visuals and complex world-building",
                        },
                        {
                            "title": "Everything Everywhere All at Once",
                            "genre": "Sci-Fi Comedy",
                            "year": 2022,
                            "rating": 8.1,
                            "reason": "Creative multiverse concept with emotional depth and humor",
                        },
                        {
                            "title": "The Tomorrow War",
                            "genre": "Action Sci-Fi",
                            "year": 2021,
                            "rating": 6.5,
                            "reason": "Time travel action with family themes and alien invasion",
                        },
                    ],
                    "total_count": 3,
                }

                st.success("Got structured movie data!")

                # Display as JSON
                st.markdown("**Raw JSON Response:**")
                st.json(mock_response)

                # Display formatted
                st.markdown("**Formatted Output:**")
                for movie in mock_response["recommendations"]:
                    with st.expander(
                        f"{movie['title']} ({movie['year']}) - ⭐ {movie['rating']}/10"
                    ):
                        st.markdown(f"**Genre:** {movie['genre']}")
                        st.markdown(f"**Why recommended:** {movie['reason']}")

            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")

    # Memory patterns
    st.markdown("## 🔄 Common Memory Patterns")

    memory_patterns = [
        {
            "name": "🎓 Learning Assistant",
            "description": "Tracks student progress and adapts teaching style",
            "code": '''
agent = Agent(
    "openai:gpt-4o-mini",
    instructions="""
    You are a personalized tutor. Remember:
    - Student's current skill level
    - Topics they struggle with
    - Their preferred learning style
    - Previous questions and explanations
    Build on previous lessons and adapt your teaching.
    """
)
''',
        },
        {
            "name": "🛒 Shopping Assistant",
            "description": "Remembers preferences and builds a shopping list",
            "code": '''
agent = Agent(
    "openai:gpt-4o-mini",
    instructions="""
    You are a shopping assistant. Remember:
    - Dietary restrictions and preferences
    - Budget constraints
    - Previous purchases
    - Items already discussed
    Help build and refine a shopping list.
    """
)
''',
        },
        {
            "name": "📝 Project Manager",
            "description": "Tracks tasks, deadlines, and project progress",
            "code": '''
agent = Agent(
    "openai:gpt-4o-mini",
    instructions="""
    You are a project manager. Track:
    - Current project status
    - Completed and pending tasks
    - Team member assignments
    - Deadlines and priorities
    Provide updates and suggest next steps.
    """
)
''',
        },
    ]

    for pattern in memory_patterns:
        with st.expander(
            f"{pattern['name']}: {pattern['description']}", expanded=False
        ):
            st.code(pattern["code"], language="python")

    st.markdown("---")

    # Advanced memory concepts
    st.markdown("## 🚀 Advanced Memory Concepts")

    advanced_concepts = [
        {
            "title": "Conversation Dependencies",
            "description": "Pass data between agent calls",
            "code": """
from pydantic_ai import RunContext

@agent.system_prompt
def get_context(ctx: RunContext) -> str:
    user_data = ctx.deps  # Access shared data
    return f"User preferences: {user_data.get('preferences', 'none')}"
""",
        },
        {
            "title": "Memory Persistence",
            "description": "Save conversations to database or files",
            "code": """
# Save conversation history
conversation_history = []
for message in result.all_messages():
    conversation_history.append({
        "role": message.role,
        "content": message.content,
        "timestamp": datetime.now()
    })

# Save to database or file
save_conversation(user_id, conversation_history)
""",
        },
        {
            "title": "Context Window Management",
            "description": "Handle long conversations that exceed model limits",
            "code": '''
def summarize_old_messages(messages):
    """Summarize older messages to save context space"""
    if len(messages) > 20:
        old_messages = messages[:-10]  # Keep recent 10
        summary = summarize_agent.run_sync(
            f"Summarize this conversation: {old_messages}"
        )
        return [summary] + messages[-10:]
    return messages
''',
        },
    ]

    for concept in advanced_concepts:
        with st.expander(concept["title"], expanded=False):
            st.markdown(concept["description"])
            st.code(concept["code"], language="python")

    st.markdown("---")

    # Memory best practices
    st.markdown("## 💡 Memory Best Practices")

    best_practices = [
        "**Start fresh when appropriate**: Some conversations benefit from clean starts",
        "**Summarize long conversations**: Prevent context window overflow in long chats",
        "**Store important data separately**: Don't rely only on conversation memory for critical information",
        "**Design for context**: Write instructions that help agents use memory effectively",
        "**Test memory behavior**: Verify that agents remember and use context appropriately",
        "**Handle memory limits**: Plan for when conversations get too long",
        "**Structure your data**: Use Pydantic models for complex, structured outputs",
    ]

    for practice in best_practices:
        st.markdown(f"- {practice}")

    st.markdown("---")

    # Real-world example
    st.markdown("## 🌍 Real-World Example: Personal Assistant")

    personal_assistant_example = '''
from pydantic import BaseModel
from pydantic_ai import Agent
from typing import List, Optional
from datetime import datetime

class Task(BaseModel):
    description: str
    priority: str  # "high", "medium", "low"
    due_date: Optional[str] = None
    completed: bool = False

class TaskList(BaseModel):
    tasks: List[Task]
    total_tasks: int
    completed_tasks: int

personal_assistant = Agent(
    "openai:gpt-4o-mini",
    result_type=TaskList,
    instructions="""
    You are a personal task assistant. You help users manage their to-do lists.

    Remember throughout the conversation:
    - All tasks mentioned
    - User's priorities and preferences
    - Deadlines and time constraints
    - Completed items

    When returning task lists, include all current tasks with their status.
    Be proactive in suggesting priorities and deadlines.
    """
)

# Example conversation flow:
# User: "I need to prepare for my presentation tomorrow"
# Assistant: Returns TaskList with presentation prep tasks

# User: "I also need to call my dentist this week"
# Assistant: Returns updated TaskList with both presentation AND dentist call

# User: "I finished the slides"
# Assistant: Returns TaskList with slides marked complete
'''

    display_interactive_code(
        personal_assistant_example,
        "This assistant remembers all tasks and maintains state across the conversation:",
        allow_edit=False,
    )

    st.markdown("---")

    # Key takeaways
    st.markdown("## 🎯 Key Takeaways")

    st.markdown("""
    **What you've learned:**
    - ✅ Memory enables agents to maintain conversation context
    - ✅ PydanticAI handles conversation history automatically
    - ✅ Structured outputs provide data in specific formats
    - ✅ Memory transforms simple Q&A into intelligent assistance
    - ✅ Conversation context builds over multiple interactions

    **The power of memory:**
    - 🧠 **Context awareness**: Agents understand the full conversation
    - 📈 **Progressive assistance**: Each interaction builds on previous ones
    - 🎯 **Personalization**: Agents adapt to user preferences over time
    - 📊 **Structured data**: Get specific formats instead of just text
    - 🔄 **Stateful interactions**: Maintain information across conversations
    """)

    # Progress button
    col1, col2, col3 = st.columns([2, 1, 2])

    with col2:
        if st.button(
            "✅ I Understand Memory!", key="understand_memory", type="primary"
        ):
            mark_progress("memory", True)
            st.success(
                "Excellent! Next, learn about **Routing** - advanced agent orchestration!"
            )

    st.markdown("---")

    st.info("""
    🚀 **Next up:** Learn about **Routing** - how to coordinate multiple agents and
    handle complex workflows with intelligent decision-making!
    """)


if __name__ == "__main__":
    show_memory_page()
