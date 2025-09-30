"""
Basic Agent page for learning how to create your first PydanticAI agent.
"""

import streamlit as st

from src.streamlit_app.components.code_display import (
    display_interactive_code,
)
from src.streamlit_app.utils.session_state import mark_progress


def show_basic_agent_page():
    """Display the basic agent page content."""
    from src.factory import create_tutorial_agent, get_recommended_setup

    st.header("🤖 Creating Your First Agent")

    st.markdown("""
    Now that you've set up your model provider, let's create your first **PydanticAI agent**!

    An agent is like a chatbot with superpowers - it can understand context, follow instructions,
    and eventually use tools to take actions in the real world.
    """)

    # Check provider status
    setup = get_recommended_setup()

    if setup["status"] != "ready":
        st.error("⚠️ Please configure a provider first (go to Model Setup page)")
        return

    # What is an agent?
    st.markdown("## 🎯 What is an Agent?")

    st.markdown("""
    Think of an agent as an intelligent assistant that can:
    - **Understand** what you're asking
    - **Think** about the best way to help
    - **Respond** with helpful information
    - **Take actions** (when given tools)
    - **Remember** previous conversations
    """)

    # Agent vs traditional programming
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Traditional Programming**")
        traditional_code = """
# You write exact instructions
def calculate_tax(amount, rate):
    return amount * rate

# Computer follows exactly
result = calculate_tax(100, 0.08)
# Always returns 8.0
"""
        st.code(traditional_code, language="python")
        st.markdown(
            "✅ **Predictable**: Same input → Same output  \n❌ **Rigid**: Can't handle unexpected requests"
        )

    with col2:
        st.markdown("**AI Agent**")
        agent_code = """
# You give general instructions
agent = Agent("openai:gpt-4o-mini",
    instructions="Help with tax calculations"
)

# Agent interprets requests
result = agent.run_sync(
    "What's 8% tax on $100?"
)
# Returns: "The tax would be $8.00"
"""
        st.code(agent_code, language="python")
        st.markdown(
            "✅ **Flexible**: Handles varied requests  \n✅ **Intelligent**: Understands intent"
        )

    st.markdown("---")

    # Your first agent
    st.markdown("## 🚀 Your First PydanticAI Agent")

    st.markdown("""
    Let's create a simple agent step by step. PydanticAI makes this incredibly easy!
    """)

    agent_steps = [
        {
            "title": "Step 1: Import PydanticAI",
            "code": """
from pydantic_ai import Agent
""",
            "explanation": "PydanticAI is a modern framework that makes building agents simple and type-safe.",
        },
        {
            "title": "Step 2: Create an Agent",
            "code": """
# Create an agent with a model provider
agent = Agent("openai:gpt-4o-mini")

# That's it! You have a working agent.
""",
            "explanation": "The Agent class handles all the complexity of talking to different AI models.",
        },
        {
            "title": "Step 3: Add Instructions (Optional)",
            "code": '''
# Give your agent personality and purpose
agent = Agent(
    "openai:gpt-4o-mini",
    instructions="""
    You are a helpful coding tutor. Explain concepts clearly
    and provide practical examples. Be encouraging and patient.
    """
)
''',
            "explanation": "Instructions guide how your agent behaves and responds to users.",
        },
        {
            "title": "Step 4: Talk to Your Agent",
            "code": """
# Synchronous (blocking) conversation
result = agent.run_sync("Explain what a function is in Python")
print(result.output)

# Asynchronous (non-blocking) conversation
result = await agent.run("What are variables used for?")
print(result.output)
""",
            "explanation": "Use run_sync() for simple scripts or run() for async applications.",
        },
    ]

    for i, step in enumerate(agent_steps):
        with st.expander(step["title"], expanded=(i == 0)):
            st.markdown(step["explanation"])
            st.code(step["code"], language="python")

    st.markdown("---")

    # Complete working example
    st.markdown("## 💻 Complete Working Example")

    complete_example = '''
from pydantic_ai import Agent

# Create a helpful coding tutor agent
coding_tutor = Agent(
    "openai:gpt-4o-mini",
    instructions="""
    You are a patient coding tutor who helps beginners learn programming.
    - Explain concepts in simple terms
    - Provide practical examples
    - Be encouraging and supportive
    - Break down complex topics into steps
    """
)

# Have a conversation
def chat_with_tutor():
    print("🤖 Coding Tutor: Hi! I'm here to help you learn programming.")

    while True:
        question = input("You: ")
        if question.lower() in ['quit', 'exit', 'bye']:
            break

        response = coding_tutor.run_sync(question)
        print(f"🤖 Coding Tutor: {response.output}")

# Start chatting!
if __name__ == "__main__":
    chat_with_tutor()
'''

    display_interactive_code(
        complete_example,
        "This creates a fully functional coding tutor agent in just a few lines!",
        allow_edit=False,
    )

    st.markdown("---")

    # Live demo
    st.markdown("## 🎮 Try Your Agent Live!")

    st.markdown("""
    Let's test a real agent right here! This agent is set up to be a helpful coding tutor.
    """)

    try:
        # Create a tutorial agent
        demo_agent = create_tutorial_agent("basic")

        if "basic_agent_messages" not in st.session_state:
            st.session_state.basic_agent_messages = []

        # Chat interface
        st.markdown("**Chat with your coding tutor agent:**")

        # Display chat history
        for message in st.session_state.basic_agent_messages:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**🤖 Agent:** {message['content']}")

        # Input for new message
        user_input = st.text_input(
            "Ask your coding tutor a question:", key="basic_agent_input"
        )

        if st.button("Send", key="basic_agent_send") and user_input:
            # Add user message
            st.session_state.basic_agent_messages.append(
                {"role": "user", "content": user_input}
            )

            # Get agent response
            with st.spinner("Agent is thinking..."):
                try:
                    result = demo_agent.run_sync(user_input)
                    response = result.output

                    # Add agent response
                    st.session_state.basic_agent_messages.append(
                        {"role": "agent", "content": response}
                    )

                    # Clear input (rerun to show updated chat)
                    st.rerun()

                except Exception as e:
                    st.error(f"Error: {e}")

        # Clear chat button
        if st.button("🗑️ Clear Chat", key="clear_basic_chat"):
            st.session_state.basic_agent_messages = []
            st.rerun()

    except Exception as e:
        st.error(f"Could not create demo agent: {e}")
        st.info(
            "💡 Make sure your provider is configured correctly in the Model Setup page."
        )

    st.markdown("---")

    # Different model providers
    st.markdown("## 🔄 Model Provider Examples")

    st.markdown("""
    PydanticAI works with many different AI providers. Here are some examples:
    """)

    provider_examples = [
        {
            "provider": "OpenAI",
            "code": 'agent = Agent("openai:gpt-4o-mini")',
            "description": "Fast, reliable, great for production",
        },
        {
            "provider": "Anthropic",
            "code": 'agent = Agent("anthropic:claude-3-haiku-20240307")',
            "description": "Excellent reasoning, very safe responses",
        },
        {
            "provider": "HuggingFace",
            "code": 'agent = Agent("huggingface:microsoft/DialoGPT-medium")',
            "description": "Free tier available, many model options",
        },
        {
            "provider": "Ollama (Local)",
            "code": 'agent = Agent("ollama:llama3.2")',
            "description": "Runs on your computer, complete privacy",
        },
        {
            "provider": "Groq",
            "code": 'agent = Agent("groq:llama3-8b-8192")',
            "description": "Extremely fast inference, great for real-time",
        },
    ]

    for example in provider_examples:
        with st.expander(
            f"{example['provider']}: {example['description']}", expanded=False
        ):
            st.code(example["code"], language="python")

    st.markdown("---")

    # Agent personality examples
    st.markdown("## 🎭 Agent Personalities")

    st.markdown("""
    Instructions are powerful! They shape how your agent behaves. Here are some examples:
    """)

    personality_examples = [
        {
            "name": "📚 Patient Teacher",
            "instructions": "You are a patient teacher who explains things simply and checks for understanding. Always encourage questions and provide examples.",
        },
        {
            "name": "🔧 Technical Expert",
            "instructions": "You are a senior software engineer. Provide precise, technical answers with code examples. Focus on best practices and efficiency.",
        },
        {
            "name": "🌟 Cheerful Helper",
            "instructions": "You are an enthusiastic assistant who loves helping people! Use emojis and positive language. Make learning fun and engaging.",
        },
        {
            "name": "📊 Data Analyst",
            "instructions": "You are a data scientist who loves numbers and insights. Always back up claims with data and suggest ways to visualize information.",
        },
        {
            "name": "🎨 Creative Partner",
            "instructions": "You are a creative collaborator who thinks outside the box. Suggest innovative solutions and encourage experimental thinking.",
        },
    ]

    for personality in personality_examples:
        with st.expander(personality["name"], expanded=False):
            st.markdown(f"**Instructions:** {personality['instructions']}")

            example_code = f'''
agent = Agent(
    "openai:gpt-4o-mini",
    instructions="""{personality["instructions"]}"""
)
'''
            st.code(example_code, language="python")

    st.markdown("---")

    # Best practices
    st.markdown("## 💡 Agent Best Practices")

    best_practices = [
        "**Clear instructions**: Be specific about how you want your agent to behave",
        "**Set boundaries**: Tell the agent what it should and shouldn't do",
        "**Define personality**: Decide if you want formal, friendly, technical, or creative responses",
        "**Provide context**: Include relevant background information in instructions",
        "**Test thoroughly**: Try different types of questions to see how your agent responds",
        "**Iterate and improve**: Refine instructions based on real usage",
        "**Handle errors gracefully**: Plan for when things go wrong",
    ]

    for practice in best_practices:
        st.markdown(f"- {practice}")

    st.markdown("---")

    # What's next?
    st.markdown("## 🎯 Key Takeaways")

    st.markdown("""
    **What you've learned:**
    - ✅ Agents are intelligent assistants, not just chatbots
    - ✅ PydanticAI makes creating agents incredibly simple
    - ✅ Instructions shape your agent's personality and behavior
    - ✅ One line of code can create a working agent
    - ✅ Agents work with many different AI providers

    **The magic of agents:**
    - 🧠 They understand context and intent
    - 🔄 They can handle varied, unpredictable requests
    - 🎯 They follow your instructions while being flexible
    - 🚀 They're the foundation for building intelligent systems
    """)

    # Progress button
    col1, col2, col3 = st.columns([2, 1, 2])

    with col2:
        if st.button(
            "✅ I Understand Agents!", key="understand_agents", type="primary"
        ):
            mark_progress("basic_agent", True)
            st.success(
                "Excellent! Next, learn how to give your agents **Tools** to take actions!"
            )

    st.markdown("---")

    st.info("""
    🚀 **Next up:** Learn about **Tools** - how to give your agents the ability to perform
    calculations, search the web, control other software, and much more!
    """)


if __name__ == "__main__":
    show_basic_agent_page()
