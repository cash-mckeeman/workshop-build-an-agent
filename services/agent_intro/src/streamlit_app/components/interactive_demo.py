"""
Interactive demo components for the Streamlit app.
"""

import streamlit as st
from typing import Any, Dict, List, Optional
from streamlit_app.utils.session_state import (
    add_message,
    get_messages,
    clear_messages,
    initialize_session_state
)
from streamlit_app.utils.formatters import format_agent_response, format_error_message
from tools.agent_helpers import analyze_agent_execution, get_agent_capabilities


def create_live_agent_chat(agent_type: str = "basic"):
    """Create interactive chat interface with the agent."""
    from factory import get_recommended_setup, create_tutorial_agent
    from models import get_available_providers, check_all_providers

    st.subheader(f"💬 Chat with {agent_type.title()} Agent")

    # Check if we have a working setup
    setup = get_recommended_setup()
    if setup["status"] != "ready":
        st.error("Please configure a provider first (see sidebar)")
        return

    # Initialize session state
    initialize_session_state()

    # Display chat history
    for message in get_messages():
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input(f"Ask your {agent_type} agent anything..."):
        # Add user message
        add_message("user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    agent = create_tutorial_agent(agent_type)
                    result = agent.run_sync(prompt)
                    response = result.output

                    st.markdown(response)
                    add_message("assistant", response)

                except Exception as e:
                    error_msg = format_error_message(e)
                    st.error(error_msg)
                    add_message("assistant", error_msg)

    # Control buttons
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🗑️ Clear Chat", key=f"clear_{agent_type}"):
            clear_messages()
            st.rerun()


def tool_execution_sandbox():
    """Sandbox for testing tool implementations."""
    st.subheader("🧪 Tool Testing Sandbox")

    # Tool selection
    tool_type = st.selectbox(
        "Select Tool Type",
        ["Math Tools", "Text Tools", "Utility Tools"],
        help="Choose which type of tools to test"
    )

    if tool_type == "Math Tools":
        math_tool_sandbox()
    elif tool_type == "Text Tools":
        text_tool_sandbox()
    else:
        utility_tool_sandbox()


def math_tool_sandbox():
    """Sandbox for testing math tools."""
    st.markdown("**Available Math Tools:**")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Addition Tool**")
        a1 = st.number_input("First number", value=5, key="add_a")
        b1 = st.number_input("Second number", value=3, key="add_b")

        if st.button("Test Add", key="test_add"):
            result = a1 + b1
            st.success(f"add({a1}, {b1}) = {result}")

    with col2:
        st.markdown("**Multiplication Tool**")
        a2 = st.number_input("First number", value=4, key="mult_a")
        b2 = st.number_input("Second number", value=7, key="mult_b")

        if st.button("Test Multiply", key="test_mult"):
            result = a2 * b2
            st.success(f"multiply({a2}, {b2}) = {result}")

    # Show how these would be used in an agent
    st.markdown("---")
    st.markdown("**How these tools work in an agent:**")
    st.code("""
@agent.tool
def add(a: int, b: int) -> int:
    \"\"\"Add two numbers together.\"\"\"
    return a + b

@agent.tool
def multiply(a: int, b: int) -> int:
    \"\"\"Multiply two numbers together.\"\"\"
    return a * b

# Agent automatically calls the right tool
result = agent.run_sync("What is 5 plus 3?")
# Agent calls add(5, 3) and responds: "The answer is 8"
    """, language="python")


def text_tool_sandbox():
    """Sandbox for testing text tools."""
    st.markdown("**Available Text Tools:**")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Word Count Tool**")
        text1 = st.text_input("Enter text", value="Hello beautiful world", key="wordcount_text")

        if st.button("Count Words", key="test_wordcount"):
            count = len(text1.split())
            st.success(f"word_count('{text1}') = {count} words")

    with col2:
        st.markdown("**Reverse Text Tool**")
        text2 = st.text_input("Enter text", value="hello", key="reverse_text")

        if st.button("Reverse Text", key="test_reverse"):
            reversed_text = text2[::-1]
            st.success(f"reverse('{text2}') = '{reversed_text}'")

    # Palindrome checker
    st.markdown("**Palindrome Checker Tool**")
    text3 = st.text_input("Enter text", value="racecar", key="palindrome_text")

    if st.button("Check Palindrome", key="test_palindrome"):
        clean_text = text3.lower().replace(" ", "")
        is_palindrome = clean_text == clean_text[::-1]
        st.success(f"is_palindrome('{text3}') = {is_palindrome}")


def utility_tool_sandbox():
    """Sandbox for testing utility tools."""
    st.markdown("**Available Utility Tools:**")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Current Time Tool**")
        if st.button("Get Current Time", key="test_time"):
            import datetime
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.success(f"current_time() = {current_time}")

    with col2:
        st.markdown("**Random Number Tool**")
        min_val = st.number_input("Minimum", value=1, key="random_min")
        max_val = st.number_input("Maximum", value=100, key="random_max")

        if st.button("Generate Random", key="test_random"):
            import random
            result = random.randint(int(min_val), int(max_val))
            st.success(f"random_number({min_val}, {max_val}) = {result}")

    # Fun fact tool
    st.markdown("**Fun Fact Tool**")
    if st.button("Get Fun Fact", key="test_fact"):
        facts = [
            "Octopuses have three hearts!",
            "Bananas are berries, but strawberries aren't.",
            "A group of flamingos is called a 'flamboyance'.",
            "Honey never spoils - archaeologists have found edible honey in ancient Egyptian tombs.",
            "A cloud can weigh more than a million pounds!"
        ]
        import random
        fact = random.choice(facts)
        st.success(f"fun_fact() = {fact}")


def agent_comparison_demo():
    """Demo comparing different agent types."""
    st.subheader("🔍 Agent Comparison Demo")

    # Test question selector
    test_questions = [
        "What is 15 multiplied by 8?",
        "What time is it right now?",
        "How many words are in 'Hello beautiful world'?",
        "Tell me a fun fact and calculate 7 + 3",
        "What's 'hello' spelled backwards?",
        "Is 'racecar' a palindrome?",
        "Get me a random fact"
    ]

    question = st.selectbox(
        "Choose a test question:",
        test_questions,
        help="See how different agent types handle the same question"
    )

    # Debug options (placed before the button to persist state)
    col1, col2 = st.columns(2)
    with col1:
        enable_verbose_debug = st.checkbox("🔬 Enable verbose debugging", key="global_verbose_debug")
    with col2:
        enable_basic_debug = st.checkbox("🔍 Enable basic debugging", key="global_basic_debug")

    if st.button("🚀 Test All Agent Types", key="compare_agents"):
        from factory import get_recommended_setup, create_tutorial_agent
        from models import get_available_providers, check_all_providers

        setup = get_recommended_setup()
        if setup["status"] != "ready":
            st.error("Please configure a provider first")
            return

        # Warn about Ollama limitations
        if setup.get("recommended_provider") == "ollama":
            st.warning(
                "⚠️ **Note**: You're using Ollama, which has limited function calling support. "
                "Agents may appear to use tools but are actually generating responses based on training data. "
                "**Expected behavior with Ollama**: Tools will be called (shown in debug info) but results may not be properly incorporated into responses. "
                "For true tool execution, consider using OpenAI, Anthropic, or other API-based providers."
            )

        agent_types = [
            ("basic", "Basic Agent (MODEL only)"),
            ("math", "Math Agent (MODEL + MATH TOOLS)"),
            ("demo", "Demo Agent (MODEL + DEMO TOOLS)"),
            ("tools", "Tool Agent (MODEL + DEMO TOOLS + MATH TOOLS)")
        ]

        # Store results in session state for persistence
        if "agent_results" not in st.session_state:
            st.session_state.agent_results = {}

        for agent_type, description in agent_types:
            with st.expander(f"🤖 {description}", expanded=True):
                try:
                    agent = create_tutorial_agent(agent_type)
                    with st.spinner(f"Running {agent_type} agent..."):
                        result = agent.run_sync(question)

                    st.markdown(f"**Response:** {result.output}")

                    # Store result for debug display
                    st.session_state.agent_results[agent_type] = {
                        'result': result,
                        'question': question,
                        'description': description
                    }

                    # Use our new diagnostic utility
                    try:
                        from tools.agent_helpers import analyze_agent_execution
                        diagnostics = analyze_agent_execution(result, agent_type, question)

                        # Show execution summary
                        if diagnostics.execution_summary:
                            if "⚠️" in diagnostics.execution_summary:
                                st.warning(diagnostics.execution_summary)
                            elif "🤔" in diagnostics.execution_summary:
                                st.info(diagnostics.execution_summary)
                            elif "✅" in diagnostics.execution_summary:
                                st.success(diagnostics.execution_summary)
                            else:
                                st.info(diagnostics.execution_summary)

                        # Show detailed tool call information if tools were executed
                        if diagnostics.has_tools_executed and diagnostics.tool_calls:
                            # Prominently display what tools actually returned
                            st.markdown("**🔧 What the tools actually returned:**")
                            for i, tool_call in enumerate(diagnostics.tool_calls, 1):
                                tool_name = tool_call.tool_name or "Unknown Tool"
                                tool_result = tool_call.result or "No result captured"
                                if tool_result and tool_result != "No result captured":
                                    # Display the actual tool result prominently
                                    st.info(f"**{tool_name}()** returned: {tool_result}")

                                    # Check if agent response matches tool result (for educational purposes)
                                    if setup.get("recommended_provider") == "ollama":
                                        agent_response_lower = result.output.lower()
                                        tool_result_lower = tool_result.lower()

                                        # Simple check if tool result is incorporated
                                        if any(word in agent_response_lower for word in tool_result_lower.split() if len(word) > 3):
                                            st.success("✅ Agent successfully used the tool result!")
                                        else:
                                            st.error(f"❌ **Ollama Limitation**: Agent called the tool but didn't use its result. The agent should have said: '{tool_result}'")

                            with st.expander("🔧 Tool execution details"):
                                for i, tool_call in enumerate(diagnostics.tool_calls, 1):
                                    tool_name = tool_call.tool_name or "Unknown Tool"
                                    tool_result = tool_call.result or "No result captured"
                                    st.text(f"Tool {i}: {tool_name}")
                                    if tool_call.arguments:
                                        st.text(f"  Arguments: {tool_call.arguments}")
                                    st.text(f"  Result: {tool_result[:200]}...")

                        # Show debug info based on global flags
                        if enable_verbose_debug and hasattr(result, 'all_messages'):
                            # Show detailed message parsing with our enhanced parser
                            from tools.agent_helpers import extract_tool_calls_from_messages
                            messages = result.all_messages()
                            st.write("**Detailed Message Analysis:**")
                            with st.expander("📋 Raw message parsing output", expanded=True):
                                import sys
                                import io

                                # Capture print output from debug mode
                                old_stdout = sys.stdout
                                sys.stdout = buffer = io.StringIO()

                                try:
                                    tool_calls_debug = extract_tool_calls_from_messages(messages, debug=True)
                                    debug_output = buffer.getvalue()
                                finally:
                                    sys.stdout = old_stdout

                                st.code(debug_output, language="text")

                        elif enable_basic_debug and agent_type in ["demo", "tools", "math"]:
                            st.write(f"**Debug Info:** {diagnostics.total_messages} total messages, {len(diagnostics.tool_calls)} tool calls detected")
                            if hasattr(result, 'all_messages'):
                                messages = result.all_messages()
                                for i, msg in enumerate(messages):
                                    st.write(f"Message {i}: Type={type(msg)}, Content={str(msg)[:100]}...")

                    except Exception as e:
                        st.error(f"Diagnostic error: {e}")

                    # Show agent capabilities
                    st.info(get_agent_capabilities(agent_type))

                except Exception as e:
                    st.error(f"Error with {agent_type} agent: {str(e)}")

    # Display debug information outside the execution loop (persists when checkboxes change)
    if (enable_verbose_debug or enable_basic_debug) and "agent_results" in st.session_state and st.session_state.agent_results:
        st.markdown("---")
        st.markdown("### 🔍 Debug Information")

        for agent_type, agent_data in st.session_state.agent_results.items():
            result = agent_data['result']
            question = agent_data['question']

            with st.expander(f"🔬 Debug info for {agent_type} agent", expanded=enable_verbose_debug):
                if enable_verbose_debug and hasattr(result, 'all_messages'):
                    from tools.agent_helpers import extract_tool_calls_from_messages
                    messages = result.all_messages()
                    st.write("**Detailed Message Analysis:**")

                    import sys
                    import io

                    # Capture print output from debug mode
                    old_stdout = sys.stdout
                    sys.stdout = buffer = io.StringIO()

                    try:
                        tool_calls_debug = extract_tool_calls_from_messages(messages, debug=True)
                        debug_output = buffer.getvalue()
                    finally:
                        sys.stdout = old_stdout

                    st.code(debug_output, language="text")

                elif enable_basic_debug and agent_type in ["demo", "tools", "math"]:
                    # Quick re-analyze for debug info
                    from tools.agent_helpers import analyze_agent_execution
                    diagnostics = analyze_agent_execution(result, agent_type, question)

                    st.write(f"**Debug Info:** {diagnostics.total_messages} total messages, {len(diagnostics.tool_calls)} tool calls detected")
                    if hasattr(result, 'all_messages'):
                        messages = result.all_messages()
                        for i, msg in enumerate(messages):
                            st.write(f"Message {i}: Type={type(msg)}, Content={str(msg)[:100]}...")


def progressive_agent_builder():
    """Interactive agent building experience."""
    st.subheader("🏗️ Build Your Agent Step by Step")

    # Track building progress
    if "build_progress" not in st.session_state:
        st.session_state.build_progress = 0

    steps = [
        {
            "title": "Step 1: Choose Your Model",
            "description": "Select the AI model that will power your agent",
            "component": "🧠 MODEL"
        },
        {
            "title": "Step 2: Add Tools",
            "description": "Give your agent the ability to perform actions",
            "component": "🛠️ TOOLS"
        },
        {
            "title": "Step 3: Configure Memory",
            "description": "Enable conversation history and context",
            "component": "💾 MEMORY"
        },
        {
            "title": "Step 4: Set Up Routing",
            "description": "Define decision-making logic",
            "component": "🚦 ROUTING"
        }
    ]

    # Progress indicator
    progress = st.session_state.build_progress / len(steps)
    st.progress(progress)
    st.markdown(f"Progress: {st.session_state.build_progress}/{len(steps)} steps completed")

    # Current step
    if st.session_state.build_progress < len(steps):
        current_step = steps[st.session_state.build_progress]
        st.markdown(f"### {current_step['title']}")
        st.markdown(current_step['description'])

        if st.button(f"✅ Complete {current_step['component']}", key=f"build_step_{st.session_state.build_progress}"):
            st.session_state.build_progress += 1
            st.success(f"Added {current_step['component']} to your agent!")
            st.rerun()

    else:
        st.success("🎉 **Congratulations!** You've built a complete AI agent!")
        st.markdown("""
        Your agent now has all four core components:
        - 🧠 **MODEL**: The AI brain
        - 🛠️ **TOOLS**: Action capabilities
        - 💾 **MEMORY**: Conversation context
        - 🚦 **ROUTING**: Decision logic

        Your agent can now understand questions, decide what tools to use, execute actions,
        and remember the conversation!
        """)

        if st.button("🔄 Start Over", key="reset_builder"):
            st.session_state.build_progress = 0
            st.rerun()


def memory_visualization():
    """Visualize how agent memory works."""
    st.subheader("💾 Memory Visualization")

    st.markdown("""
    Agent memory is like a conversation diary that keeps track of everything that happens.
    Let's see how it works:
    """)

    # Simulate a conversation
    conversation = [
        {"role": "user", "content": "Hi, my name is Alice"},
        {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
        {"role": "user", "content": "What's 5 + 3?"},
        {"role": "assistant", "content": "I'll calculate that for you. [uses add tool] 5 + 3 = 8"},
        {"role": "user", "content": "What's my name?"},
        {"role": "assistant", "content": "Your name is Alice, as you told me earlier."}
    ]

    st.markdown("**Example Conversation:**")
    for i, msg in enumerate(conversation, 1):
        role_icon = "👤" if msg["role"] == "user" else "🤖"
        st.markdown(f"{i}. {role_icon} **{msg['role'].title()}:** {msg['content']}")

    st.markdown("---")
    st.markdown("**What the agent remembers:**")
    st.markdown("""
    - 💬 **Conversation turns**: All messages exchanged
    - 👤 **User information**: Name is "Alice"
    - 🔧 **Tool usage**: Used add(5, 3) tool
    - 📝 **Context**: Can reference earlier parts of conversation
    """)

    st.info("💡 This is why the agent can answer 'What's my name?' correctly - it remembers!")


def routing_decision_tree():
    """Visualize agent routing decisions."""
    st.subheader("🚦 Routing Decision Visualization")

    st.markdown("See how an agent decides what to do with different inputs:")

    # Example decision tree
    scenarios = [
        {
            "input": "What is 5 + 3?",
            "analysis": "Contains mathematical operation",
            "decision": "Use math tool (add)",
            "action": "Call add(5, 3)",
            "result": "Return: 8"
        },
        {
            "input": "What time is it?",
            "analysis": "Asking for current time",
            "decision": "Use time tool",
            "action": "Call get_current_time()",
            "result": "Return: current timestamp"
        },
        {
            "input": "Hello, how are you?",
            "analysis": "Casual greeting",
            "decision": "Generate text response",
            "action": "Use language model only",
            "result": "Return: friendly greeting"
        },
        {
            "input": "What's 7 × 4 and what time is it?",
            "analysis": "Multiple requests: math + time",
            "decision": "Use multiple tools",
            "action": "Call multiply(7, 4) AND get_current_time()",
            "result": "Return: both results"
        }
    ]

    for i, scenario in enumerate(scenarios, 1):
        with st.expander(f"Scenario {i}: '{scenario['input']}'", expanded=False):
            st.markdown(f"**🔍 Analysis:** {scenario['analysis']}")
            st.markdown(f"**🚦 Decision:** {scenario['decision']}")
            st.markdown(f"**⚡ Action:** {scenario['action']}")
            st.markdown(f"**📤 Result:** {scenario['result']}")

    st.info("💡 The routing component analyzes each input and decides the best way to respond!")