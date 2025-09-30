"""
Tool Agent Implementation - Demonstrating Tools Component

This module shows the progression from basic agent to tool-enabled agent,
focusing on the TOOLS component of the four core components.
"""

from pydantic_ai import Agent

from src.tools.agent_helpers import register_multiple_tools
from src.tools.demo_tools import (
    current_time,
    get_random_fact,
    is_palindrome,
    reverse_string,
    word_count,
)
from src.tools.math_tools import add, multiply


def create_tool_agent(
    model_provider: str = "openai:gpt-4o-mini", include_math_tools: bool = False
) -> Agent[None, str]:
    """Create an agent with both demo tools and math tools.

    This demonstrates the TOOLS component by showing how to add
    different types of tools to an agent - both utility tools
    (time, facts, text manipulation) and math tools (arithmetic).

    Args:
        model_provider: Model identifier
        include_math_tools: Whether to include math tools as well

    Returns:
        A configured PydanticAI agent with comprehensive tool access
    """
    tools_description = "various utility tools"
    if include_math_tools:
        tools_description = (
            "comprehensive tools including utility functions and math operations"
        )

    agent = Agent(
        model_provider,
        instructions=(
            f"You are a helpful assistant with access to {tools_description}. "
            "When a user asks a question that requires using a tool, ALWAYS call the appropriate tool function and present EXACTLY what the tool returns. "
            "Available tools include: get_current_time (for current time), get_fun_fact (for random facts), "
            "count_words (to count words in text), reverse_text (to reverse strings), "
            "check_palindrome (to check if text is a palindrome)"
            + (
                ", add (for addition), multiply (for multiplication)"
                if include_math_tools
                else ""
            )
            + ". "
            "CRITICAL: When you get a tool result, present it directly without adding extra information or explanations. "
            "Do not elaborate or add additional context beyond what the tool provides. "
            "Just use the tool and share its exact result with the user."
        ),
    )

    # Define the basic demo tools to register
    basic_tools = {
        "get_current_time": current_time,
        "get_fun_fact": get_random_fact,
        "count_words": word_count,
        "reverse_text": reverse_string,
        "check_palindrome": is_palindrome,
    }

    # Register basic tools
    register_multiple_tools(agent, basic_tools)

    # Optionally add math tools
    if include_math_tools:
        math_tools = {
            "add": add,
            "multiply": multiply,
        }
        register_multiple_tools(agent, math_tools)

    return agent


def demo_tool_progression():
    """Demonstrate the progression from basic agent to tool-enabled agent."""
    print("🔧 Tool Agent Demo - Component Progression")
    print("=" * 60)

    # Show basic agent (no tools)
    print("1️⃣ Basic Agent (MODEL only):")
    basic_agent = Agent(
        "openai:gpt-4o-mini", instructions="You are a helpful assistant."
    )
    result = basic_agent.run_sync("What time is it?")
    print("Question: What time is it?")
    print(f"Response: {result.output}")
    print("❌ Cannot provide actual time - no tools available")
    print()

    # Show tool-enabled agent
    print("2️⃣ Tool-Enabled Agent (MODEL + TOOLS):")
    tool_agent = create_tool_agent()
    result = tool_agent.run_sync("What time is it?")
    print("Question: What time is it?")
    print(f"Response: {result.output}")
    print("✅ Can provide actual time using the get_current_time() tool")
    print()


def demo_various_tools():
    """Demonstrate different types of tools."""
    print("🛠️ Various Tools Demo")
    print("=" * 40)

    agent = create_tool_agent()

    questions_and_tools = [
        ("What time is it right now?", "get_current_time"),
        ("Tell me an interesting fact", "get_fun_fact"),
        ("How many words are in 'Hello world example'?", "count_words"),
        ("What is 'hello' spelled backwards?", "reverse_text"),
        ("Is 'racecar' a palindrome?", "check_palindrome"),
    ]

    for question, expected_tool in questions_and_tools:
        print(f"❓ Question: {question}")
        print(f"🔧 Expected tool: {expected_tool}")
        result = agent.run_sync(question)
        print(f"🤖 Response: {result.output}")
        print()


def demo_tool_combination():
    """Demonstrate using multiple tools in sequence."""
    print("🔗 Tool Combination Demo")
    print("=" * 40)

    agent = create_tool_agent(include_math_tools=True)

    # Complex question requiring multiple tools
    question = (
        "First tell me the current time, then give me a fun fact, "
        "and finally calculate what 15 plus 27 equals."
    )

    print(f"❓ Complex Question: {question}")
    print("🔧 Expected tools: get_current_time, get_fun_fact, add")
    print()

    result = agent.run_sync(question)
    print(f"🤖 Response: {result.output}")


def demo_tool_schemas():
    """Show how PydanticAI automatically generates tool schemas."""
    print("📋 Tool Schemas Demo")
    print("=" * 40)

    _ = create_tool_agent()

    print("Available tools and their descriptions:")
    print("1. get_current_time() -> str")
    print("   - Get the current date and time")
    print()
    print("2. get_fun_fact() -> str")
    print("   - Get a random interesting fact")
    print()
    print("3. count_words(text: str) -> int")
    print("   - Count the number of words in a text")
    print()
    print("4. reverse_text(text: str) -> str")
    print("   - Reverse a string")
    print()
    print("5. check_palindrome(text: str) -> bool")
    print("   - Check if a text is a palindrome")
    print()
    print("🎯 PydanticAI automatically:")
    print("- Generates JSON schemas for each tool")
    print("- Handles function calling protocol")
    print("- Validates inputs and outputs")
    print("- Manages tool execution and responses")


def interactive_tool_agent():
    """Interactive demo with tool-enabled agent."""
    print("🎮 Interactive Tool Agent")
    print("=" * 40)
    print("Ask me questions! I have access to various tools:")
    print("- Current time")
    print("- Random facts")
    print("- Word counting")
    print("- Text reversal")
    print("- Palindrome checking")
    print("- Basic math (if enabled)")
    print()
    print("Type 'quit' to exit")
    print()

    agent = create_tool_agent(include_math_tools=True)

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ["quit", "exit", "bye"]:
                print("👋 Goodbye!")
                break

            if not user_input:
                continue

            result = agent.run_sync(user_input)
            print(f"🤖: {result.output}")
            print()

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print()


if __name__ == "__main__":
    # Show the progression from basic to tool-enabled
    demo_tool_progression()

    print("\n" + "=" * 60 + "\n")

    # Demonstrate various tools
    demo_various_tools()

    print("\n" + "=" * 60 + "\n")

    # Show tool combination
    demo_tool_combination()

    print("\n" + "=" * 60 + "\n")

    # Show tool schemas
    demo_tool_schemas()

    # Uncomment for interactive mode
    # print("\n" + "=" * 60 + "\n")
    # interactive_tool_agent()
