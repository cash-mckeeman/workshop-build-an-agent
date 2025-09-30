"""
Demo Agent Implementation - Demonstrating Demo Tools Only

This module shows an agent with only the demo tools (no math tools),
focusing on utility functions like time, facts, text manipulation, etc.
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


def create_demo_agent(model_provider: str = "openai:gpt-4o-mini") -> Agent[None, str]:
    """Create an agent with only demo tools (no math tools).

    This demonstrates the TOOLS component by showing how to add
    utility tools to an agent without math capabilities.

    Args:
        model_provider: Model identifier

    Returns:
        A configured PydanticAI agent with demo tools only
    """
    agent = Agent(
        model_provider,
        instructions=(
            "You are a helpful assistant with access to utility tools. "
            "When users ask for the current time, random facts, word counts, text reversal, or palindrome checks, "
            "ALWAYS use your available tools to provide accurate, real-time information. "
            "IMPORTANT: After calling a tool, you MUST include the tool's result in your response. "
            "Never make up or guess responses - always use the appropriate tool when available. "
            "When you get a tool result, present it clearly to the user. "
            "You cannot perform mathematical calculations."
        ),
    )

    # Define the demo tools to register
    demo_tools = {
        "get_current_time": current_time,
        "get_fun_fact": get_random_fact,
        "count_words": word_count,
        "reverse_text": reverse_string,
        "check_palindrome": is_palindrome,
    }

    # Register demo tools
    register_multiple_tools(agent, demo_tools)

    return agent


def demo_demo_agent():
    """Demonstrate the demo agent capabilities."""
    print("🎭 Demo Agent Demo - Utility Tools Only")
    print("=" * 60)

    agent = create_demo_agent()

    test_questions = [
        ("What time is it right now?", "get_current_time"),
        ("Tell me an interesting fact", "get_fun_fact"),
        ("How many words are in 'Hello beautiful world today'?", "count_words"),
        ("What is 'hello world' spelled backwards?", "reverse_text"),
        ("Is 'madam' a palindrome?", "check_palindrome"),
        ("What is 15 multiplied by 8?", "Should explain no math tools available"),
    ]

    for question, expected_behavior in test_questions:
        print(f"❓ Question: {question}")
        print(f"🔧 Expected: {expected_behavior}")
        result = agent.run_sync(question)
        print(f"🤖 Response: {result.output}")
        print()


if __name__ == "__main__":
    demo_demo_agent()
