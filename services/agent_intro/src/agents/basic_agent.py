"""
Basic Agent Implementation using PydanticAI

This module provides the simplest possible agent implementation,
corresponding to the basic concepts from intro_to_agents.ipynb.
"""

from pydantic_ai import Agent


def create_basic_agent(
    model_provider: str = "openai:gpt-4o-mini", instructions: str | None = None
) -> Agent[None, str]:
    """Create a basic PydanticAI agent for tutorial introduction.

    This demonstrates the first core component of agents: MODEL.
    We create an agent with a model and basic instructions, no tools yet.

    Args:
        model_provider: Model identifier (e.g., "openai:gpt-4o-mini")
        instructions: Custom instructions for the agent

    Returns:
        A configured PydanticAI agent
    """
    default_instructions = (
        "You are a friendly AI assistant helping users learn about agents. "
        "Be concise, educational, and helpful. Explain concepts clearly and "
        "provide practical examples when appropriate."
    )

    return Agent(model_provider, instructions=instructions or default_instructions)


async def demo_basic_agent():
    """Demonstrate the basic agent functionality.

    This shows the same progression as the original tutorial:
    1. Create an agent (MODEL component)
    2. Run a simple query
    3. Show the response

    This demonstrates agents without tools, focusing on the MODEL component.
    """
    print("🤖 Basic Agent Demo")
    print("=" * 50)

    # Create a basic agent (like the original tutorial setup)
    agent = create_basic_agent()

    print("Creating a basic agent...")
    print(f"Model: {agent.model}")
    print(f"Instructions: {agent.instructions}")
    print()

    # Test with a simple question (no tools needed)
    question = "What is an AI agent?"
    print(f"❓ Question: {question}")
    print()

    # Run the agent synchronously (simpler for demo)
    result = agent.run_sync(question)

    print("✅ Agent Response:")
    print(result.output)
    print()

    # Show that this is just the beginning - no tools, no complex memory yet
    print("📝 What we've demonstrated:")
    print("- MODEL: PydanticAI agent with a specific model")
    print("- Simple instructions/system prompt")
    print("- Basic question-answer capability")
    print("- No tools (yet!)")
    print("- No complex memory management (yet!)")


def demo_basic_agent_sync():
    """Synchronous version of the demo for easier execution."""
    print("🤖 Basic Agent Demo (Sync)")
    print("=" * 50)

    # Create a basic agent
    agent = create_basic_agent()

    # Ask a simple question
    question = "Explain what an AI agent is in one sentence."
    print(f"❓ Question: {question}")

    # Get response
    result = agent.run_sync(question)
    print(f"🤖 Response: {result.output}")

    # Ask about the four components
    question2 = "What are the four core components of an AI agent?"
    print(f"\n❓ Question: {question2}")
    result2 = agent.run_sync(question2)
    print(f"🤖 Response: {result2.output}")


def interactive_basic_agent():
    """Interactive demo allowing user to chat with the basic agent."""
    print("🤖 Interactive Basic Agent Chat")
    print("=" * 50)
    print("Type 'quit' to exit")
    print()

    agent = create_basic_agent()

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


# Comparison with different model providers
def compare_model_providers():
    """Demonstrate the same agent with different model providers."""
    print("🔄 Model Provider Comparison")
    print("=" * 50)

    providers = [
        "openai:gpt-4o-mini",
        # "anthropic:claude-3-haiku-20240307",  # Uncomment if you have Anthropic API key
        # "groq:llama3-8b-8192",               # Uncomment if you have Groq API key
        # "ollama:llama3.2",                   # Uncomment if you have Ollama running
    ]

    question = "What makes a good AI agent?"

    for provider in providers:
        try:
            print(f"\n🤖 Using {provider}:")
            agent = create_basic_agent(provider)
            result = agent.run_sync(question)
            print(f"Response: {result.output}")
        except Exception as e:
            print(f"❌ Error with {provider}: {e}")


if __name__ == "__main__":
    # Run the basic demo
    demo_basic_agent_sync()

    print("\n" + "=" * 50)

    # Uncomment for interactive mode
    # interactive_basic_agent()

    # Uncomment to compare providers
    # compare_model_providers()
