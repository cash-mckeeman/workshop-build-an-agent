"""
Math Agent Implementation - Replicating the Original Tutorial

This module replicates the exact workflow from intro_to_agents.ipynb:
- Create an agent with the add() tool
- Ask "What is 3 plus 12?"
- Show the tool calling process
- Demonstrate all four core components: MODEL, TOOLS, MEMORY, ROUTING
"""

from pydantic_ai import Agent
from tools.math_tools import add, multiply
from tools.agent_helpers import register_multiple_tools


def create_math_agent(
    model_provider: str = "openai:gpt-4o-mini",
) -> Agent[None, str]:
    """Create an agent with math tools, replicating the original tutorial.

    This demonstrates all four core components:
    1. MODEL: PydanticAI agent with a specific model
    2. TOOLS: Math functions (add, multiply) as tools
    3. MEMORY: Conversation history managed by PydanticAI
    4. ROUTING: PydanticAI handles tool calling automatically

    Args:
        model_provider: Model identifier (e.g., "openai:gpt-4o-mini")

    Returns:
        A configured PydanticAI agent with math tools
    """
    # Create the agent
    agent = Agent(
        model_provider,
        instructions=(
            "You are a helpful math assistant. When a user asks a mathematical question, "
            "call the appropriate math tool function and return the result. "
            "Available tools: add (for addition), multiply (for multiplication). "
            "Always execute the tool and provide the actual result, not just the tool call specification. "
            "For questions that aren't math-related, explain that you only have math tools available."
        )
    )

    # Register math tools using the helper function
    math_tools = {
        'add': add,
        'multiply': multiply,
    }
    register_multiple_tools(agent, math_tools)

    return agent


def demo_original_tutorial():
    """Demonstrate the exact workflow from intro_to_agents.ipynb.

    This replicates:
    1. Creating an agent with add() tool
    2. Asking "What is 3 plus 12?"
    3. Showing the tool call process
    4. Getting the final answer
    """
    print("🧮 Math Agent Demo - Original Tutorial Recreation")
    print("=" * 60)

    # Step 1: Create the agent (MODEL + TOOLS)
    print("Step 1: Creating agent with math tools...")
    agent = create_math_agent()
    print("✅ Agent created with add() and multiply() tools")
    print()

    # Step 2: Ask the exact question from the tutorial
    question = "What is 3 plus 12?"
    print(f"Step 2: Asking the question: '{question}'")
    print()

    # Step 3: Run the agent and show the process
    print("Step 3: Agent processing (MODEL + TOOLS + MEMORY + ROUTING)...")
    result = agent.run_sync(question)

    print("✅ Agent Response:")
    print(result.output)
    print()

    # Step 4: Show what happened behind the scenes
    print("📋 What happened (the four components):")
    print("1. MODEL: Agent understood the math question")
    print("2. TOOLS: Agent decided to use the add() tool")
    print("3. MEMORY: Agent kept track of the conversation")
    print("4. ROUTING: Agent routed to tool, executed it, and responded")
    print()

    return result


def demo_multi_step_math():
    """Demonstrate multi-step math problems using multiple tools."""
    print("🔢 Multi-Step Math Demo")
    print("=" * 40)

    agent = create_math_agent()

    # Test a problem requiring multiple tools
    question = "What is (3 plus 12) multiplied by 2?"
    print(f"Question: {question}")

    result = agent.run_sync(question)
    print(f"Answer: {result.output}")
    print()

    # Another multi-step problem
    question2 = "If I have 5 groups of 4 items, and then add 7 more items, how many items do I have total?"
    print(f"Question: {question2}")

    result2 = agent.run_sync(question2)
    print(f"Answer: {result2.output}")


def demo_conversation_memory():
    """Demonstrate how the agent maintains conversation memory."""
    print("🧠 Conversation Memory Demo")
    print("=" * 40)

    agent = create_math_agent()

    # Start a conversation
    print("Starting a math conversation...")

    # First calculation
    result1 = agent.run_sync("What is 5 plus 3?")
    print(f"User: What is 5 plus 3?")
    print(f"Agent: {result1.output}")
    print()

    # Continue the conversation - agent should remember context
    result2 = agent.run_sync(
        "Now multiply that result by 4",
        message_history=result1.new_messages()
    )
    print(f"User: Now multiply that result by 4")
    print(f"Agent: {result2.output}")
    print()

    # Show the message history
    print("📝 Message History:")
    for i, message in enumerate(result2.all_messages(), 1):
        role = message.get('role', 'unknown')
        content = message.get('content', 'N/A')
        if content:
            print(f"{i}. {role.capitalize()}: {content}")


def interactive_math_agent():
    """Interactive demo for hands-on exploration."""
    print("🎯 Interactive Math Agent")
    print("=" * 40)
    print("Ask me math questions! I can add and multiply.")
    print("Type 'quit' to exit")
    print()

    agent = create_math_agent()
    messages = []

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break

            if not user_input:
                continue

            # Run with conversation history
            result = agent.run_sync(user_input, message_history=messages)
            print(f"🤖: {result.output}")
            print()

            # Update message history for next interaction
            messages = result.new_messages()

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print()


def show_tool_schemas():
    """Show the tool schemas that PydanticAI generates automatically."""
    print("🔧 Tool Schemas")
    print("=" * 30)

    agent = create_math_agent()

    print("The agent has access to these tools:")
    print("1. add(a: int, b: int) -> int")
    print("   - Adds two integers together")
    print("2. multiply(a: int, b: int) -> int")
    print("   - Multiplies two integers together")
    print()
    print("PydanticAI automatically generates JSON schemas for these tools")
    print("and handles the tool calling protocol with the language model.")


if __name__ == "__main__":
    # Replicate the original tutorial
    demo_original_tutorial()

    print("\n" + "=" * 60 + "\n")

    # Show additional capabilities
    demo_multi_step_math()

    print("\n" + "=" * 60 + "\n")

    # Demonstrate memory
    demo_conversation_memory()

    print("\n" + "=" * 60 + "\n")

    # Show tool information
    show_tool_schemas()

    # Uncomment for interactive mode
    # print("\n" + "=" * 60 + "\n")
    # interactive_math_agent()