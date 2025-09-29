"""
Tool Calling Example

This example demonstrates the agent's ability to use tools for mathematical
calculations, replicating the tutorial from the original notebook.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Import from the package
from agents.basic_agent import create_basic_agent


def tool_calling_demo():
    """Demonstrate tool calling functionality with mathematical operations."""
    print("🔧 Tool Calling Demo")
    print("=" * 40)

    # Load environment variables
    load_dotenv("../../../../variables.env")

    # Create agent optimized for tool calling
    agent = create_basic_agent(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        temperature=0.1,  # Lower temperature for more consistent tool usage
        max_new_tokens=512
    )

    # Test cases that should trigger tool usage
    math_questions = [
        "What is 3 plus 12?",  # Original notebook example
        "Can you multiply 7 by 8?",
        "What's 100 minus 37?",
        "Calculate 15 + 25, then multiply by 2",
        "What is (10 + 5) * (20 - 15)?"
    ]

    for question in math_questions:
        print(f"\n🧮 Testing: {question}")
        try:
            response = agent.chat(question)
            print(f"   📊 Response: {response}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        print("-" * 50)

    # Show final conversation history
    print("\n📚 Full Conversation History:")
    agent.print_memory()

    print("\n✅ Tool calling demo completed!")


def interactive_tool_demo():
    """Interactive demo where user can ask questions."""
    print("🎮 Interactive Tool Calling Demo")
    print("=" * 45)
    print("Ask me math questions! Type 'quit' to exit.")
    print("Available operations: add, multiply, subtract")
    print("-" * 45)

    # Load environment variables
    load_dotenv("../../../../variables.env")

    # Create agent
    agent = create_basic_agent()

    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            response = agent.chat(user_input)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n📚 Final conversation:")
    agent.print_memory()


if __name__ == "__main__":
    tool_calling_demo()

    # Uncomment to run interactive demo
    # print("\n" + "="*60 + "\n")
    # interactive_tool_demo()