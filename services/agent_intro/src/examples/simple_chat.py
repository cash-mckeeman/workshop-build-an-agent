"""
Simple Chat Example

This example demonstrates how to use the BasicAgent for a simple
conversation without tool calling.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Import from the package
from agents.basic_agent import create_basic_agent


def simple_chat_demo():
    """Demonstrate basic chat functionality without tools."""
    print("💬 Simple Chat Demo")
    print("=" * 40)

    # Load environment variables
    load_dotenv("../../../../variables.env")

    # Create agent with a model good for conversation
    agent = create_basic_agent(
        model_name="microsoft/DialoGPT-medium",
        temperature=0.8,  # Higher temperature for more creative responses
        max_new_tokens=256
    )

    # Example conversations
    conversations = [
        "Hello! How are you today?",
        "What's the weather like?",
        "Tell me a joke",
        "What can you help me with?"
    ]

    for question in conversations:
        print(f"\n📝 Testing: {question}")
        response = agent.chat(question)
        print("-" * 30)

    print("\n✅ Simple chat demo completed!")


if __name__ == "__main__":
    simple_chat_demo()