#!/usr/bin/env python3
"""
Command Line Interface for Agent Intro Tutorial

This provides a simple CLI for testing and demonstrating the agent functionality
without needing to set up the full Streamlit interface.
"""

import argparse
import sys
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from factory import (
    AgentFactory,
    get_recommended_setup,
    create_tutorial_agent
)
from models import check_all_providers
from models.health import print_health_status


def cmd_status():
    """Show system status and provider health."""
    print("🔍 Agent Intro System Status")
    print("=" * 50)

    setup = get_recommended_setup()
    print(f"Setup Status: {setup['status']}")

    if setup['status'] == 'ready':
        print(f"✅ Recommended Provider: {setup['recommended_provider']}")
        print(f"✅ Recommended Model: {setup['recommended_model']}")
    else:
        print("❌ No providers available")

    print("\n" + "=" * 50)
    print_health_status()


def cmd_demo(agent_type="math", question=None):
    """Run a demo with the specified agent type."""
    setup = get_recommended_setup()

    if setup['status'] != 'ready':
        print("❌ No providers available. Please configure API keys or start Ollama.")
        return

    print(f"🤖 Running {agent_type} agent demo...")
    print(f"Provider: {setup['recommended_provider']}")
    print(f"Model: {setup['recommended_model']}")
    print()

    # Default questions for each agent type
    default_questions = {
        "basic": "Hello! Can you tell me what you are?",
        "math": "What is 3 plus 12?",
        "tools": "What time is it and what's 5 multiplied by 7?"
    }

    question = question or default_questions.get(agent_type, "Hello!")

    try:
        agent = create_tutorial_agent(agent_type)
        print(f"Question: {question}")
        print("Thinking...", end="", flush=True)

        result = agent.run_sync(question)

        print("\r" + " " * 20 + "\r", end="")  # Clear "Thinking..."
        print(f"🤖 Response: {result.output}")

    except Exception as e:
        print(f"\n❌ Error: {e}")


def cmd_interactive(agent_type="tools"):
    """Start an interactive session with the specified agent."""
    setup = get_recommended_setup()

    if setup['status'] != 'ready':
        print("❌ No providers available. Please configure API keys or start Ollama.")
        return

    print(f"🎮 Interactive {agent_type} Agent Session")
    print(f"Provider: {setup['recommended_provider']} | Model: {setup['recommended_model']}")
    print("Type 'quit', 'exit', or 'bye' to exit\n")

    try:
        agent = create_tutorial_agent(agent_type)
        messages = []

        while True:
            try:
                user_input = input("You: ").strip()

                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye!")
                    break

                if not user_input:
                    continue

                print("🤖 Thinking...", end="", flush=True)
                result = agent.run_sync(user_input, message_history=messages)

                print("\r" + " " * 20 + "\r", end="")  # Clear "Thinking..."
                print(f"🤖: {result.output}\n")

                # Update message history for context
                messages = result.new_messages()

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\r❌ Error: {e}\n")

    except Exception as e:
        print(f"❌ Failed to create agent: {e}")


def cmd_tutorial():
    """Show the original tutorial progression."""
    print("📚 Agent Tutorial Progression")
    print("=" * 50)

    setup = get_recommended_setup()

    if setup['status'] != 'ready':
        print("❌ No providers available. Please configure API keys or start Ollama.")
        print("\nSetup Instructions:")
        print("- Set OPENAI_API_KEY environment variable, or")
        print("- Set ANTHROPIC_API_KEY environment variable, or")
        print("- Install and start Ollama with: ollama serve && ollama pull llama3.2")
        return

    print("Step 1: Basic Agent (MODEL only)")
    print("Question: What is 3 plus 12?")
    try:
        agent = create_tutorial_agent("basic")
        result = agent.run_sync("What is 3 plus 12?")
        print(f"Response: {result.output}")
        print("❌ Cannot provide exact calculation - no tools!")
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "-" * 30 + "\n")

    print("Step 2: Math Agent (MODEL + TOOLS)")
    print("Question: What is 3 plus 12?")
    try:
        agent = create_tutorial_agent("math")
        result = agent.run_sync("What is 3 plus 12?")
        print(f"Response: {result.output}")
        print("✅ Exact calculation using add() tool!")
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "-" * 30 + "\n")

    print("Step 3: Tool Agent (MODEL + TOOLS + more)")
    print("Question: What time is it and what's 'hello' backwards?")
    try:
        agent = create_tutorial_agent("tools")
        result = agent.run_sync("What time is it and what's 'hello' backwards?")
        print(f"Response: {result.output}")
        print("✅ Multiple tools used in one response!")
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Agent Intro Tutorial CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Status command
    subparsers.add_parser('status', help='Show system status and provider health')

    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run a quick demo')
    demo_parser.add_argument('--type', choices=['basic', 'math', 'tools'],
                           default='math', help='Agent type to demo')
    demo_parser.add_argument('--question', help='Custom question to ask')

    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Start interactive session')
    interactive_parser.add_argument('--type', choices=['basic', 'math', 'tools'],
                                   default='tools', help='Agent type for interactive session')

    # Tutorial command
    subparsers.add_parser('tutorial', help='Show the complete tutorial progression')

    # App command
    subparsers.add_parser('app', help='Launch the Streamlit app')

    args = parser.parse_args()

    if args.command == 'status':
        cmd_status()
    elif args.command == 'demo':
        cmd_demo(args.type, args.question)
    elif args.command == 'interactive':
        cmd_interactive(args.type)
    elif args.command == 'tutorial':
        cmd_tutorial()
    elif args.command == 'app':
        import subprocess
        print("🚀 Launching Streamlit app...")
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'src/agent_intro/app.py'])
    else:
        parser.print_help()


if __name__ == "__main__":
    main()