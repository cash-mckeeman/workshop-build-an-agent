"""
Basic Agent Implementation

This module demonstrates building a simple AI agent from scratch using the
four core components: Model, Tools, Memory, and Routing.

Based on the intro_to_agents.answers.ipynb notebook but refactored to use
Hugging Face models instead of NVIDIA endpoints.
"""

import json
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from .hf_client import get_hf_client, call_llm
from .tools import get_tools_schema, parse_tool_call, execute_tool, create_tool_result_message


class BasicAgent:
    """
    A simple AI agent that demonstrates the four core components:
    1. Model - HuggingFace LLM via ChatHuggingFace
    2. Tools - Simple math functions (add, multiply, subtract)
    3. Memory - Conversation history as a list of messages
    4. Routing - Manual routing logic for tool calls
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        temperature: float = 0.1,  # Lower temperature for more consistent tool usage
        max_new_tokens: int = 256  # Shorter responses for cleaner output
    ):
        """
        Initialize the basic agent.

        Args:
            model_name: HuggingFace model to use
            temperature: Sampling temperature for generation
            max_new_tokens: Maximum tokens to generate
        """
        # Component 1: Model
        self.model = get_hf_client(
            model_name=model_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )

        # Component 2: Tools
        self.tools_schema = get_tools_schema()

        # Component 3: Memory (starts empty)
        self.memory: List[Dict[str, Any]] = []

        print(f"🤖 Basic Agent initialized with model: {model_name}")
        print(f"🔧 Available tools: {[tool['function']['name'] for tool in self.tools_schema]}")

    def add_user_message(self, content: str):
        """Add a user message to memory."""
        message = {"role": "user", "content": content}
        self.memory.append(message)
        print(f"👤 User: {content}")

    def process_response(self) -> Dict[str, Any]:
        """
        Process the current conversation and generate a response.
        This demonstrates Component 4: Routing.

        Returns:
            The LLM's response (either content or tool calls)
        """
        # Call the model with current memory and available tools
        response = call_llm(self.model, self.memory, self.tools_schema)
        self.memory.append(response)

        return response

    def handle_tool_calls(self, response: Dict[str, Any]) -> bool:
        """
        Handle tool calls from the LLM response.

        Args:
            response: LLM response containing tool calls

        Returns:
            True if tool calls were processed, False otherwise
        """
        if "tool_calls" not in response or not response["tool_calls"]:
            return False

        print("🔧 Model requested tool execution...")

        # Parse and execute each tool call
        for i, tool_call in enumerate(response["tool_calls"]):
            try:
                tool_name = tool_call["function"]["name"]
                tool_id = tool_call["id"]
                tool_args_str = tool_call["function"]["arguments"]

                # Parse arguments
                if isinstance(tool_args_str, str):
                    try:
                        # Try JSON first (standard format)
                        tool_args = json.loads(tool_args_str)
                    except json.JSONDecodeError:
                        try:
                            # Fallback to eval for Python dict format
                            tool_args = eval(tool_args_str)
                        except:
                            raise ValueError(f"Could not parse arguments: {tool_args_str}")
                else:
                    tool_args = tool_args_str

                print(f"  📋 Tool {i+1}: {tool_name}({tool_args})")

                # Execute the tool
                tool_output = execute_tool(tool_name, tool_args)
                print(f"  ✅ Result: {tool_output}")

                # Add tool result to memory
                tool_result = create_tool_result_message(tool_id, tool_name, tool_output)
                self.memory.append(tool_result)

            except Exception as e:
                print(f"  ❌ Error executing tool: {e}")
                # Add error message to memory
                error_result = create_tool_result_message(
                    tool_call["id"],
                    tool_call["function"]["name"],
                    f"Error: {str(e)}"
                )
                self.memory.append(error_result)

        return True

    def chat(self, user_input: str) -> str:
        """
        Complete chat interaction: user input -> model response -> tool execution -> final response.

        Args:
            user_input: User's message

        Returns:
            Final response content from the agent
        """
        # Add user message to memory
        self.add_user_message(user_input)

        # Get initial model response
        response = self.process_response()

        # Check if model wants to use tools
        if self.handle_tool_calls(response):
            # If tools were used, provide a simple response based on the tool result
            print("🤖 Generating response from tool result...")
            if self.memory and self.memory[-1].get("role") == "tool":
                tool_result = self.memory[-1].get("content", "")
                tool_name = self.memory[-1].get("name", "calculation")

                # Extract the original question for better responses
                user_message = next((msg for msg in self.memory if msg.get("role") == "user"), {})
                user_content = user_message.get("content", "").lower()

                if tool_name == "add" or "plus" in user_content or "add" in user_content:
                    final_content = f"The answer is {tool_result}."
                elif tool_name == "multiply" or "multiply" in user_content or "times" in user_content:
                    final_content = f"The product is {tool_result}."
                elif tool_name == "subtract" or "minus" in user_content or "subtract" in user_content:
                    final_content = f"The difference is {tool_result}."
                else:
                    final_content = f"The answer is {tool_result}."

                # Add this response to memory
                final_response = {"role": "assistant", "content": final_content}
                self.memory.append(final_response)
            else:
                final_content = "I used a tool but couldn't find the result."
        else:
            # No tools needed, use the original response
            final_content = response.get("content", "")

        print(f"🤖 Assistant: {final_content}")
        return final_content

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the full conversation history."""
        return self.memory.copy()

    def clear_memory(self):
        """Clear the conversation history."""
        self.memory = []
        print("🧹 Memory cleared")

    def print_memory(self):
        """Print the current conversation history in a readable format."""
        print("\n📚 Conversation History:")
        print("=" * 50)
        for i, message in enumerate(self.memory):
            role = message.get("role", "unknown")
            content = message.get("content", "")

            if role == "user":
                print(f"{i+1}. 👤 User: {content}")
            elif role == "assistant":
                if message.get("tool_calls"):
                    print(f"{i+1}. 🤖 Assistant: [Requested tool execution]")
                    for j, tool_call in enumerate(message["tool_calls"]):
                        print(f"    🔧 Tool {j+1}: {tool_call['function']['name']} -> {tool_call['function']['arguments']}")
                else:
                    print(f"{i+1}. 🤖 Assistant: {content}")
            elif role == "tool":
                tool_name = message.get("name", "unknown")
                print(f"{i+1}. 🔧 Tool ({tool_name}): {content}")
        print("=" * 50)


def demo_basic_agent():
    """
    Demonstrate the basic agent with the classic "What is 3 plus 12?" example
    from the original notebook.
    """
    print("🚀 Starting Basic Agent Demo")
    print("=" * 50)

    # Load environment variables
    load_dotenv("../../../variables.env")

    # Create agent
    agent = BasicAgent()

    # Test the classic example from the notebook
    print("\n🧪 Testing: What is 3 plus 12?")
    response = agent.chat("What is 3 plus 12?")

    # Show the full conversation history
    agent.print_memory()

    print("\n✅ Demo completed!")
    return agent


if __name__ == "__main__":
    demo_basic_agent()