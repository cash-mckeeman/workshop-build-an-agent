"""
Math tools for PydanticAI agents - Educational examples from intro_to_agents.ipynb

These tools demonstrate how to create functions that agents can use to perform
mathematical operations, mirroring the original tutorial's add() function.
"""

from typing import Annotated
from pydantic_ai import Agent


def add(a: Annotated[int, "First number"], b: Annotated[int, "Second number"]) -> int:
    """Add two integers together.

    This is the same function from the original intro_to_agents.ipynb tutorial,
    demonstrating how simple it is to create tools for PydanticAI agents.

    Args:
        a: First integer to add
        b: Second integer to add

    Returns:
        The sum of a and b
    """
    return a + b


def multiply(a: Annotated[int, "First number"], b: Annotated[int, "Second number"]) -> int:
    """Multiply two integers together.

    An additional math tool to demonstrate multiple tool usage.

    Args:
        a: First integer to multiply
        b: Second integer to multiply

    Returns:
        The product of a and b
    """
    return a * b


def divide(
    a: Annotated[float, "Dividend (number to be divided)"],
    b: Annotated[float, "Divisor (number to divide by)"]
) -> float:
    """Divide one number by another.

    Args:
        a: Number to be divided (dividend)
        b: Number to divide by (divisor)

    Returns:
        The result of a divided by b

    Raises:
        ValueError: If divisor is zero
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


def power(
    base: Annotated[float, "Base number"],
    exponent: Annotated[float, "Exponent"]
) -> float:
    """Raise a number to a power.

    Args:
        base: The base number
        exponent: The exponent to raise the base to

    Returns:
        base raised to the power of exponent
    """
    return base ** exponent


def register_math_tools(agent: Agent) -> Agent:
    """Register all math tools with a PydanticAI agent.

    This is a helper function that registers all the math tools with an agent,
    making it easy to create agents with mathematical capabilities.

    Args:
        agent: The PydanticAI agent to register tools with

    Returns:
        The agent with math tools registered
    """
    # Register each tool with the agent using the @agent.tool decorator pattern
    agent.tool(add)
    agent.tool(multiply)
    agent.tool(divide)
    agent.tool(power)

    return agent


# Example usage for demonstration
def demo_math_tools():
    """Demonstrate the math tools independently (without an agent)."""
    print("Math Tools Demo:")
    print(f"add(3, 12) = {add(3, 12)}")
    print(f"multiply(4, 5) = {multiply(4, 5)}")
    print(f"divide(15, 3) = {divide(15, 3)}")
    print(f"power(2, 3) = {power(2, 3)}")


if __name__ == "__main__":
    demo_math_tools()