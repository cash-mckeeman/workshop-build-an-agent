"""
Demo tools for PydanticAI agents - Additional examples for educational purposes

These tools provide more diverse examples of agent capabilities beyond math.
"""

import random
from datetime import datetime
from typing import Annotated


def get_random_fact() -> str:
    """Get a random interesting fact.

    Returns:
        A randomly selected interesting fact
    """
    facts = [
        "Octopuses have three hearts and blue blood.",
        "A group of flamingos is called a 'flamboyance'.",
        "Honey never spoils - archaeologists have found edible honey in ancient tombs.",
        "A single cloud can weigh more than a million pounds.",
        "Butterflies taste with their feet.",
        "The shortest war in history lasted only 38-45 minutes.",
        "A day on Venus is longer than its year.",
        "Bananas are berries, but strawberries are not.",
        "There are more possible games of chess than atoms in the observable universe.",
        "Dolphins have names for each other.",
    ]
    return random.choice(facts)


def current_time() -> str:
    """Get the current date and time.

    Returns:
        Current date and time as a formatted string
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def word_count(text: Annotated[str, "Text to count words in"]) -> int:
    """Count the number of words in a text.

    Args:
        text: The text to count words in

    Returns:
        Number of words in the text
    """
    if not text:
        return 0
    return len(text.split())


def reverse_string(text: Annotated[str, "Text to reverse"]) -> str:
    """Reverse a string.

    Args:
        text: The text to reverse

    Returns:
        The reversed text
    """
    return text[::-1]


def generate_password(
    length: Annotated[int, "Length of password (between 8 and 50)"] = 12,
) -> str:
    """Generate a random password.

    Args:
        length: Length of the password (default 12, min 8, max 50)

    Returns:
        A randomly generated password

    Raises:
        ValueError: If length is not between 8 and 50
    """
    if length < 8 or length > 50:
        raise ValueError("Password length must be between 8 and 50 characters")

    import string

    characters = string.ascii_letters + string.digits + "!@#$%^&*"
    return "".join(random.choice(characters) for _ in range(length))


def fibonacci(n: Annotated[int, "Position in Fibonacci sequence (0-50)"]) -> int:
    """Calculate the nth Fibonacci number.

    Args:
        n: Position in the sequence (0-indexed, max 50)

    Returns:
        The nth Fibonacci number

    Raises:
        ValueError: If n is negative or greater than 50
    """
    if n < 0 or n > 50:
        raise ValueError("n must be between 0 and 50")

    if n <= 1:
        return n

    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def is_palindrome(text: Annotated[str, "Text to check"]) -> bool:
    """Check if a text is a palindrome.

    Args:
        text: The text to check

    Returns:
        True if the text is a palindrome, False otherwise
    """
    # Remove spaces and convert to lowercase for comparison
    cleaned = text.replace(" ", "").lower()
    return cleaned == cleaned[::-1]


# Example usage for demonstration
def demo_all_tools():
    """Demonstrate all the demo tools."""
    print("Demo Tools Examples:")
    print(f"Random fact: {get_random_fact()}")
    print(f"Current time: {current_time()}")
    print(f"Word count of 'Hello world': {word_count('Hello world')}")
    print(f"Reverse 'Hello': {reverse_string('Hello')}")
    print(f"Generate password: {generate_password(10)}")
    print(f"5th Fibonacci number: {fibonacci(5)}")
    print(f"Is 'racecar' a palindrome? {is_palindrome('racecar')}")


if __name__ == "__main__":
    demo_all_tools()
