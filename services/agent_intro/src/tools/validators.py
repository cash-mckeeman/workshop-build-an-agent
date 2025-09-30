"""
Validators for PydanticAI agent inputs and outputs

These validators demonstrate how to add validation logic to ensure
tool inputs and outputs meet specific criteria.
"""

from typing import Any


def validate_positive_int(value: int) -> int:
    """Validate that a number is positive.

    Args:
        value: The integer to validate

    Returns:
        The validated integer

    Raises:
        ValueError: If the value is not positive
    """
    if value <= 0:
        raise ValueError(f"Value must be positive, got {value}")
    return value


def validate_string_length(
    text: str, min_length: int = 1, max_length: int = 1000
) -> str:
    """Validate that a string is within specified length bounds.

    Args:
        text: The string to validate
        min_length: Minimum allowed length (default 1)
        max_length: Maximum allowed length (default 1000)

    Returns:
        The validated string

    Raises:
        ValueError: If the string length is outside the bounds
    """
    if len(text) < min_length:
        raise ValueError(
            f"String must be at least {min_length} characters, got {len(text)}"
        )
    if len(text) > max_length:
        raise ValueError(
            f"String must be at most {max_length} characters, got {len(text)}"
        )
    return text


def validate_range(value: float, min_val: float, max_val: float) -> float:
    """Validate that a number is within a specified range.

    Args:
        value: The number to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)

    Returns:
        The validated number

    Raises:
        ValueError: If the value is outside the range
    """
    if value < min_val or value > max_val:
        raise ValueError(f"Value must be between {min_val} and {max_val}, got {value}")
    return value


def validate_non_empty(value: Any) -> Any:
    """Validate that a value is not None or empty.

    Args:
        value: The value to validate

    Returns:
        The validated value

    Raises:
        ValueError: If the value is None or empty
    """
    if value is None:
        raise ValueError("Value cannot be None")

    # Check for empty strings, lists, dicts, etc.
    if hasattr(value, "__len__") and len(value) == 0:
        raise ValueError("Value cannot be empty")

    return value


def validate_email_format(email: str) -> str:
    """Validate basic email format.

    Args:
        email: The email string to validate

    Returns:
        The validated email

    Raises:
        ValueError: If the email format is invalid
    """
    import re

    email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    if not re.match(email_pattern, email):
        raise ValueError(f"Invalid email format: {email}")
    return email


def validate_url_format(url: str) -> str:
    """Validate basic URL format.

    Args:
        url: The URL string to validate

    Returns:
        The validated URL

    Raises:
        ValueError: If the URL format is invalid
    """
    import re

    url_pattern = r"^https?://[^\s/$.?#].[^\s]*$"
    if not re.match(url_pattern, url):
        raise ValueError(f"Invalid URL format: {url}")
    return url


# Example usage for demonstration
def demo_validators():
    """Demonstrate the validators."""
    print("Validator Examples:")

    try:
        print(f"Positive int validation (5): {validate_positive_int(5)}")
        print(f"String length validation: {validate_string_length('Hello', 1, 10)}")
        print(f"Range validation (7.5 in 0-10): {validate_range(7.5, 0, 10)}")
        print(f"Non-empty validation: {validate_non_empty('test')}")
        print(f"Email validation: {validate_email_format('test@example.com')}")
        print(f"URL validation: {validate_url_format('https://example.com')}")
    except ValueError as e:
        print(f"Validation error: {e}")

    print("\nTesting validation failures:")
    try:
        validate_positive_int(-1)
    except ValueError as e:
        print(f"Expected error: {e}")


if __name__ == "__main__":
    demo_validators()
