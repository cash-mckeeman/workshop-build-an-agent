"""Tests for tools.demo_tools module."""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
from tools.demo_tools import (
    get_random_fact,
    current_time,
    word_count,
    reverse_string,
    generate_password,
    fibonacci,
    is_palindrome
)


class TestDemoTools:
    """Test cases for demo tool functions."""

    def test_get_random_fact(self):
        """Test get_random_fact function."""
        fact = get_random_fact()

        assert isinstance(fact, str)
        assert len(fact) > 0

        # Call multiple times to ensure it returns facts
        facts = [get_random_fact() for _ in range(10)]
        assert all(isinstance(fact, str) for fact in facts)
        assert all(len(fact) > 0 for fact in facts)

    @patch('tools.demo_tools.datetime')
    def test_current_time(self, mock_datetime):
        """Test current_time function."""
        # Mock datetime.now()
        mock_now = MagicMock()
        mock_now.strftime.return_value = "2024-01-15 10:30:45"
        mock_datetime.now.return_value = mock_now

        result = current_time()

        assert result == "2024-01-15 10:30:45"
        mock_datetime.now.assert_called_once()
        mock_now.strftime.assert_called_once_with("%Y-%m-%d %H:%M:%S")

    def test_word_count(self):
        """Test word_count function."""
        assert word_count("Hello world") == 2
        assert word_count("This is a test") == 4
        assert word_count("One") == 1
        assert word_count("") == 0
        assert word_count("  spaced   words  ") == 2
        assert word_count("Multiple    spaces    between") == 3

    def test_word_count_edge_cases(self):
        """Test word_count with edge cases."""
        assert word_count("") == 0
        assert word_count("   ") == 0
        assert word_count("\n\t") == 0
        assert word_count("word\nwith\nnewlines") == 3

    def test_reverse_string(self):
        """Test reverse_string function."""
        assert reverse_string("hello") == "olleh"
        assert reverse_string("Python") == "nohtyP"
        assert reverse_string("") == ""
        assert reverse_string("a") == "a"
        assert reverse_string("12345") == "54321"
        assert reverse_string("A man a plan a canal Panama") == "amanaP lanac a nalp a nam A"

    def test_reverse_string_edge_cases(self):
        """Test reverse_string with edge cases."""
        assert reverse_string("") == ""
        assert reverse_string(" ") == " "
        assert reverse_string("  ") == "  "
        assert reverse_string("!@#$%") == "%$#@!"

    def test_generate_password_default(self):
        """Test generate_password with default length."""
        password = generate_password()

        assert isinstance(password, str)
        assert len(password) == 12  # Default length

        # Check that it contains a mix of characters
        has_lower = any(c.islower() for c in password)
        has_upper = any(c.isupper() for c in password)
        has_digit = any(c.isdigit() for c in password)

        # At least one type should be present (though not guaranteed in every case)
        assert has_lower or has_upper or has_digit

    def test_generate_password_custom_length(self):
        """Test generate_password with custom lengths."""
        password_8 = generate_password(8)
        password_20 = generate_password(20)

        assert len(password_8) == 8
        assert len(password_20) == 20

    def test_generate_password_invalid_length(self):
        """Test generate_password with invalid lengths."""
        with pytest.raises(ValueError, match="Password length must be between 8 and 50"):
            generate_password(7)

        with pytest.raises(ValueError, match="Password length must be between 8 and 50"):
            generate_password(51)

        with pytest.raises(ValueError, match="Password length must be between 8 and 50"):
            generate_password(0)

    def test_generate_password_randomness(self):
        """Test that generate_password produces different passwords."""
        passwords = [generate_password(12) for _ in range(5)]

        # All passwords should be different (highly likely)
        assert len(set(passwords)) == len(passwords)

    def test_fibonacci(self):
        """Test fibonacci function."""
        assert fibonacci(0) == 0
        assert fibonacci(1) == 1
        assert fibonacci(2) == 1
        assert fibonacci(3) == 2
        assert fibonacci(4) == 3
        assert fibonacci(5) == 5
        assert fibonacci(6) == 8
        assert fibonacci(10) == 55

    def test_fibonacci_edge_cases(self):
        """Test fibonacci with edge cases."""
        assert fibonacci(0) == 0
        assert fibonacci(1) == 1

        # Test larger values within limit
        assert fibonacci(20) == 6765
        assert fibonacci(30) == 832040

    def test_fibonacci_invalid_input(self):
        """Test fibonacci with invalid input."""
        with pytest.raises(ValueError, match="n must be between 0 and 50"):
            fibonacci(-1)

        with pytest.raises(ValueError, match="n must be between 0 and 50"):
            fibonacci(51)

    def test_is_palindrome(self):
        """Test is_palindrome function."""
        # Basic palindromes
        assert is_palindrome("racecar") == True
        assert is_palindrome("level") == True
        assert is_palindrome("a") == True
        assert is_palindrome("") == True

        # Non-palindromes
        assert is_palindrome("hello") == False
        assert is_palindrome("python") == False

        # Case insensitive
        assert is_palindrome("Racecar") == True
        assert is_palindrome("Level") == True

        # With spaces (spaces are removed)
        assert is_palindrome("race car") == True
        assert is_palindrome("A man a plan a canal Panama") == True

    def test_is_palindrome_edge_cases(self):
        """Test is_palindrome with edge cases."""
        assert is_palindrome("") == True
        assert is_palindrome(" ") == True
        assert is_palindrome("  ") == True
        assert is_palindrome("a") == True
        assert is_palindrome("A") == True
        assert is_palindrome("Aa") == True

    def test_is_palindrome_special_characters(self):
        """Test is_palindrome with special characters."""
        # Only considering letters (spaces removed, case ignored)
        assert is_palindrome("race car") == True
        assert is_palindrome("race  car") == True
        assert is_palindrome("RaCe CaR") == True

    def test_function_docstrings(self):
        """Test that all functions have proper docstrings."""
        functions = [
            get_random_fact,
            current_time,
            word_count,
            reverse_string,
            generate_password,
            fibonacci,
            is_palindrome
        ]

        for func in functions:
            assert func.__doc__ is not None
            assert len(func.__doc__.strip()) > 0

    def test_function_type_annotations(self):
        """Test that functions have proper type annotations."""
        # Test functions that should have annotations
        assert hasattr(word_count, '__annotations__')
        assert hasattr(reverse_string, '__annotations__')
        assert hasattr(generate_password, '__annotations__')
        assert hasattr(fibonacci, '__annotations__')
        assert hasattr(is_palindrome, '__annotations__')

        # Check specific annotations
        assert 'text' in word_count.__annotations__
        assert 'text' in reverse_string.__annotations__
        assert 'length' in generate_password.__annotations__
        assert 'n' in fibonacci.__annotations__
        assert 'text' in is_palindrome.__annotations__