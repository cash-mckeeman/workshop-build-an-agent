"""Tests for tools.validators module."""

import pytest

from src.tools.validators import (
    validate_email_format,
    validate_non_empty,
    validate_positive_int,
    validate_range,
    validate_string_length,
    validate_url_format,
)


@pytest.mark.unit
class TestValidators:
    """Test cases for validator functions."""

    def test_validate_positive_int_valid(self):
        """Test validate_positive_int with valid inputs."""
        assert validate_positive_int(1) == 1
        assert validate_positive_int(100) == 100
        assert validate_positive_int(999999) == 999999

    def test_validate_positive_int_invalid(self):
        """Test validate_positive_int with invalid inputs."""
        with pytest.raises(ValueError, match="Value must be positive, got 0"):
            validate_positive_int(0)

        with pytest.raises(ValueError, match="Value must be positive, got -1"):
            validate_positive_int(-1)

        with pytest.raises(ValueError, match="Value must be positive, got -100"):
            validate_positive_int(-100)

    def test_validate_string_length_valid(self):
        """Test validate_string_length with valid inputs."""
        # Default bounds (1-1000)
        assert validate_string_length("hello") == "hello"
        assert validate_string_length("a") == "a"
        assert validate_string_length("a" * 1000) == "a" * 1000

        # Custom bounds
        assert validate_string_length("test", 3, 10) == "test"
        assert validate_string_length("ab", 1, 5) == "ab"

    def test_validate_string_length_invalid(self):
        """Test validate_string_length with invalid inputs."""
        # Too short (default min_length=1)
        with pytest.raises(
            ValueError, match="String must be at least 1 characters, got 0"
        ):
            validate_string_length("")

        # Too long (default max_length=1000)
        long_string = "a" * 1001
        with pytest.raises(
            ValueError, match="String must be at most 1000 characters, got 1001"
        ):
            validate_string_length(long_string)

        # Custom bounds - too short
        with pytest.raises(
            ValueError, match="String must be at least 5 characters, got 3"
        ):
            validate_string_length("abc", 5, 10)

        # Custom bounds - too long
        with pytest.raises(
            ValueError, match="String must be at most 3 characters, got 5"
        ):
            validate_string_length("hello", 1, 3)

    def test_validate_range_valid(self):
        """Test validate_range with valid inputs."""
        assert validate_range(5.0, 0.0, 10.0) == 5.0
        assert validate_range(0.0, 0.0, 10.0) == 0.0  # Min boundary
        assert validate_range(10.0, 0.0, 10.0) == 10.0  # Max boundary
        assert validate_range(-5.0, -10.0, 0.0) == -5.0

    def test_validate_range_invalid(self):
        """Test validate_range with invalid inputs."""
        with pytest.raises(
            ValueError, match="Value must be between 0.0 and 10.0, got -1.0"
        ):
            validate_range(-1.0, 0.0, 10.0)

        with pytest.raises(
            ValueError, match="Value must be between 0.0 and 10.0, got 11.0"
        ):
            validate_range(11.0, 0.0, 10.0)

        with pytest.raises(
            ValueError, match="Value must be between -5.0 and 5.0, got 6.0"
        ):
            validate_range(6.0, -5.0, 5.0)

    def test_validate_non_empty_valid(self):
        """Test validate_non_empty with valid inputs."""
        assert validate_non_empty("hello") == "hello"
        assert validate_non_empty([1, 2, 3]) == [1, 2, 3]
        assert validate_non_empty({"key": "value"}) == {"key": "value"}
        assert validate_non_empty(42) == 42  # No __len__ method
        assert validate_non_empty(True) == True

    def test_validate_non_empty_invalid(self):
        """Test validate_non_empty with invalid inputs."""
        with pytest.raises(ValueError, match="Value cannot be None"):
            validate_non_empty(None)

        with pytest.raises(ValueError, match="Value cannot be empty"):
            validate_non_empty("")

        with pytest.raises(ValueError, match="Value cannot be empty"):
            validate_non_empty([])

        with pytest.raises(ValueError, match="Value cannot be empty"):
            validate_non_empty({})

    def test_validate_email_format_valid(self):
        """Test validate_email_format with valid emails."""
        valid_emails = [
            "test@example.com",
            "user.name@domain.org",
            "first.last+tag@subdomain.example.co.uk",
            "user123@test-domain.net",
            "a@b.co",
        ]

        for email in valid_emails:
            assert validate_email_format(email) == email

    def test_validate_email_format_invalid(self):
        """Test validate_email_format with invalid emails."""
        invalid_emails = [
            "invalid-email",
            "@domain.com",
            "user@",
            "user@domain",
            "user.domain.com",
            "",
            "user@domain.",
            "user name@domain.com",  # space in local part
        ]

        for email in invalid_emails:
            with pytest.raises(ValueError, match=f"Invalid email format: {email}"):
                validate_email_format(email)

    def test_validate_url_format_valid(self):
        """Test validate_url_format with valid URLs."""
        valid_urls = [
            "https://example.com",
            "http://example.com",
            "https://www.example.com",
            "http://subdomain.example.org",
            "https://example.com/path/to/resource",
            "https://example.com:8080",
            "https://example.com/path?query=value",
        ]

        for url in valid_urls:
            assert validate_url_format(url) == url

    def test_validate_url_format_invalid(self):
        """Test validate_url_format with invalid URLs."""
        invalid_urls = [
            "not-a-url",
            "ftp://example.com",  # Only http/https allowed
            "https://",
            "http://",
            "",
            "example.com",
            "www.example.com",
        ]

        for url in invalid_urls:
            with pytest.raises(ValueError, match=f"Invalid URL format: {url}"):
                validate_url_format(url)

    def test_validator_return_values(self):
        """Test that validators return the original values when valid."""
        # Test that the same object is returned
        test_string = "test_value"
        assert validate_string_length(test_string, 1, 20) is test_string

        test_list = [1, 2, 3]
        assert validate_non_empty(test_list) is test_list

        test_int = 42
        assert validate_positive_int(test_int) is test_int

    def test_validator_edge_cases(self):
        """Test validators with edge cases."""
        # String length with exact boundaries
        assert validate_string_length("exact", 5, 5) == "exact"

        # Range with exact boundaries
        assert validate_range(5.0, 5.0, 5.0) == 5.0

        # Non-empty with single character
        assert validate_non_empty(" ") == " "  # Space is not empty

        # Email with minimum valid format
        assert validate_email_format("a@b.co") == "a@b.co"

        # URL with minimum valid format
        assert validate_url_format("http://a.co") == "http://a.co"
