"""
Global pytest configuration.
"""
import os
import sys
import pytest

# Add src directory to Python path
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: Fast unit tests that don't require external dependencies"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests that require external services"
    )
    config.addinivalue_line(
        "markers", "requires_api: Tests that require API keys to be configured"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take a long time to run (>1 second)"
    )


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests requiring API keys when not available."""
    # Check for common API keys
    has_api_keys = any([
        os.getenv("OPENAI_API_KEY"),
        os.getenv("ANTHROPIC_API_KEY"),
        os.getenv("GROQ_API_KEY"),
        os.getenv("HUGGINGFACE_API_KEY"),
    ])

    if not has_api_keys:
        skip_api = pytest.mark.skip(reason="API keys not configured")
        for item in items:
            if "requires_api" in item.keywords:
                item.add_marker(skip_api)
