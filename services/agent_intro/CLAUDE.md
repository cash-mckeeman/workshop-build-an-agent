# Claude Code Configuration for Agent Intro Service

This file contains Claude Code-specific configurations and commands for the Agent Intro service.

## Project Structure

This is a microservice located at `services/agent_intro/` within the workshop-build-an-agent project. It demonstrates the four core components of AI agents using the PydanticAI framework.

### Import Pattern

This project uses src-layout with absolute imports. **All imports MUST use the `src.` prefix**:

```python
# ✅ CORRECT
from src.agents.basic_agent import BasicAgent
from src.tools.math_tools import add, multiply
from src.models.providers import get_available_providers

# ❌ WRONG
from agents.basic_agent import BasicAgent
from tools.math_tools import add
```

For more details on Python import patterns, see `~/.claude-prompts/python/python-import-fixer.md`.

## Development Commands

### Testing
```bash
# Run all tests with coverage
uv run pytest

# Run specific test file
uv run pytest tests/agents/test_basic_agent.py -v

# Run tests with verbose output
uv run pytest tests/ -v --tb=short

# Run specific test case
uv run pytest tests/agents/test_basic_agent.py::TestBasicAgent::test_agent_creation -v
```

### Code Quality
```bash
# Run linting
uv run ruff check src/ tests/

# Format code with Ruff
uv run ruff format src/ tests/

# Fix auto-fixable issues
uv run ruff check --fix src/ tests/
```

### Running the Service
```bash
# Start the Streamlit web interface
uv run streamlit run src/streamlit_app/app.py

# Or use a different port
uv run streamlit run src/streamlit_app/app.py --server.port 8502

# Run the interactive demo
uv run agent-intro-demo

# Run CLI examples
uv run agent-intro-chat
uv run agent-intro-tools
```

### Development Setup
```bash
# Install dependencies
uv sync

# Install with dev dependencies
uv sync --dev

# Update dependencies
uv lock --upgrade

# Add new dependency
uv add package-name

# Add dev dependency
uv add --dev package-name
```

## Environment Configuration

### Required Environment Variables
```bash
# At least one API key is required
OPENAI_API_KEY=sk-your_openai_key_here
ANTHROPIC_API_KEY=sk-ant-your_anthropic_key_here
HUGGINGFACE_API_KEY=hf_your_huggingface_key_here
GROQ_API_KEY=gsk_your_groq_key_here

# Optional settings
MODEL_PROVIDER=openai  # openai, anthropic, huggingface, groq
MODEL_NAME=gpt-4      # specific model name
LOG_LEVEL=INFO        # DEBUG, INFO, WARNING, ERROR
```

### Setup Environment
```bash
# Create .env file from template
cat > .env << EOF
OPENAI_API_KEY=your_key_here
MODEL_PROVIDER=openai
MODEL_NAME=gpt-4
LOG_LEVEL=INFO
EOF
```

## Common Tasks

### Adding New Tools
1. Define tool function in `src/tools/`
2. Add proper type hints and docstring
3. Register with agent in `src/agents/`
4. Add tests in `tests/tools/`
5. Update documentation

### Adding New Agent Types
1. Create agent class in `src/agents/`
2. Inherit from base agent patterns
3. Configure tools and model
4. Add tests in `tests/agents/`
5. Update factory if needed

### Debugging
```bash
# Run with debug logging
LOG_LEVEL=DEBUG uv run agent-intro-demo

# Run tests with detailed output
uv run pytest -v -s --tb=long

# Check specific test with debugging
uv run pytest tests/agents/test_basic_agent.py::test_function_name -v -s
```

## File Locations

### Key Source Files
- `src/agents/basic_agent.py` - Main agent implementation
- `src/tools/math_tools.py` - Mathematical tool definitions
- `src/models/providers.py` - Model provider configurations
- `src/models/settings.py` - Application settings
- `src/streamlit_app/app.py` - Main Streamlit application

### Configuration Files
- `pyproject.toml` - Project configuration and dependencies
- `conftest.py` - Pytest configuration and fixtures
- `.env` - Environment variables (create from template)

### Test Files
- `tests/agents/` - Agent-related tests
- `tests/tools/` - Tool-related tests
- `tests/models/` - Model and provider tests
- `tests/examples/` - Example script tests
- `tests/streamlit_app/` - Streamlit component tests

## Container Development

### Using Podman
```bash
# Build container
podman build -t agent-intro .

# Run container with environment
podman run --env-file .env -p 8501:8501 agent-intro

# Development with volume mounting
podman run --env-file .env -p 8501:8501 -v $(pwd):/app agent-intro

# Use podman-compose for development
podman-compose up -d
```

## Troubleshooting

### Common Issues
1. **Import errors**: Ensure you're in the right directory and virtual environment is activated
2. **API key errors**: Check `.env` file exists and contains valid keys
3. **Port conflicts**: Use different port with `--server.port 8502`
4. **Test failures**: Run with `-v` flag for detailed output

### Debug Commands
```bash
# Check Python path and imports
uv run python -c "import sys; print(sys.path)"

# Verify environment variables
uv run python -c "import os; print(os.environ.get('OPENAI_API_KEY', 'Not set'))"

# Test basic imports
uv run python -c "from src.agents.basic_agent import BasicAgent; print('Import successful')"
```

## Integration with Main Project

This service is part of the larger workshop-build-an-agent project:
- Main project README: `../../README.md`
- Project plan: `../../PLAN.md`
- Other services: `../agentic_rag/`, `../docgen_agent/`

Use the main project's documentation for overall architecture and deployment strategies.