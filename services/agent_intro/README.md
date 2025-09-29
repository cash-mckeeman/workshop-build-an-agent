# Agent Intro Service - Educational Agent Tutorial

A microservice demonstrating the four core components of AI agents using PydanticAI framework.

## Overview

This service teaches you how to build AI agents from scratch by implementing the four essential components:

1. **Model** - The LLM that powers intelligent responses (OpenAI, Anthropic, HuggingFace, Groq)
2. **Tools** - Functions the agent can call to interact with the world
3. **Memory** - Conversation history and context management
4. **Routing** - Logic to decide what action to take next

## Features

- 🤖 **PydanticAI Framework** - Modern, type-safe agent framework built on Pydantic
- 🔧 **Tool Calling** - Demonstrates function calling with mathematical operations
- 💬 **Interactive Web UI** - Complete Streamlit-based learning interface
- 📚 **Memory Management** - Conversation history tracking and persistence
- 🎯 **Educational Focus** - Step-by-step tutorials with interactive demos
- 🚀 **uv Package Management** - Modern, fast dependency management
- 🐳 **Container Ready** - Podman and podman-compose support
- 🌐 **Multiple Providers** - Support for OpenAI, Anthropic, HuggingFace, and Groq models

## Quick Start

### Prerequisites

1. **API Keys**: Get API keys from one or more providers:
   - **OpenAI**: [platform.openai.com](https://platform.openai.com/api-keys)
   - **Anthropic**: [console.anthropic.com](https://console.anthropic.com/)
   - **HuggingFace**: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - **Groq**: [console.groq.com](https://console.groq.com/keys)
2. **Python 3.11+**: This service requires Python 3.11 or higher
3. **uv**: Modern Python package manager
4. **Podman**: Container engine (alternative to Docker)

### Installation with uv

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install Podman** (if not already installed):
   ```bash
   # macOS (using Homebrew)
   brew install podman

   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install podman

   # RHEL/CentOS/Fedora
   sudo dnf install podman

   # Install podman-compose
   pip install podman-compose
   ```

3. **Set up the service**:
   ```bash
   cd services/agent_intro

   # Create virtual environment and install dependencies
   uv sync

   # Activate virtual environment (optional - uv run handles this)
   source .venv/bin/activate
   ```

4. **Set up environment variables**:
   ```bash
   # Copy example environment file (if available)
   cp .env.example .env

   # Or create .env with your API keys
   cat > .env << EOF
   OPENAI_API_KEY=sk-your_openai_key_here
   ANTHROPIC_API_KEY=sk-ant-your_anthropic_key_here
   HUGGINGFACE_API_KEY=hf_your_huggingface_key_here
   GROQ_API_KEY=gsk_your_groq_key_here
   EOF
   ```

### Basic Usage

```python
from src.agents.basic_agent import BasicAgent

# Create an agent
agent = BasicAgent()

# Ask a question that requires calculation
response = agent.run("What is 3 plus 12?")
print(response)  # The agent will use the add tool and return "15"

# View conversation history
print(agent.get_conversation_history())
```

## Running the Service

### 1. Interactive Web Interface (Recommended)

The service includes a comprehensive Streamlit web application for learning:

```bash
# Run the Streamlit learning interface
uv run python run_streamlit_app.py

# Or run with specific settings
uv run streamlit run run_streamlit_app.py --server.headless true --server.port 8501
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

### 2. Command Line Scripts

The service provides convenient command-line scripts:

```bash
# Run the main demo
uv run agent-intro-demo

# Simple chat example
uv run agent-intro-chat

# Tool calling example
uv run agent-intro-tools
```

### 3. Python Module Execution

```bash
# Run the main demo
uv run python -m src.agents.basic_agent

# Run examples
uv run python -m src.examples.simple_chat
uv run python -m src.examples.tool_calling
```

### 4. Podman Deployment

Podman is a daemonless container engine that provides several advantages over Docker:
- **Rootless containers**: Runs without requiring root privileges
- **No daemon**: No background service needed
- **Pod support**: Native support for Kubernetes-style pods
- **Docker compatibility**: Drop-in replacement for most Docker commands

```bash
# Build and run with Podman
podman build -t agent-intro .
podman run --env-file .env agent-intro

# Or use podman-compose for development
podman-compose up -d

# Alternative: Use docker-compose with podman (if podman-compose unavailable)
# podman system service --time=0 unix:///tmp/podman.sock &
# DOCKER_HOST=unix:///tmp/podman.sock docker-compose up -d
```

## Available Tools

The agent comes with comprehensive mathematical tools defined in `src/tools/math_tools.py`:

- `add(a, b)` - Add two numbers
- `subtract(a, b)` - Subtract b from a
- `multiply(a, b)` - Multiply two numbers
- `divide(a, b)` - Divide a by b (with error handling for division by zero)
- `power(base, exponent)` - Calculate base raised to the power of exponent
- `square_root(number)` - Calculate the square root of a number

## Models

The service supports multiple model providers through PydanticAI:

- **OpenAI**: GPT-4, GPT-3.5-turbo, etc.
- **Anthropic**: Claude-3.5-sonnet, Claude-3-haiku, etc.
- **HuggingFace**: Various open-source models via Inference API
- **Groq**: Fast inference for Llama, Mixtral, and other models

Configuration is handled through environment variables and the settings system in `src/models/settings.py`.

## API Usage and Costs

Different providers have different pricing models:

### OpenAI
- **Free Trial**: $5 credit for new accounts
- **Pay-per-use**: $0.03-$0.06 per 1K tokens (varies by model)
- **Usage-based billing**: Only pay for what you use

### Anthropic
- **Free Trial**: Limited free usage for new accounts
- **Pay-per-use**: $0.25-$15 per million tokens (varies by model)
- **Claude-3.5-sonnet**: $3 per million input tokens

### HuggingFace
- **Free Tier**: Monthly credits included with free account
- **HF PRO**: $9/month for 20x credits and higher quotas
- **Pay-as-you-go**: Only pay for actual inference time

### Groq
- **Free Tier**: Generous free allowance
- **Pay-per-use**: Very competitive pricing for fast inference

## Code Structure

```
services/agent_intro/
├── src/
│   ├── agents/           # Agent implementations
│   │   ├── basic_agent.py    # Core BasicAgent class
│   │   ├── math_agent.py     # Math-focused agent
│   │   ├── tool_agent.py     # Tool-calling agent
│   │   └── demo_agent.py     # Demo and tutorial agent
│   ├── tools/            # Tool definitions and utilities
│   │   ├── math_tools.py     # Mathematical operations
│   │   ├── demo_tools.py     # Demo and example tools
│   │   ├── validators.py     # Input validation
│   │   └── agent_helpers.py  # Agent utility functions
│   ├── models/           # Model providers and settings
│   │   ├── providers.py      # Model provider configurations
│   │   ├── settings.py       # Application settings
│   │   └── health.py         # Health check utilities
│   ├── streamlit_app/    # Interactive web interface
│   │   ├── app.py           # Main Streamlit application
│   │   ├── pages/           # Individual tutorial pages
│   │   ├── components/      # Reusable UI components
│   │   └── utils/           # Utility functions
│   ├── examples/         # Example scripts
│   │   ├── simple_chat.py   # Basic conversation example
│   │   └── tool_calling.py  # Tool usage examples
│   ├── factory.py        # Agent factory pattern
│   ├── cli.py           # Command-line interface
│   └── app.py           # Main application entry point
├── tests/               # Test suite
├── config/             # Configuration files
├── pyproject.toml      # Project configuration and dependencies
└── README.md           # This file
```

## Understanding the Components

### 1. Model (src/models/providers.py)

The service uses PydanticAI's model providers for accessing different LLMs:

```python
from pydantic_ai import Agent
from pydantic_ai.models import OpenAIModel, AnthropicModel

# Create an agent with OpenAI
agent = Agent(
    model=OpenAIModel('gpt-4'),
    system_prompt="You are a helpful math tutor."
)

# Or with Anthropic
agent = Agent(
    model=AnthropicModel('claude-3-5-sonnet-20241022'),
    system_prompt="You are a helpful math tutor."
)
```

### 2. Tools (src/tools/math_tools.py)

Tools are defined as PydanticAI-compatible functions with type hints:

```python
from pydantic_ai import tool

@tool
def add(a: float, b: float) -> float:
    """Add two numbers together.

    Args:
        a: First number
        b: Second number

    Returns:
        The sum of a and b
    """
    return a + b
```

### 3. Memory (Agent class)

Memory is handled automatically by PydanticAI through conversation history:

```python
# Conversations maintain context automatically
result1 = agent.run("What is 5 + 3?")
result2 = agent.run("Now multiply that by 2")  # Refers to previous result

# Access conversation history
conversation = agent.get_conversation()
```

### 4. Routing (PydanticAI framework)

Routing is handled automatically by the PydanticAI framework:

```python
# The agent automatically decides when to use tools
agent = Agent(
    model=OpenAIModel('gpt-4'),
    tools=[add, subtract, multiply, divide]
)

# Agent will use tools when appropriate
result = agent.run("Calculate 15 * 8 + 7")
# Automatically calls multiply(15, 8) then add(120, 7)
```

## Troubleshooting

### Common Issues

1. **Missing API Key**:
   ```
   ValueError: API key required for provider
   ```
   **Solution**: Set your API key in environment variables (`.env` file)

2. **Rate Limiting**:
   ```
   HTTP 429: Too Many Requests
   ```
   **Solution**: Wait a few minutes or check your provider's usage limits

3. **Model Not Found**:
   ```
   Model not found or not accessible
   ```
   **Solution**: Check the model name and ensure you have access permissions

4. **PydanticAI Installation Issues**:
   ```
   ModuleNotFoundError: No module named 'pydantic_ai'
   ```
   **Solution**: Run `uv sync` to install all dependencies

5. **Streamlit Port Conflicts**:
   ```
   Port 8501 is already in use
   ```
   **Solution**: Use a different port with `--server.port 8502`

6. **Podman Permission Issues**:
   ```
   Error: cannot clone: Operation not permitted
   ```
   **Solution**: Run `podman system migrate` or use rootless mode

### Getting Help

- **PydanticAI Docs**: [ai.pydantic.dev](https://ai.pydantic.dev)
- **OpenAI Docs**: [platform.openai.com/docs](https://platform.openai.com/docs)
- **Anthropic Docs**: [docs.anthropic.com](https://docs.anthropic.com)
- **HuggingFace Docs**: [huggingface.co/docs](https://huggingface.co/docs)
- **Groq Docs**: [console.groq.com/docs](https://console.groq.com/docs)

## Next Steps

After mastering the basics here, explore:

1. **`agentic_rag/`** - Learn about Retrieval-Augmented Generation
2. **`docgen_agent/`** - Discover multi-agent workflows
3. **Custom Tools** - Build domain-specific agent capabilities
4. **Production Deployment** - Scale with HF Inference Endpoints

## Contributing

This module is part of the workshop-build-an-agent project. See the main [PLAN.md](../../PLAN.md) for the overall refactoring strategy.