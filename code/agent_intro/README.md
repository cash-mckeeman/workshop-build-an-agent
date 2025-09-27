# Agent Intro Service - Educational Agent Tutorial

A microservice demonstrating the four core components of AI agents using Hugging Face models.

## Overview

This service teaches you how to build AI agents from scratch by implementing the four essential components:

1. **Model** - The LLM that powers intelligent responses (Hugging Face models)
2. **Tools** - Functions the agent can call to interact with the world
3. **Memory** - Conversation history and context management
4. **Routing** - Logic to decide what action to take next

## Features

- 🤗 **Hugging Face Integration** - Uses HF Inference Providers for cloud-based model access
- 🔧 **Tool Calling** - Demonstrates function calling with simple math operations
- 💬 **Interactive Chat** - Simple conversation interface
- 📚 **Memory Management** - Conversation history tracking
- 🎯 **Educational Focus** - Clear, well-documented code for learning
- 🚀 **uv Package Management** - Modern, fast dependency management
- 🐳 **Container Ready** - Podman and podman-compose support

## Quick Start

### Prerequisites

1. **Hugging Face Account**: Sign up at [huggingface.co](https://huggingface.co)
2. **API Key**: Get your free API token from [HF Settings](https://huggingface.co/settings/tokens)
3. **Python 3.11+**: This service requires Python 3.11 or higher
4. **uv**: Modern Python package manager
5. **Podman**: Container engine (alternative to Docker)

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
   cd code/agent_intro

   # Create virtual environment and install dependencies
   uv sync

   # Activate virtual environment (optional - uv run handles this)
   source .venv/bin/activate
   ```

4. **Set up environment variables**:
   ```bash
   # Copy example environment file
   cp .env.example .env

   # Edit .env and add your Hugging Face API key
   HUGGING_FACE_API_KEY=hf_your_token_here
   ```

### Basic Usage

```python
from agent_intro.basic_agent import BasicAgent

# Create an agent
agent = BasicAgent()

# Ask a question that requires calculation
response = agent.chat("What is 3 plus 12?")
print(response)  # The agent will use the add tool and return "15"

# View conversation history
agent.print_memory()
```

## Running the Service

### 1. Command Line Scripts

The service provides convenient command-line scripts:

```bash
# Run the main demo
uv run agent-intro-demo

# Simple chat example
uv run agent-intro-chat

# Tool calling example
uv run agent-intro-tools
```

### 2. Python Module Execution

```bash
# Run the main demo
uv run python -m agent_intro.basic_agent

# Run examples
uv run python -m agent_intro.examples.simple_chat
uv run python -m agent_intro.examples.tool_calling
```

### 3. Podman Deployment

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

The agent comes with three simple mathematical tools:

- `add(a, b)` - Add two integers
- `multiply(a, b)` - Multiply two integers
- `subtract(a, b)` - Subtract b from a

## Models

Default model: `microsoft/DialoGPT-medium`

You can use other Hugging Face models:

```python
agent = BasicAgent(
    model_name="HuggingFaceH4/zephyr-7b-beta",  # Better instruction following
    temperature=0.7,
    max_new_tokens=512
)
```

## API Usage and Costs

### Free Tier
- Monthly credits included with free HF account
- Rate limits: ~few hundred requests per hour
- Perfect for learning and development

### Upgrading
- **HF PRO**: $9/month for 20x credits and higher quotas
- **Pay-as-you-go**: Only pay for actual inference time
- **Dedicated Endpoints**: For production workloads

## Code Structure

```
agent_intro/
├── __init__.py           # Module initialization
├── basic_agent.py        # Main BasicAgent class and demo
├── hf_client.py         # Hugging Face client utilities
├── tools.py             # Tool definitions and execution
├── examples/
│   ├── simple_chat.py   # Basic conversation example
│   └── tool_calling.py  # Mathematical tool usage
├── requirements.txt     # Module dependencies
└── README.md           # This file
```

## Understanding the Components

### 1. Model (hf_client.py)

The model component uses `ChatHuggingFace` with `HuggingFaceEndpoint` to connect to HF Inference Providers:

```python
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Create endpoint
llm = HuggingFaceEndpoint(
    repo_id="microsoft/DialoGPT-medium",
    temperature=0.7,
    max_new_tokens=512,
    huggingfacehub_api_token=api_key
)

# Wrap for chat interface
chat_model = ChatHuggingFace(llm=llm)
```

### 2. Tools (tools.py)

Tools are defined with OpenAI-compatible schemas and execution functions:

```python
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

# Tool schema for the model
{
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two integers together",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "First integer"},
                "b": {"type": "integer", "description": "Second integer"},
            },
            "required": ["a", "b"],
        },
    },
}
```

### 3. Memory (basic_agent.py)

Memory is implemented as a list of messages in OpenAI format:

```python
self.memory = [
    {"role": "user", "content": "What is 3 plus 12?"},
    {"role": "assistant", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "...", "name": "add", "content": "15"},
    {"role": "assistant", "content": "3 plus 12 equals 15."}
]
```

### 4. Routing (basic_agent.py)

Routing logic decides whether to use tools or provide a direct response:

```python
def chat(self, user_input: str) -> str:
    self.add_user_message(user_input)
    response = self.process_response()

    if self.handle_tool_calls(response):
        # Tools were used, get final response
        final_response = self.process_response()
        return final_response.get("content", "")
    else:
        # Direct response, no tools needed
        return response.get("content", "")
```

## Troubleshooting

### Common Issues

1. **Missing API Key**:
   ```
   ValueError: HUGGINGFACE_API_KEY environment variable is required
   ```
   **Solution**: Set your HF API key in environment variables

2. **Rate Limiting**:
   ```
   HTTP 429: Too Many Requests
   ```
   **Solution**: Wait a few minutes or upgrade to HF PRO

3. **Model Not Found**:
   ```
   Repository not found
   ```
   **Solution**: Check the model name exists on Hugging Face Hub

4. **Podman Permission Issues**:
   ```
   Error: cannot clone: Operation not permitted
   ```
   **Solution**: Run `podman system migrate` or use rootless mode

5. **Podman Compose Not Found**:
   ```
   command not found: podman-compose
   ```
   **Solution**: Install with `pip install podman-compose` or use docker-compose with podman socket

### Getting Help

- **Hugging Face Docs**: [huggingface.co/docs](https://huggingface.co/docs)
- **LangChain HF Integration**: [python.langchain.com/docs/integrations/providers/huggingface/](https://python.langchain.com/docs/integrations/providers/huggingface/)
- **Model Hub**: [huggingface.co/models](https://huggingface.co/models)

## Next Steps

After mastering the basics here, explore:

1. **`agentic_rag/`** - Learn about Retrieval-Augmented Generation
2. **`docgen_agent/`** - Discover multi-agent workflows
3. **Custom Tools** - Build domain-specific agent capabilities
4. **Production Deployment** - Scale with HF Inference Endpoints

## Contributing

This module is part of the workshop-build-an-agent project. See the main [PLAN.md](../../PLAN.md) for the overall refactoring strategy.