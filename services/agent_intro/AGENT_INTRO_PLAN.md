# Agent Intro Microservice Enhancement Plan with PydanticAI

## Overview

This plan extends the agent_intro microservice to use **PydanticAI** framework for implementing agents with support for multiple model providers. The service provides an interactive Streamlit application that guides users through the core concepts of agents using PydanticAI's type-safe, production-ready approach. The goal is to create a comprehensive educational platform that supports both local (Ollama, vLLM) and cloud (HuggingFace, OpenAI, Anthropic) inference while delivering the same learning experience as the intro_to_agents.ipynb notebook, but using modern PydanticAI patterns.

## Current Agent Intro Architecture (from PLAN.md)

### 1. `code/agent_intro/` - Basic Agent Tutorial
**Purpose**: Educational module demonstrating the four core components of agents
**Source**: `intro_to_agents.answers.ipynb`

```
agent_intro/                 # 🔬 Educational Agent Service
├── pyproject.toml          # uv dependency management
├── uv.lock                 # Locked dependencies
├── .python-version         # Python version specification
├── src/
│   └── agent_intro/
│       ├── __init__.py
│       ├── basic_agent.py  # Main tutorial implementation
│       ├── hf_client.py    # Hugging Face client setup
│       ├── tools.py        # Simple tools (add, multiply, etc.)
│       └── examples/
│           ├── simple_chat.py    # Basic chat example
│           └── tool_calling.py   # Tool calling example
├── tests/                  # Unit tests
├── Containerfile          # Podman container deployment
├── podman-compose.yml     # Local development with podman
└── README.md              # Service documentation
```

**Current HF Integration** (Legacy):
- Use `ChatHuggingFace` with HuggingFaceEndpoint for basic LLM calls
- Start with free-tier Inference Providers (monthly credits included)
- Example models: `microsoft/DialoGPT-medium`, `HuggingFaceH4/zephyr-7b-beta`

## PydanticAI-Based Architecture: Modern Agent Framework

### Why PydanticAI?

PydanticAI provides significant advantages over custom implementations:

1. **Type Safety**: Built-in type checking and validation
2. **Model Agnostic**: Native support for OpenAI, Anthropic, Google, Bedrock, Cohere, Groq, Mistral, and HuggingFace
3. **Production Ready**: Built by the Pydantic team with enterprise-grade reliability
4. **Simplified API**: Clean, intuitive interface for agent creation and execution
5. **Tool Integration**: Decorator-based tool registration with automatic schema generation
6. **Structured Outputs**: Built-in support for structured responses and validation

### PydanticAI Model Support

PydanticAI natively supports multiple model providers through simple string identifiers:

- **OpenAI**: `openai:gpt-4o`, `openai:gpt-4o-mini`
- **Anthropic**: `anthropic:claude-3-5-sonnet-20241022`, `anthropic:claude-3-haiku-20240307`
- **HuggingFace**: `huggingface:microsoft/DialoGPT-medium`, `huggingface:meta-llama/Llama-2-7b-chat-hf`
- **Groq**: `groq:llama3-8b-8192`, `groq:mixtral-8x7b-32768`
- **Ollama**: `ollama:llama3.2`, `ollama:mistral`, `ollama:codellama`
- **Custom Models**: Support for custom model implementations

### Enhanced Service Structure

```
agent_intro/                     # 🔬 PydanticAI Educational Agent Service
├── pyproject.toml              # uv dependency management with PydanticAI
├── uv.lock                     # Locked dependencies
├── .python-version             # Python version specification
├── src/
│   └── agent_intro/
│       ├── __init__.py
│       ├── agents/             # PydanticAI agent implementations
│       │   ├── __init__.py
│       │   ├── basic_agent.py  # Simple tutorial agent
│       │   ├── tool_agent.py   # Agent with tools (add, multiply)
│       │   ├── math_agent.py   # Complete math agent example
│       │   └── factory.py      # Agent factory for different providers
│       ├── tools/              # PydanticAI tool definitions
│       │   ├── __init__.py
│       │   ├── math_tools.py   # Add, multiply, etc.
│       │   ├── demo_tools.py   # Demo tools for tutorial
│       │   └── validators.py   # Input/output validators
│       ├── models/             # Model configuration and management
│       │   ├── __init__.py
│       │   ├── providers.py    # Model provider configurations
│       │   ├── settings.py     # Model-specific settings
│       │   └── health.py       # Health checking for model availability
│       ├── tutorial/           # Interactive tutorial content
│       │   ├── __init__.py
│       │   ├── steps.py        # Tutorial step definitions
│       │   ├── content.py      # Tutorial content from .md and .ipynb
│       │   └── exercises.py    # Interactive exercises
│       ├── streamlit_app/      # Streamlit application
│       │   ├── __init__.py
│       │   ├── app.py          # Main Streamlit app
│       │   ├── pages/          # Multi-page app structure
│       │   │   ├── 01_welcome.py
│       │   │   ├── 02_model_setup.py
│       │   │   ├── 03_tools.py
│       │   │   ├── 04_memory.py
│       │   │   ├── 05_routing.py
│       │   │   └── 06_complete_agent.py
│       │   ├── components/     # Reusable UI components
│       │   │   ├── code_display.py
│       │   │   ├── concept_explanation.py
│       │   │   └── interactive_demo.py
│       │   └── utils/          # Streamlit utilities
│       │       ├── session_state.py
│       │       └── formatters.py
│       └── examples/           # Example implementations
│           ├── simple_chat.py
│           ├── tool_calling.py
│           └── mode_comparison.py  # Compare local vs cloud
├── config/                     # Configuration files
│   ├── models.yaml            # Model configurations
│   ├── local_setup.yaml       # Local environment setup
│   └── cloud_setup.yaml       # Cloud service setup
├── tests/                      # Comprehensive tests
│   ├── test_implementations/
│   ├── test_tutorial/
│   └── test_streamlit/
├── containers/                 # Multiple container setups
│   ├── Containerfile.base     # Base container
│   ├── Containerfile.local    # With Ollama support
│   └── Containerfile.cloud    # Cloud-only minimal
├── scripts/                    # Setup and utility scripts
│   ├── setup_ollama.sh       # Ollama installation
│   ├── setup_vllm.sh         # vLLM setup
│   └── health_check.py       # Service health checks
├── podman-compose.yml         # Multi-service development
└── README.md                  # Enhanced documentation
```

## PydanticAI Implementation Details

### 1. Basic Agent Creation with PydanticAI

#### Simple Agent (`agents/basic_agent.py`)
```python
from pydantic_ai import Agent
from typing import Union

# Basic agent with different model providers
def create_basic_agent(model_provider: str = "openai:gpt-4o-mini") -> Agent[None, str]:
    """Create a basic agent for tutorial introduction"""
    return Agent(
        model_provider,
        instructions="You are a friendly AI assistant helping users learn about agents. Be concise and educational."
    )

# Example usage with different providers
openai_agent = create_basic_agent("openai:gpt-4o-mini")
anthropic_agent = create_basic_agent("anthropic:claude-3-haiku-20240307")
huggingface_agent = create_basic_agent("huggingface:microsoft/DialoGPT-medium")
ollama_agent = create_basic_agent("ollama:llama3.2")
```

#### Agent with Tools (`agents/tool_agent.py`)
```python
from pydantic_ai import Agent
from typing import Annotated

# Create agent with math tools
math_agent = Agent(
    'openai:gpt-4o-mini',
    instructions='Use the provided tools to help with mathematical calculations.'
)

@math_agent.tool
def add(a: Annotated[int, "First number"], b: Annotated[int, "Second number"]) -> int:
    """Add two integers together."""
    return a + b

@math_agent.tool
def multiply(a: Annotated[int, "First number"], b: Annotated[int, "Second number"]) -> int:
    """Multiply two integers together."""
    return a * b

# Example conversation
async def demo_math_agent():
    result = await math_agent.run("What is 3 plus 12?")
    print(result.output)  # Agent will use the add tool
```

#### Model Provider Factory (`agents/factory.py`)
```python
from pydantic_ai import Agent
from typing import Dict, List, Optional, Union
from ..models.providers import ModelProvider
from ..tools.math_tools import register_math_tools

class AgentFactory:
    """Factory for creating agents with different model providers"""

    SUPPORTED_PROVIDERS = {
        "openai": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
        "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
        "huggingface": ["microsoft/DialoGPT-medium", "meta-llama/Llama-2-7b-chat-hf"],
        "ollama": ["llama3.2", "mistral", "codellama"],
        "groq": ["llama3-8b-8192", "mixtral-8x7b-32768"]
    }

    @classmethod
    def create_tutorial_agent(
        cls,
        provider: str,
        model: str,
        with_tools: bool = False,
        instructions: Optional[str] = None
    ) -> Agent:
        """Create an agent for tutorial purposes"""

        model_id = f"{provider}:{model}"
        default_instructions = "You are a helpful AI assistant teaching users about agents."

        agent = Agent(
            model_id,
            instructions=instructions or default_instructions
        )

        if with_tools:
            register_math_tools(agent)

        return agent

    @classmethod
    def get_available_models(cls, provider: str) -> List[str]:
        """Get available models for a provider"""
        return cls.SUPPORTED_PROVIDERS.get(provider, [])

    @classmethod
    def check_provider_health(cls, provider: str, model: str) -> bool:
        """Check if a provider/model combination is available"""
        try:
            test_agent = cls.create_tutorial_agent(provider, model)
            # Simple test to verify the model is accessible
            # Implementation would depend on the provider
            return True
        except Exception:
            return False
```

### 2. Interactive Streamlit Application

#### Core Features

**Mode Selection Page** (`pages/02_model_setup.py`)
- **Local Mode Options**:
  - Ollama (easiest local setup)
  - vLLM (performance-focused)
  - Local HuggingFace pipeline (fallback)
- **Cloud Mode Options**:
  - HuggingFace Inference API (default)
  - HuggingFace Inference Endpoints
- Model availability checking and setup guidance
- Performance and resource requirement comparisons

#### PydanticAI Tutorial Flow (Based on intro_to_agents.ipynb)

**Page 1: Welcome & Introduction** (`pages/01_welcome.py`)
- Content from `introduction_to_agents.md`
- PydanticAI framework introduction
- Four core components with PydanticAI examples
- Comparison: Raw implementation vs PydanticAI

**Page 2: Model Setup** (`pages/02_model_setup.py`)
- PydanticAI model provider selection interface
- Support for OpenAI, Anthropic, HuggingFace, Ollama, Groq
- Interactive model testing and capability demonstration
- Provider health checking and setup verification

**Page 3: Basic Agent Creation** (`pages/03_basic_agent.py`)
- Creating your first PydanticAI agent
- Understanding the Agent class
- Simple instructions and system prompts
- Running synchronous and asynchronous agents
- Interactive agent chat interface

**Page 4: Tools & Function Calling** (`pages/04_tools.py`)
- PydanticAI tool decorator explanation
- Creating math tools (add, multiply) with type annotations
- Tool schema auto-generation
- Live tool testing and execution
- Understanding tool call workflow

**Page 5: Structured Outputs & Memory** (`pages/05_structured_outputs.py`)
- PydanticAI's type-safe output handling
- Conversation memory and message history
- Understanding RunContext and dependencies
- Structured data validation examples

**Page 6: Complete Agent Workflow** (`pages/06_complete_agent.py`)
- End-to-end PydanticAI agent implementation
- Multi-turn conversations with tool usage
- Error handling and retries
- Performance comparison across providers
- Deployment considerations

#### Interactive Components

**Code Display Component** (`components/code_display.py`)
```python
def display_interactive_code(code: str, explanation: str, allow_edit: bool = False):
    """Display code with syntax highlighting and optional editing"""
    pass

def show_execution_step(step_name: str, input_data: Any, output_data: Any):
    """Show step-by-step execution with data flow"""
    pass
```

**Concept Explanation Component** (`components/concept_explanation.py`)
```python
def render_concept_card(title: str, description: str, example: Any):
    """Render expandable concept explanation cards"""
    pass

def show_agent_architecture_diagram():
    """Interactive agent architecture visualization"""
    pass
```

**Interactive Demo Component** (`components/interactive_demo.py`)
```python
def create_live_agent_chat(client: BaseModelClient):
    """Create interactive chat interface with the agent"""
    pass

def tool_execution_sandbox():
    """Sandbox for testing tool implementations"""
    pass
```

### 3. PydanticAI Configuration Management

#### Model Configuration (`config/models.yaml`)
```yaml
# PydanticAI Model Provider Configuration
providers:
  openai:
    default_model: "gpt-4o-mini"
    available_models:
      - "gpt-4o"
      - "gpt-4o-mini"
      - "gpt-3.5-turbo"
    env_var: "OPENAI_API_KEY"
    description: "OpenAI's GPT models"

  anthropic:
    default_model: "claude-3-haiku-20240307"
    available_models:
      - "claude-3-5-sonnet-20241022"
      - "claude-3-haiku-20240307"
      - "claude-3-sonnet-20240229"
    env_var: "ANTHROPIC_API_KEY"
    description: "Anthropic's Claude models"

  huggingface:
    default_model: "microsoft/DialoGPT-medium"
    available_models:
      - "microsoft/DialoGPT-medium"
      - "meta-llama/Llama-2-7b-chat-hf"
      - "mistralai/Mistral-7B-Instruct-v0.1"
    env_var: "HUGGINGFACE_API_KEY"
    description: "HuggingFace models via Inference API"

  ollama:
    default_model: "llama3.2"
    available_models:
      - "llama3.2"
      - "mistral"
      - "codellama"
      - "phi3"
    connection:
      host: "localhost"
      port: 11434
    description: "Local Ollama models"

  groq:
    default_model: "llama3-8b-8192"
    available_models:
      - "llama3-8b-8192"
      - "mixtral-8x7b-32768"
      - "gemma-7b-it"
    env_var: "GROQ_API_KEY"
    description: "Groq's fast inference models"

# Tutorial configuration
tutorial:
  recommended_providers:
    beginner: "openai"     # Most reliable for learning
    local: "ollama"        # Best local experience
    free: "huggingface"    # Free tier available

  demo_models:
    fast: "openai:gpt-4o-mini"
    capable: "anthropic:claude-3-haiku-20240307"
    local: "ollama:llama3.2"
```

#### Setup Configuration (`config/local_setup.yaml`)
```yaml
ollama:
  installation:
    supported_platforms: ["linux", "darwin", "windows"]
    install_commands:
      linux: "curl -fsSL https://ollama.ai/install.sh | sh"
      darwin: "brew install ollama"
      windows: "Download from https://ollama.ai/download"

  models:
    recommended_for_tutorial: "llama3.2"
    size_requirements:
      "llama3.2": "2.0GB"
      "mistral": "4.1GB"
      "codellama": "3.8GB"

vllm:
  requirements:
    min_gpu_memory: "8GB"
    recommended_gpu: "NVIDIA RTX 3080 or better"
  installation:
    pip_install: "pip install vllm"
    container_image: "vllm/vllm-openai:latest"
```

### 4. Enhanced Tutorial Content Integration

#### Content from introduction_to_agents.md
- Agent definition and comparison with LLMs/workflows
- Four core components detailed explanation
- Visual diagrams and interactive elements

#### Content from intro_to_agents.ipynb
- Step-by-step agent building process
- Hands-on coding exercises
- Complete conversation flow demonstration
- Memory structure examples
- Tool calling mechanics

#### Interactive Learning Elements
1. **Concept Quizzes**: Test understanding of each component
2. **Code Sandbox**: Modify and run agent code
3. **Visual Flow**: See data flow through agent components
4. **Comparison Mode**: Side-by-side local vs cloud execution
5. **Progress Tracking**: Save user progress through tutorial

### 5. Development and Deployment

#### Development Workflow
```bash
# Setup development environment
cd code/agent_intro
uv sync

# Run with specific mode
uv run streamlit run src/agent_intro/streamlit_app/app.py

# Environment variables for different providers
# OpenAI
export OPENAI_API_KEY=sk-xxx

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-xxx

# HuggingFace
export HUGGINGFACE_API_KEY=hf_xxx

# Groq
export GROQ_API_KEY=gsk_xxx

# Ollama (no API key needed, just ensure service is running)
# ollama serve
```

#### PydanticAI Dependencies and Setup

```toml
# pyproject.toml
[project]
name = "agent-intro"
version = "1.0.0"
requires-python = ">=3.11"
dependencies = [
    # Core PydanticAI framework
    "pydantic-ai>=0.0.1",
    "pydantic>=2.11.7",

    # Model providers (install as needed)
    "openai>=1.0.0",           # For OpenAI models
    "anthropic>=0.19.0",       # For Anthropic Claude models
    "huggingface-hub>=0.20.0", # For HuggingFace models
    "groq>=0.4.0",             # For Groq models

    # Web interface and utilities
    "streamlit>=1.49.1",
    "pyyaml>=6.0",
    "python-dotenv>=1.0.0",
    "httpx>=0.25.0",
    "rich>=13.0.0",
]

[tool.uv]
dev-dependencies = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "ruff>=0.1.0",
    "mypy>=1.5.0",
]
```

#### Container Support
- **Base container**: Common dependencies using Containerfile
- **Local container**: Includes Ollama and/or vLLM setup
- **Cloud container**: Minimal, cloud-only dependencies
- **Development container**: Full development environment
- **Podman-native**: Uses podman-compose for orchestration

### 6. Testing Strategy

#### Implementation Testing
- Unit tests for each model client implementation
- Integration tests for client factory
- Mock testing for API reliability

#### Tutorial Testing
- Content validation tests
- Interactive component tests
- Cross-platform compatibility tests

#### User Experience Testing
- Tutorial flow completion tests
- Mode switching functionality
- Error handling and fallback testing

## Benefits of Enhanced Architecture

### 1. Flexibility
- Support for multiple inference backends
- Easy switching between local and cloud
- Scalable from development to production

### 2. Educational Value
- Interactive learning experience
- Hands-on coding practice
- Real-time execution and feedback
- Comprehensive concept coverage

### 3. Accessibility
- Local inference for privacy/offline use
- Cloud inference for immediate access
- Progressive complexity (simple → advanced)
- Multiple learning styles supported

### 4. Production Readiness
- Performance comparison tools
- Resource requirement guidance
- Production deployment options
- Monitoring and health checks

## Implementation Phases

### Phase 1: Core Infrastructure
1. ✅ Create enhanced directory structure
2. ✅ Implement abstract base client
3. ✅ Create HuggingFace implementation
4. ✅ Create client factory pattern
5. ✅ Basic configuration management

### Phase 2: Local Implementations
1. Implement Ollama client with auto-setup
2. Implement vLLM client with performance optimization
3. Add fallback and health checking mechanisms
4. Create local installation scripts

### Phase 3: Streamlit Application
1. Create base Streamlit app structure
2. Implement mode selection and setup pages
3. Port tutorial content from notebook
4. Create interactive components and demos
5. Add progress tracking and session management

### Phase 4: Integration and Testing
1. Comprehensive testing suite
2. Cross-platform compatibility testing
3. Performance benchmarking
4. Documentation and user guides

### Phase 5: Enhancement and Optimization
1. Advanced tutorial features
2. Additional model support
3. Performance monitoring
4. User feedback integration

## Success Criteria

### Technical Requirements
- [ ] Support for HuggingFace, Ollama, and vLLM implementations
- [ ] Seamless mode switching in Streamlit app
- [ ] Complete tutorial content integration
- [ ] Interactive learning components functional
- [ ] Comprehensive test coverage
- [ ] Container deployment support

### Educational Effectiveness
- [ ] Users can complete tutorial in both local and cloud modes
- [ ] Interactive elements enhance understanding
- [ ] Progressive complexity maintains engagement
- [ ] Hands-on exercises reinforce concepts
- [ ] Performance comparison educates on trade-offs

### Production Readiness
- [ ] Reliable fallback mechanisms
- [ ] Performance monitoring capabilities
- [ ] Easy deployment and scaling
- [ ] Comprehensive documentation
- [ ] Health checking and monitoring

This enhanced agent_intro microservice will provide a comprehensive, interactive educational platform that maintains the core learning objectives while offering flexibility in deployment and execution environments.