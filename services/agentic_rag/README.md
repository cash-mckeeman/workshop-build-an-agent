# Agentic RAG Service

An educational implementation of Retrieval Augmented Generation (RAG) using PydanticAI, demonstrating the evolution from basic LLM inference to sophisticated agentic RAG systems.

## Overview

This service builds on the patterns established in the `agent_intro` service to demonstrate advanced RAG concepts including:

- **Agentic Decision Making**: Agents that decide when and how to retrieve information
- **Multi-Provider Support**: Local and cloud-based embedding and reranking models
- **Flexible Architecture**: Pluggable components for embeddings, vector stores, and rerankers
- **Production Patterns**: Factory pattern, error handling, and structured outputs

## Architecture

```
services/agentic_rag/
├── src/
│   ├── factory.py              # Central factory for creating components
│   ├── agents/                 # RAG agent implementations
│   │   ├── rag_agent.py        # Main agentic RAG agent
│   │   └── it_helpdesk_agent.py # Specialized IT support agent
│   ├── retrieval/              # Retrieval system components
│   │   ├── base.py             # Abstract base classes
│   │   ├── vector_store.py     # FAISS vector store implementation
│   │   ├── embeddings/         # Embedding providers
│   │   │   ├── base.py         # Abstract embedding interface
│   │   │   ├── local.py        # sentence-transformers models
│   │   │   └── openai.py       # OpenAI embeddings API
│   │   └── reranker/           # Reranking providers
│   │       ├── base.py         # Abstract reranker interface
│   │       └── local.py        # cross-encoder models
│   ├── tools/                  # PydanticAI tools
│   │   └── retrieval_tools.py  # RAG tools for agents
│   └── models/                 # Model provider configuration
│       └── providers.py        # LLM provider setup
├── tests/                      # Test suite
├── data/                       # Sample knowledge bases
└── run_demo.py                 # Complete demonstration script
```

## Key Components

### 1. Embedding Providers
- **Local**: sentence-transformers models (all-MiniLM-L6-v2, all-mpnet-base-v2)
- **OpenAI**: Cloud-based embeddings (text-embedding-ada-002, text-embedding-3-*)

### 2. Reranking
- **Local**: Cross-encoder models (ms-marco-MiniLM-L-6-v2) for improving retrieval quality

### 3. Vector Store
- **FAISS**: Efficient similarity search with multiple index types (flat, IVF, HNSW)
- **Persistence**: Save/load vector stores for production use

### 4. Agents
- **RAG Agent**: General-purpose agent with retrieval capabilities
- **Simple RAG Agent**: Streamlined version for basic use cases
- **IT Helpdesk Agent**: Domain-specific agent for technical support

### 5. Tools
- **Structured Input/Output**: Pydantic models for reliable agent interactions
- **Search Tools**: Basic and advanced search capabilities
- **Metadata Filtering**: Query-specific document filtering

## Quick Start

### 1. Install Dependencies

```bash
cd services/agentic_rag
uv sync
```

### 2. Set Environment Variables

```bash
# At least one embedding provider is required
export OPENAI_API_KEY="your_openai_key"  # For cloud embeddings
# OR use local embeddings (no API key needed)

# Optional: for better LLM performance
export ANTHROPIC_API_KEY="your_anthropic_key"
```

### 3. Using Ollama (Local Models)

This service supports Ollama for local model inference. Due to PydanticAI's architecture, Ollama models require special handling:

#### Setup Ollama
```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama server
ollama serve

# Pull a model (in another terminal)
ollama pull llama3.2:latest
```

#### Important: Ollama Model Configuration

**Unlike other providers**, Ollama models cannot use the simple string format (`"ollama:model_name"`). Instead, they are automatically converted to use Ollama's OpenAI-compatible API:

```python
# This happens automatically in the factory:
# Input:  "ollama:llama3.2:latest"
# Becomes: OpenAIChatModel(
#   model_name="llama3.2",
#   provider=OpenAIProvider(base_url="http://localhost:11434/v1")
# )
```

**Key points:**
- Ollama models use the OpenAI-compatible API endpoint (`/v1`)
- The `:latest` suffix is automatically stripped
- No API key required (local inference)
- Requires Ollama server running on `localhost:11434`

This special handling is implemented in the `RAGFactory._create_ollama_model()` method and ensures seamless integration with PydanticAI's model interface.

### 4. Run the Demo

```bash
python run_demo.py
```

This will demonstrate:
- Component initialization and testing
- Basic RAG workflows
- Advanced agentic behaviors
- Specialized domain agents

## Usage Examples

### Simple RAG System

```python
from factory import create_rag_system
from retrieval.base import Document

# Create documents
documents = [
    Document(
        content="PydanticAI is a framework for building AI agents...",
        metadata={"topic": "pydantic-ai"}
    ),
    # Add more documents...
]

# Create RAG agent
agent = create_rag_system(documents, agent_type="simple_rag")

# Query the agent
result = await agent.run("What is PydanticAI?")
print(result.data)
```

### Advanced RAG with Factory

```python
from factory import RAGFactory, AgentType
from retrieval.base import Document

# Create factory
factory = RAGFactory(store_path="./vector_stores")

# Create and configure components
vector_store = factory.create_vector_store("knowledge_base")
factory.load_documents(documents, "knowledge_base")

# Create specialized agent
agent = factory.create_agent(
    agent_type=AgentType.IT_HELPDESK,
    vector_store_name="knowledge_base"
)

# Use the agent
result = await agent.run("How do I reset a user's password?")
```

### IT Helpdesk System

```python
from factory import create_it_helpdesk_system
from retrieval.base import Document

# Create IT documentation
it_docs = [
    Document(
        content="Password reset procedure: 1. Verify identity...",
        metadata={"topic": "password", "category": "procedure"}
    ),
    # Add more IT procedures...
]

# Create helpdesk agent
helpdesk_agent = create_it_helpdesk_system(it_docs)

# Handle support requests
result = await helpdesk_agent.run("User can't access their email")
```

## Features

### Educational Progression
- **Step-by-step learning**: From basic concepts to advanced implementation
- **Comparative analysis**: Traditional RAG vs. Agentic RAG
- **Real-world examples**: IT helpdesk and technical documentation scenarios

### Production Ready
- **Error handling**: Graceful degradation when models are unavailable
- **Multiple providers**: Fallback options for different environments
- **Persistence**: Save and load vector stores
- **Structured outputs**: Type-safe agent interactions

### Extensible Design
- **Abstract interfaces**: Easy to add new providers
- **Factory pattern**: Centralized component creation
- **Pluggable tools**: Framework for new capabilities

## Testing

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_factory.py -v

# Run with coverage
uv run pytest --cov=src --cov-report=html
```

## Development

### Adding New Embedding Providers

1. Implement the `EmbeddingProvider` interface in `retrieval/embeddings/`
2. Add configuration to the factory
3. Update tests and documentation

### Adding New Agent Types

1. Create agent in `agents/` following existing patterns
2. Add to `AgentType` enum in `factory.py`
3. Update factory creation logic
4. Add tests and examples

### Local Development

```bash
# Format code
uv run black src/ tests/

# Type checking
uv run mypy src/

# Linting
uv run ruff check src/ tests/
```

## Integration with Agent Intro

This service builds on concepts from `../agent_intro/`:
- **Shared patterns**: Model providers, factory design, tool registration
- **Progressive complexity**: From basic agents to sophisticated RAG systems
- **Consistent interfaces**: Similar API patterns across services

## Educational Objectives

Students will learn:
1. **RAG Evolution**: From basic LLM → Traditional RAG → Agentic RAG
2. **Component Architecture**: Embeddings, vector stores, rerankers, agents
3. **Production Patterns**: Error handling, fallbacks, persistence
4. **Tool Integration**: Using retrieval as agent tools
5. **Domain Specialization**: Adapting agents for specific use cases

## Next Steps

- **Streamlit Interface**: Interactive web demo (see `streamlit_app/`)
- **Multi-modal RAG**: Support for images and structured data
- **Graph RAG**: Knowledge graph integration
- **Multi-agent Systems**: Collaborative RAG agents

This implementation provides a solid foundation for understanding and building production-ready agentic RAG systems.