# Workshop Build An Agent - Microservices Refactoring Plan

## Overview

This plan outlines the refactoring of the workshop-build-an-agent repository to create three distinct, standalone **microservices** while removing NVIDIA dependencies and integrating Hugging Face for cloud-based inference. Each service is self-contained with its own:
- **Package Management**: Python dependencies managed via `uv` and `pyproject.toml`
- **Containerization**: Container deployment handled by `podman` with `Containerfile` specifications

## Current State Analysis

The repository currently contains:
- A Jupyter notebook tutorial (`intro_to_agents.answers.ipynb`) demonstrating basic agent concepts
- An agentic RAG system (`rag_agent.answers.py` + `simple_client.py`) for IT help desk queries
- A multi-agent document generation system (`docgen_agent/`) for research and writing
- All modules currently depend on NVIDIA AI endpoints via `langchain-nvidia-ai-endpoints`

## Target Architecture: Microservices with uv

### Three Standalone Microservices

Each service is completely self-contained with its own:
- **Dependencies**: Python packages managed via `uv` and `pyproject.toml`
- **Virtual Environment**: Isolated Python environment via `uv`
- **API/Interface**: Independent entry points
- **Containerization**: Podman-based deployment with `Containerfile` and `podman-compose.yml`
- **Testing**: Independent test suites

#### 1. `code/agent_intro/` - Basic Agent Tutorial
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

**HF Integration**:
- Use `ChatHuggingFace` with HuggingFaceEndpoint for basic LLM calls
- Start with free-tier Inference Providers (monthly credits included)
- Example models: `microsoft/DialoGPT-medium`, `HuggingFaceH4/zephyr-7b-beta`

#### 2. `code/agentic_rag/` - IT Help Desk RAG Agent
**Purpose**: Production-ready RAG agent for IT support queries
**Source**: `rag_agent.answers.py` + `simple_client.py`

```
agentic_rag/                # 🤖 RAG Agent Service
├── pyproject.toml         # uv dependency management
├── uv.lock               # Locked dependencies
├── .python-version       # Python version specification
├── src/
│   └── agentic_rag/
│       ├── __init__.py
│       ├── agent.py      # Main RAG agent
│       ├── retriever.py  # FAISS + HF embeddings
│       ├── reranker.py   # HF-based reranking
│       ├── tools.py      # Retriever tools
│       └── api/
│           ├── __init__.py
│           ├── client.py    # Streamlit interface
│           └── routes.py    # FastAPI routes
├── data/                 # Knowledge base documents
├── tests/               # Unit and integration tests
├── Containerfile       # Podman container deployment
├── podman-compose.yml  # Local development with vector DB
└── README.md           # Service documentation
```

**HF Integration**:
- `HuggingFaceEmbeddings` with `sentence-transformers/all-mpnet-base-v2`
- `ChatHuggingFace` for main LLM (e.g., `mistralai/Mistral-7B-Instruct-v0.1`)
- `HuggingFaceInferenceAPIEmbeddings` for scalable embedding generation
- Alternative reranking with `cross-encoder/ms-marco-MiniLM-L-6-v2`

#### 3. `code/docgen_agent/` - Sales Meeting Prep Multi-Agent
**Purpose**: Multi-agent workflow for research and document generation, enhanced for sales meeting prep
**Source**: Existing `docgen_agent/` + notebook clients

```
docgen_agent/               # 📝 Multi-Agent Document Service
├── pyproject.toml         # uv dependency management
├── uv.lock               # Locked dependencies
├── .python-version       # Python version specification
├── src/
│   └── docgen_agent/
│       ├── __init__.py
│       ├── agent.py          # Main orchestrator
│       ├── researcher.py     # Research agent
│       ├── author.py         # Writing agent
│       ├── tools.py          # Tavily + custom tools
│       ├── prompts.py        # Enhanced prompts
│       ├── meeting_prep/     # Sales-focused module
│       │   ├── __init__.py
│       │   ├── company_research.py
│       │   ├── meeting_prompts.py
│       │   └── sales_templates.py
│       └── api/
│           ├── __init__.py
│           ├── workflows.py  # Multi-agent orchestration
│           └── routes.py     # FastAPI endpoints
├── tests/               # Unit and integration tests
├── Containerfile       # Podman container deployment
├── podman-compose.yml  # Local development
└── README.md           # Service documentation
```

**HF Integration**:
- `ChatHuggingFace` for researcher and author agents
- Support multiple model options (Mistral, Llama, etc.)
- Use Inference Providers for cost-effective scaling
- Keep Tavily for web search (external API)

## Hugging Face Integration Strategy

### Primary Approach
Use `langchain-huggingface` partner package for seamless integration

### Inference Options
1. **HF Inference Providers** (Serverless) - Pay-as-you-go, 200+ models, no infrastructure
2. **HF Inference Endpoints** (Dedicated) - For production workloads requiring consistent performance
3. **Local Pipeline** - For development/testing with local models

### API Requirements & Setup
```bash
# Environment variables needed:
HUGGINGFACE_API_KEY=hf_xxx...  # Free account provides monthly credits
TAVILY_API_KEY=tvly-xxx...     # For web search
LANGSMITH_API_KEY=lsv2_xxx...  # Optional: for tracing
```

### Cost Structure
- **Free Tier**: Monthly credits for experimentation
- **PRO Account**: $9/month for 20x credits and higher quotas
- **Pay-as-you-go**: Only pay for actual inference time
- **Rate Limits**: Free tier ~few hundred requests/hour, PRO gets higher limits

## Microservices Architecture Benefits

### Why Microservices?

1. **Independent Deployment**: Each service can be deployed, scaled, and updated independently
2. **Technology Isolation**: Different services can use different models, frameworks, or Python versions
3. **Team Ownership**: Each service can be owned by different teams or developers
4. **Fault Isolation**: Issues in one service don't affect others
5. **Resource Optimization**: Scale only the services that need it
6. **Development Velocity**: Parallel development without conflicts

### Service Communication

- **agent_intro**: Standalone educational service (no external communication)
- **agentic_rag**: REST API for knowledge base queries
- **docgen_agent**: Workflow orchestration API with external integrations

### uv Dependency Management

Each service uses `uv` for modern Python dependency management:

```toml
# pyproject.toml example
[project]
name = "agent-intro"
version = "1.0.0"
requires-python = ">=3.11"
dependencies = [
    "langchain-huggingface>=0.1.0",
    "langchain>=0.3.27",
    "python-dotenv>=1.0.0",
]

[tool.uv]
dev-dependencies = [
    "pytest>=7.0.0",
    "ruff>=0.1.0",  # Used for both linting and formatting
]
```

**Benefits of uv**:
- ⚡ **10-100x faster** than pip
- 🔒 **Deterministic builds** with uv.lock
- 🚀 **Virtual environment management** built-in
- 📦 **Modern resolution** algorithm
- 🐍 **Python version management**

## Dependencies Migration

### Remove NVIDIA Dependencies
```txt
# Remove:
langchain-nvidia-ai-endpoints~=0.3.12
```

### Add Hugging Face Dependencies
```txt
# Add:
langchain-huggingface>=0.1.0
huggingface-hub>=0.20.0
sentence-transformers>=2.2.0  # For embeddings
transformers>=4.36.0          # For local fallback
torch>=2.1.0                  # For local inference
```

### Keep Existing Dependencies
```txt
# Keep these:
langchain~=0.3.27
langgraph~=0.6.7
tavily-python~=0.7.10
faiss-cpu~=1.12.0
streamlit~=1.49.1
pydantic~=2.11.7
```

## Model Recommendations by Module

1. **Agent Intro**: `microsoft/DialoGPT-medium` (lightweight, good for learning)
2. **Agentic RAG**:
   - LLM: `mistralai/Mistral-7B-Instruct-v0.1`
   - Embeddings: `sentence-transformers/all-mpnet-base-v2`
   - Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
3. **DocGen Agent**: `meta-llama/Llama-2-7b-chat-hf` or `mistralai/Mixtral-8x7B-Instruct-v0.1`

## Fallback Strategy

- Each module can fall back to local inference using `HuggingFacePipeline`
- Provide both API and local setup instructions
- Use smaller models for local testing (e.g., `microsoft/DialoGPT-small`)

## Development Workflow

### Setting Up a Service

```bash
# Navigate to service directory
cd code/agent_intro

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate

# Run the service
uv run python -m agent_intro.basic_agent
```

### Adding Dependencies

```bash
# Add runtime dependency
uv add langchain-huggingface

# Add development dependency
uv add --dev pytest

# Update all dependencies
uv sync
```

### Container Deployment with Podman

```bash
# Build service image
podman build -t agent-intro .

# Run with podman-compose
podman-compose up -d

# Scale specific service
podman-compose up -d --scale docgen-agent=3
```

## Implementation Steps (Updated for Microservices)

### Phase 1: Service Structure Setup
1. ✅ Create PLAN.md file with microservices architecture
2. ✅ Create agent_intro service structure
3. Convert agent_intro to use uv and pyproject.toml
4. Create agentic_rag service and decouple from NVIDIA
5. Enhance docgen_agent service for sales meeting prep

### Phase 2: Service Implementation
6. Implement FastAPI endpoints for agentic_rag
7. Add workflow orchestration API to docgen_agent
8. Create Podman containers for each service
9. Set up podman-compose for local development

### Phase 3: Testing and Production
10. Add comprehensive test suites for each service
11. Set up CI/CD pipelines for independent deployment
12. Create Kubernetes manifests for production deployment
13. Add monitoring and logging infrastructure

## Environment Variables (Updated)

Based on the current `variables.env` file:

```bash
# LangSmith (Optional - for tracing)
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_PROJECT=nv-devx
LANGSMITH_API_KEY=lsv2_xxx...

# Hugging Face (Primary LLM and Embeddings)
HUGGINGFACE_API_KEY=hf_xxx...

# Web Search
TAVILY_API_KEY=tvly-xxx...

# Legacy (to be removed)
OPENROUTER_API_KEY=sk-or-v1-xxx...  # Remove after migration
NVIDIA_API_KEY=nvapi-xxx...         # Remove after migration
```

## Benefits of This Architecture

1. **Separation of Concerns**: Each module has a focused purpose and can be used independently
2. **Educational Value**: Clear progression from basic concepts to complex multi-agent systems
3. **Cloud Integration**: Maintains cloud inference capabilities via Hugging Face
4. **Cost Effective**: Free tier available, pay-as-you-go pricing
5. **Model Diversity**: Access to 200+ models from the HF ecosystem
6. **No Vendor Lock-in**: Can switch between API and local inference
7. **Enterprise Ready**: Dedicated endpoints available for production workloads

## Success Criteria (Microservices)

### Technical Requirements
- [ ] All three services run independently without NVIDIA dependencies
- [ ] Each service uses `uv` for dependency management with `pyproject.toml`
- [ ] Hugging Face integration works with both API and local inference
- [ ] Each service can be containerized with Podman and deployed independently
- [ ] FastAPI endpoints provide REST interfaces where appropriate
- [ ] Services communicate through well-defined APIs

### Development Experience
- [ ] Each service has comprehensive documentation and examples
- [ ] Development workflow is streamlined with `uv` commands
- [ ] Local development uses podman-compose for multi-service orchestration
- [ ] Cost-effective operation within free tier limits for development
- [ ] Independent testing and deployment pipelines

### Business Value
- [ ] **agent_intro**: Effective educational tool for understanding agents
- [ ] **agentic_rag**: Production-ready IT help desk automation
- [ ] **docgen_agent**: Sales meeting prep workflow demonstrates real business value
- [ ] Clear upgrade path to production-ready configurations with scaling

### Architecture Benefits
- [ ] Services can be scaled independently based on demand
- [ ] Technology choices are isolated per service
- [ ] Teams can work on different services without conflicts
- [ ] Fault isolation prevents cascading failures
- [ ] Resource optimization through targeted deployment