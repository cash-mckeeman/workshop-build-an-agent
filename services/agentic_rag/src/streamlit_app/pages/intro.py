"""
Introduction Page for Agentic RAG Service

This page provides an overview of the Agentic RAG implementation
and educational objectives.
"""

import streamlit as st
from ..utils.session_state import initialize_session_state


def render():
    """Render the introduction page."""
    st.title("🤖 Agentic RAG Service")
    st.markdown("**Educational Tutorial: From Basic LLM to Sophisticated RAG Agents**")

    # Introduction section
    st.header("🎯 Welcome to Agentic RAG")

    st.markdown("""
    This interactive tutorial demonstrates the evolution from basic Large Language Model (LLM)
    inference to sophisticated **Agentic Retrieval Augmented Generation (RAG)** systems.

    You'll learn how AI agents can make intelligent decisions about when and how to retrieve
    information, moving beyond static pipelines to dynamic, context-aware systems.
    """)

    # Learning path overview
    st.header("📚 Learning Path")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏗️ What You'll Build")
        st.markdown("""
        - **Vector Database**: Semantic search over documents
        - **Embedding Systems**: Text representation for similarity
        - **Reranking Models**: Improved retrieval relevance
        - **RAG Agents**: Intelligent retrieval decision-making
        - **IT Help Desk**: Real-world application example
        """)

    with col2:
        st.subheader("🧠 What You'll Learn")
        st.markdown("""
        - **RAG Evolution**: Traditional → Agentic approaches
        - **ReAct Architecture**: Reasoning and Acting patterns
        - **Tool Integration**: Using retrieval as agent tools
        - **Vector Similarity**: Understanding embedding spaces
        - **Production Patterns**: Scalable RAG implementations
        """)

    # Architecture overview
    st.header("🏛️ Architecture Overview")

    st.markdown("""
    The Agentic RAG system consists of several key components working together:
    """)

    # Create architecture diagram using columns
    st.subheader("System Components")

    arch_col1, arch_col2, arch_col3 = st.columns(3)

    with arch_col1:
        st.markdown("""
        **📄 Knowledge Base**
        - Document loading & processing
        - Text chunking & preprocessing
        - Metadata extraction
        - IT documentation corpus
        """)

    with arch_col2:
        st.markdown("""
        **🗃️ Vector Store**
        - Embedding generation
        - Similarity search (FAISS)
        - Semantic retrieval
        - Metadata filtering
        """)

    with arch_col3:
        st.markdown("""
        **🤖 RAG Agent**
        - Query understanding
        - Retrieval decisions
        - Tool selection
        - Response generation
        """)

    # Flow diagram
    st.subheader("Information Flow")
    st.markdown("""
    ```
    User Query → Agent Analysis → Retrieval Decision → Vector Search →
    Reranking → Context Selection → LLM Generation → Response
    ```
    """)

    # RAG evolution section
    st.header("🔄 RAG Evolution")

    evolution_tabs = st.tabs(["Basic LLM", "Traditional RAG", "Agentic RAG"])

    with evolution_tabs[0]:
        st.markdown("""
        **Basic LLM Inference**
        - Direct question-answer interaction
        - Limited to training data knowledge
        - No access to updated information
        - Fast but potentially outdated

        ```python
        response = llm.generate("How do I reset my password?")
        # May give generic advice, not company-specific
        ```
        """)

    with evolution_tabs[1]:
        st.markdown("""
        **Traditional RAG**
        - Always retrieves documents for every query
        - Fixed retrieval pipeline
        - Static chunk selection
        - Good for knowledge-heavy tasks

        ```python
        # Always retrieve, regardless of query
        docs = vector_store.search(query)
        context = combine_docs(docs)
        response = llm.generate(f"Context: {context}\\nQuery: {query}")
        ```
        """)

    with evolution_tabs[2]:
        st.markdown("""
        **Agentic RAG**
        - Agent decides when to retrieve
        - Dynamic tool selection
        - Multiple retrieval strategies
        - Intelligent context management

        ```python
        # Agent chooses whether and how to retrieve
        agent = RAGAgent(tools=[search_tool, advanced_search])
        response = await agent.run(query)
        # Agent may use tools, combine results, or answer directly
        ```
        """)

    # Use cases section
    st.header("💼 Use Cases")

    use_case_tabs = st.tabs(["IT Help Desk", "Document Q&A", "Knowledge Management"])

    with use_case_tabs[0]:
        st.markdown("""
        **IT Help Desk Agent**

        Our primary example demonstrates an intelligent IT support system:

        - **Password Resets**: Step-by-step procedures
        - **Software Installation**: Platform-specific guides
        - **Network Issues**: Troubleshooting workflows
        - **Hardware Support**: Device-specific instructions
        - **Security Policies**: Compliance information

        The agent can understand context, search relevant documentation,
        and provide accurate, company-specific guidance.
        """)

    with use_case_tabs[1]:
        st.markdown("""
        **Document Question-Answering**

        - Technical documentation search
        - Policy and procedure lookup
        - Troubleshooting guide navigation
        - Multi-document synthesis
        - Source attribution and verification
        """)

    with use_case_tabs[2]:
        st.markdown("""
        **Knowledge Management**

        - Intelligent document organization
        - Automated content discovery
        - Cross-reference identification
        - Knowledge gap detection
        - Content freshness tracking
        """)

    # Getting started section
    st.header("🚀 Getting Started")

    st.markdown("""
    Ready to explore Agentic RAG? Follow this recommended path:

    1. **📊 Vector Database**: Understand how documents become searchable
    2. **🔍 Retrieval Demo**: See semantic search in action
    3. **⚙️ Agent Setup**: Configure your RAG agent
    4. **🎮 Live Demo**: Test the complete system

    Each section builds on the previous one, so we recommend following them in order.
    """)

    # Action buttons
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        if st.button("🗃️ Start with Vector Database", type="primary", use_container_width=True):
            st.session_state.current_page_key = "🗃️ Vector Database"
            st.rerun()

    # Additional resources
    with st.expander("📖 Additional Resources"):
        st.markdown("""
        **Learn More About:**
        - [PydanticAI Documentation](https://pydantic-ai.readthedocs.io/)
        - [FAISS Vector Database](https://github.com/facebookresearch/faiss)
        - [Sentence Transformers](https://www.sbert.net/)
        - [Retrieval Augmented Generation](https://arxiv.org/abs/2005.11401)
        - [ReAct: Reasoning and Acting](https://arxiv.org/abs/2210.03629)

        **Code Examples:**
        - Browse the source code to understand implementation details
        - Modify parameters to see how they affect performance
        - Extend the knowledge base with your own documents
        """)

    # Developer info
    with st.expander("🔧 Developer Information"):
        st.markdown("""
        **System Requirements:**
        - Python 3.11+
        - 4GB+ RAM recommended
        - Local or cloud LLM access

        **Optional Enhancements:**
        - GPU acceleration for faster embeddings
        - Cloud providers for production deployment
        - Custom reranking models for domain-specific use

        **Architecture Notes:**
        - Modular design for easy customization
        - Abstract base classes for component swapping
        - Comprehensive error handling and logging
        """)


if __name__ == "__main__":
    render()