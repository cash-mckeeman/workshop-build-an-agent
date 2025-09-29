"""
Knowledge Management Module for Agentic RAG Service

This module provides utilities for loading, processing, and managing
the knowledge base for RAG applications.
"""

from .loader import DocumentLoader, load_it_knowledge_base, setup_knowledge_base
from .preprocessor import TextPreprocessor, preprocess_documents, enhance_metadata

__all__ = [
    "DocumentLoader",
    "load_it_knowledge_base",
    "setup_knowledge_base",
    "TextPreprocessor",
    "preprocess_documents",
    "enhance_metadata"
]