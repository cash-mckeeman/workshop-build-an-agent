"""
Text Preprocessing for Agentic RAG Knowledge Base

This module provides utilities for cleaning and preprocessing text content
before it's added to the vector store.
"""

import re
from typing import List, Dict, Any, Optional
import logging
from html import unescape

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Preprocesses text content for better RAG performance."""

    def __init__(self,
                 remove_extra_whitespace: bool = True,
                 normalize_quotes: bool = True,
                 remove_html: bool = True,
                 min_length: int = 10):
        """Initialize the text preprocessor.

        Args:
            remove_extra_whitespace: Whether to remove extra whitespace
            normalize_quotes: Whether to normalize quote characters
            remove_html: Whether to remove HTML tags
            min_length: Minimum length for text chunks to keep
        """
        self.remove_extra_whitespace = remove_extra_whitespace
        self.normalize_quotes = normalize_quotes
        self.remove_html = remove_html
        self.min_length = min_length

    def preprocess_text(self, text: str) -> str:
        """Preprocess text content.

        Args:
            text: Raw text content

        Returns:
            Cleaned and preprocessed text
        """
        if not text or not text.strip():
            return ""

        processed = text

        # Remove HTML tags if requested
        if self.remove_html:
            processed = self._remove_html_tags(processed)

        # Normalize whitespace
        if self.remove_extra_whitespace:
            processed = self._normalize_whitespace(processed)

        # Normalize quotes
        if self.normalize_quotes:
            processed = self._normalize_quotes(processed)

        # Additional cleaning
        processed = self._clean_text(processed)

        # Check minimum length
        if len(processed.strip()) < self.min_length:
            return ""

        return processed.strip()

    def preprocess_markdown(self, text: str) -> str:
        """Preprocess markdown content while preserving structure.

        Args:
            text: Markdown text content

        Returns:
            Cleaned markdown text
        """
        if not text or not text.strip():
            return ""

        processed = text

        # Clean up markdown-specific issues
        processed = self._clean_markdown(processed)

        # Apply general text cleaning
        processed = self.preprocess_text(processed)

        return processed

    def extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases and terms from text.

        Args:
            text: Text content

        Returns:
            List of key phrases
        """
        if not text:
            return []

        # Simple keyword extraction (can be enhanced with NLP libraries)
        phrases = []

        # Extract phrases in quotes
        quoted_phrases = re.findall(r'"([^"]*)"', text)
        phrases.extend([p.strip() for p in quoted_phrases if len(p.strip()) > 2])

        # Extract capitalized phrases (likely proper nouns or important terms)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        phrases.extend([p for p in capitalized if len(p) > 3])

        # Extract phrases with specific patterns (e.g., "step 1", "method A")
        patterns = [
            r'\b(?:step|method|option|type|version|level)\s+\w+\b',
            r'\b\w+\s+(?:configuration|setting|option|mode|method)\b',
            r'\b(?:configure|install|setup|enable|disable)\s+\w+\b'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            phrases.extend(matches)

        # Remove duplicates and return
        return list(set([p.strip() for p in phrases if len(p.strip()) > 2]))

    def _remove_html_tags(self, text: str) -> str:
        """Remove HTML tags from text."""
        # Unescape HTML entities
        text = unescape(text)

        # Remove HTML tags
        clean = re.compile('<.*?>')
        text = re.sub(clean, '', text)

        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)

        # Replace multiple newlines with double newline
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

        # Clean up spaces around newlines
        text = re.sub(r' *\n *', '\n', text)

        return text

    def _normalize_quotes(self, text: str) -> str:
        """Normalize quote characters."""
        # Replace smart quotes with regular quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")

        # Replace backticks used as quotes
        text = re.sub(r'`([^`]+)`', r'"\1"', text)

        return text

    def _clean_text(self, text: str) -> str:
        """Apply general text cleaning."""
        # Remove URLs (optional - might want to keep some)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]', text)

        # Remove email addresses (optional)
        text = re.sub(r'\S+@\S+\.\S+', '[EMAIL]', text)

        # Clean up special characters
        text = re.sub(r'[^\w\s\-\.\,\!\?\;\:\(\)\[\]\{\}\'"\/\\]', ' ', text)

        # Fix spacing around punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([,.!?;:])\s*', r'\1 ', text)

        return text

    def _clean_markdown(self, text: str) -> str:
        """Clean markdown-specific formatting issues."""
        # Fix table formatting
        text = re.sub(r'\|\s*\|\s*\|', '|', text)

        # Clean up code blocks
        text = re.sub(r'```\w*\n', '```\n', text)

        # Fix heading spacing
        text = re.sub(r'^(#{1,6})\s*(.+)$', r'\1 \2', text, flags=re.MULTILINE)

        # Fix list formatting
        text = re.sub(r'^(\s*)([-*+])\s+', r'\1\2 ', text, flags=re.MULTILINE)
        text = re.sub(r'^(\s*)(\d+\.)\s+', r'\1\2 ', text, flags=re.MULTILINE)

        return text


def preprocess_documents(documents: List['Document']) -> List['Document']:
    """Preprocess a list of documents.

    Args:
        documents: List of Document objects

    Returns:
        List of preprocessed Document objects
    """
    preprocessor = TextPreprocessor()
    processed_docs = []

    for doc in documents:
        try:
            # Determine the best preprocessing method based on document type
            doc_type = doc.metadata.get("type", "text")

            if doc_type == "markdown":
                processed_content = preprocessor.preprocess_markdown(doc.content)
            else:
                processed_content = preprocessor.preprocess_text(doc.content)

            # Only keep documents with content after preprocessing
            if processed_content:
                # Extract key phrases for enhanced metadata
                key_phrases = preprocessor.extract_key_phrases(processed_content)

                # Create new document with processed content and enhanced metadata
                new_metadata = doc.metadata.copy()
                new_metadata["key_phrases"] = key_phrases
                new_metadata["processed"] = True
                new_metadata["original_length"] = len(doc.content)
                new_metadata["processed_length"] = len(processed_content)

                from ..retrieval.base import Document
                processed_doc = Document(
                    content=processed_content,
                    metadata=new_metadata,
                    doc_id=doc.doc_id
                )
                processed_docs.append(processed_doc)

        except Exception as e:
            logger.warning(f"Failed to preprocess document {doc.doc_id}: {str(e)}")
            # Keep original document if preprocessing fails
            processed_docs.append(doc)

    logger.info(f"Preprocessed {len(processed_docs)} documents (from {len(documents)} originals)")
    return processed_docs


def enhance_metadata(documents: List['Document']) -> List['Document']:
    """Enhance document metadata with additional information.

    Args:
        documents: List of Document objects

    Returns:
        List of documents with enhanced metadata
    """
    enhanced_docs = []

    for doc in documents:
        try:
            content = doc.content
            metadata = doc.metadata.copy()

            # Calculate content statistics
            metadata["word_count"] = len(content.split())
            metadata["char_count"] = len(content)
            metadata["line_count"] = len(content.split('\n'))

            # Detect content characteristics
            if content.count('\n') / len(content.split()) > 0.5:
                metadata["format"] = "structured"
            else:
                metadata["format"] = "prose"

            # Detect if content contains code
            if any(indicator in content.lower() for indicator in ['```', 'def ', 'function', 'import ', 'class ']):
                metadata["contains_code"] = True
            else:
                metadata["contains_code"] = False

            # Detect if content contains commands
            if any(indicator in content for indicator in ['$', '>', 'cmd', 'bash', 'sh']):
                metadata["contains_commands"] = True
            else:
                metadata["contains_commands"] = False

            # Estimate reading time (words per minute)
            words = len(content.split())
            metadata["estimated_reading_time_minutes"] = max(1, words // 200)

            from ..retrieval.base import Document
            enhanced_doc = Document(
                content=content,
                metadata=metadata,
                doc_id=doc.doc_id
            )
            enhanced_docs.append(enhanced_doc)

        except Exception as e:
            logger.warning(f"Failed to enhance metadata for document {doc.doc_id}: {str(e)}")
            enhanced_docs.append(doc)

    return enhanced_docs


if __name__ == "__main__":
    # Demo preprocessing functionality
    print("🔧 Text Preprocessing Demo")
    print("=" * 40)

    sample_text = """
    # IT Support Guide

    This guide provides information about common IT support procedures.

    ## Password Reset

    To reset your password:
    1. Go to the password reset page
    2. Enter your email address
    3. Check your email for instructions

    For more information, contact support@company.com or visit https://support.company.com
    """

    preprocessor = TextPreprocessor()

    print("Original text:")
    print(repr(sample_text))

    processed = preprocessor.preprocess_markdown(sample_text)
    print("\nProcessed text:")
    print(repr(processed))

    key_phrases = preprocessor.extract_key_phrases(sample_text)
    print(f"\nKey phrases: {key_phrases}")