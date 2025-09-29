"""
Document Loader for Agentic RAG Knowledge Base

This module provides utilities for loading and processing documents
from various sources into the RAG knowledge base.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging
import hashlib
from datetime import datetime

try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

from ..retrieval.base import Document

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Loads documents from various sources into the RAG knowledge base."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize the document loader.

        Args:
            chunk_size: Maximum size of text chunks
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_markdown_file(self, file_path: Union[str, Path]) -> List[Document]:
        """Load a markdown file and convert it to documents.

        Args:
            file_path: Path to the markdown file

        Returns:
            List of Document objects
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract metadata from filename and content
            metadata = self._extract_markdown_metadata(file_path, content)

            # Split into chunks if content is large
            if len(content) <= self.chunk_size:
                return [Document(content=content, metadata=metadata)]
            else:
                return self._create_chunks(content, metadata)

        except Exception as e:
            logger.error(f"Error loading markdown file {file_path}: {str(e)}")
            raise

    def load_pdf_file(self, file_path: Union[str, Path]) -> List[Document]:
        """Load a PDF file and convert it to documents.

        Args:
            file_path: Path to the PDF file

        Returns:
            List of Document objects
        """
        if not PYPDF_AVAILABLE:
            raise ImportError("pypdf is required for PDF loading. Install with: pip install pypdf")

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            with open(file_path, 'rb') as f:
                reader = pypdf.PdfReader(f)

                documents = []
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text.strip():  # Only process pages with text
                        metadata = {
                            "source": str(file_path),
                            "page": page_num + 1,
                            "type": "pdf",
                            "filename": file_path.name,
                            "loaded_at": datetime.now().isoformat()
                        }

                        # Create chunks if page is large
                        if len(text) <= self.chunk_size:
                            documents.append(Document(content=text, metadata=metadata))
                        else:
                            page_chunks = self._create_chunks(text, metadata)
                            documents.extend(page_chunks)

                return documents

        except Exception as e:
            logger.error(f"Error loading PDF file {file_path}: {str(e)}")
            raise

    def load_text_file(self, file_path: Union[str, Path]) -> List[Document]:
        """Load a plain text file and convert it to documents.

        Args:
            file_path: Path to the text file

        Returns:
            List of Document objects
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            metadata = {
                "source": str(file_path),
                "type": "text",
                "filename": file_path.name,
                "loaded_at": datetime.now().isoformat()
            }

            # Split into chunks if content is large
            if len(content) <= self.chunk_size:
                return [Document(content=content, metadata=metadata)]
            else:
                return self._create_chunks(content, metadata)

        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {str(e)}")
            raise

    def load_directory(self,
                      directory_path: Union[str, Path],
                      file_patterns: Optional[List[str]] = None,
                      recursive: bool = True) -> List[Document]:
        """Load all supported files from a directory.

        Args:
            directory_path: Path to the directory
            file_patterns: List of file patterns to include (e.g., ['*.md', '*.txt'])
            recursive: Whether to search subdirectories

        Returns:
            List of Document objects
        """
        directory_path = Path(directory_path)
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        if file_patterns is None:
            file_patterns = ['*.md', '*.txt', '*.pdf']

        documents = []
        search_pattern = "**/*" if recursive else "*"

        for pattern in file_patterns:
            for file_path in directory_path.glob(f"{search_pattern}.{pattern.lstrip('*.')}"):
                if file_path.is_file():
                    try:
                        file_docs = self.load_file(file_path)
                        documents.extend(file_docs)
                        logger.info(f"Loaded {len(file_docs)} documents from {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to load {file_path}: {str(e)}")
                        continue

        logger.info(f"Loaded total of {len(documents)} documents from {directory_path}")
        return documents

    def load_file(self, file_path: Union[str, Path]) -> List[Document]:
        """Load a file based on its extension.

        Args:
            file_path: Path to the file

        Returns:
            List of Document objects
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        if extension == '.md':
            return self.load_markdown_file(file_path)
        elif extension == '.pdf':
            return self.load_pdf_file(file_path)
        elif extension in ['.txt', '.text']:
            return self.load_text_file(file_path)
        else:
            # Try to load as text file
            logger.warning(f"Unknown file extension {extension}, trying as text file")
            return self.load_text_file(file_path)

    def _extract_markdown_metadata(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Extract metadata from markdown file and content.

        Args:
            file_path: Path to the file
            content: File content

        Returns:
            Dictionary of metadata
        """
        metadata = {
            "source": str(file_path),
            "type": "markdown",
            "filename": file_path.name,
            "loaded_at": datetime.now().isoformat()
        }

        # Extract title from first heading or filename
        lines = content.split('\n')
        for line in lines:
            if line.strip().startswith('# '):
                metadata["title"] = line.strip()[2:].strip()
                break
        else:
            # Use filename as title if no heading found
            metadata["title"] = file_path.stem.replace('-', ' ').replace('_', ' ').title()

        # Extract topic from filename or directory
        filename_lower = file_path.stem.lower()
        if 'password' in filename_lower or 'auth' in filename_lower:
            metadata["topic"] = "authentication"
        elif 'network' in filename_lower or 'wifi' in filename_lower or 'vpn' in filename_lower:
            metadata["topic"] = "networking"
        elif 'software' in filename_lower or 'install' in filename_lower:
            metadata["topic"] = "software"
        elif 'hardware' in filename_lower:
            metadata["topic"] = "hardware"
        elif 'security' in filename_lower:
            metadata["topic"] = "security"
        elif 'help' in filename_lower or 'support' in filename_lower:
            metadata["topic"] = "support"
        else:
            metadata["topic"] = "general"

        return metadata

    def _create_chunks(self, content: str, base_metadata: Dict[str, Any]) -> List[Document]:
        """Split content into chunks with overlap.

        Args:
            content: Text content to split
            base_metadata: Base metadata to apply to all chunks

        Returns:
            List of Document chunks
        """
        chunks = []
        start = 0
        chunk_num = 1

        while start < len(content):
            # Find end of chunk
            end = start + self.chunk_size

            # If not at end of content, try to break at sentence or word boundary
            if end < len(content):
                # Look for sentence end
                sentence_break = content.rfind('.', start, end)
                if sentence_break > start + self.chunk_size // 2:
                    end = sentence_break + 1
                else:
                    # Look for word boundary
                    word_break = content.rfind(' ', start, end)
                    if word_break > start + self.chunk_size // 2:
                        end = word_break

            chunk_text = content[start:end].strip()
            if chunk_text:
                # Create metadata for this chunk
                chunk_metadata = base_metadata.copy()
                chunk_metadata["chunk"] = chunk_num
                chunk_metadata["chunk_start"] = start
                chunk_metadata["chunk_end"] = end

                chunks.append(Document(content=chunk_text, metadata=chunk_metadata))
                chunk_num += 1

            # Move start position with overlap
            start = end - self.chunk_overlap
            if start < 0:
                start = end

        return chunks


def load_it_knowledge_base(data_dir: Union[str, Path] = None) -> List[Document]:
    """Load the IT knowledge base from the data directory.

    Args:
        data_dir: Path to the data directory (defaults to ./data)

    Returns:
        List of Document objects
    """
    if data_dir is None:
        # Default to data directory relative to this file
        current_dir = Path(__file__).parent.parent.parent
        data_dir = current_dir / "data"

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    loader = DocumentLoader(chunk_size=800, chunk_overlap=100)

    # Load all markdown files from the data directory
    documents = loader.load_directory(data_dir, file_patterns=['*.md'], recursive=True)

    logger.info(f"Loaded {len(documents)} documents from IT knowledge base")
    return documents


def setup_knowledge_base():
    """CLI command to setup the knowledge base."""
    print("🔧 Setting up IT Knowledge Base")
    print("=" * 40)

    try:
        documents = load_it_knowledge_base()
        print(f"✅ Successfully loaded {len(documents)} documents")

        # Show some statistics
        topics = {}
        for doc in documents:
            topic = doc.metadata.get("topic", "unknown")
            topics[topic] = topics.get(topic, 0) + 1

        print("\n📊 Documents by topic:")
        for topic, count in sorted(topics.items()):
            print(f"   {topic}: {count} documents")

    except Exception as e:
        print(f"❌ Error setting up knowledge base: {str(e)}")
        raise


if __name__ == "__main__":
    setup_knowledge_base()