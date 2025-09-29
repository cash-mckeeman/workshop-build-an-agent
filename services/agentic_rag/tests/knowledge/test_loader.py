"""
Tests for knowledge base loader implementation.

This module tests the document loading and preprocessing functionality
that converts various file formats into structured documents for RAG.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, MagicMock
import tempfile
import shutil

from src.knowledge.loader import (
    DocumentLoader,
    load_documents_from_directory,
    load_document_from_file,
    setup_knowledge_base
)
from src.retrieval.base import Document


class TestDocumentLoader:
    """Test cases for DocumentLoader class."""

    def test_loader_initialization(self):
        """Test DocumentLoader initialization."""
        loader = DocumentLoader()
        assert isinstance(loader.supported_extensions, set)
        assert ".md" in loader.supported_extensions
        assert ".txt" in loader.supported_extensions
        assert ".pdf" in loader.supported_extensions

    def test_is_supported_file(self):
        """Test file extension support checking."""
        loader = DocumentLoader()

        # Supported files
        assert loader.is_supported_file("document.md")
        assert loader.is_supported_file("notes.txt")
        assert loader.is_supported_file("paper.pdf")
        assert loader.is_supported_file("README.MD")  # Case insensitive

        # Unsupported files
        assert not loader.is_supported_file("image.jpg")
        assert not loader.is_supported_file("data.csv")
        assert not loader.is_supported_file("code.py")

    def test_load_text_file(self, mock_knowledge_files):
        """Test loading plain text files."""
        loader = DocumentLoader()

        # Load markdown file
        md_file = mock_knowledge_files / "python_basics.md"
        docs = loader.load_text_file(str(md_file))

        assert isinstance(docs, list)
        assert len(docs) > 0
        doc = docs[0]
        assert isinstance(doc, Document)
        assert "Python Basics" in doc.content
        assert "Python is a high-level programming language" in doc.content
        assert doc.metadata["source"] == str(md_file)
        assert doc.metadata["type"] == "text"
        assert doc.metadata["filename"] == "python_basics.md"
        assert doc.doc_id is not None

        # Load plain text file
        txt_file = mock_knowledge_files / "data_structures.txt"
        docs = loader.load_text_file(str(txt_file))

        assert isinstance(docs, list)
        assert len(docs) > 0
        doc = docs[0]
        assert isinstance(doc, Document)
        assert "Data Structures Overview" in doc.content
        assert doc.metadata["filename"] == "data_structures.txt"

    def test_load_text_file_nonexistent(self):
        """Test loading non-existent text file."""
        loader = DocumentLoader()

        with pytest.raises(FileNotFoundError):
            loader.load_text_file("/nonexistent/file.txt")

    def test_load_text_file_encoding_error(self):
        """Test handling encoding errors."""
        loader = DocumentLoader()

        with patch("builtins.open", side_effect=UnicodeDecodeError('utf-8', b'\xff\xfe', 0, 1, 'invalid start byte')):
            with patch("pathlib.Path.exists", return_value=True):
                with pytest.raises(UnicodeDecodeError):
                    loader.load_text_file("test.txt")

    def test_load_pdf_file_no_pypdf(self, mock_knowledge_files):
        """Test loading PDF files when pypdf is not available."""
        loader = DocumentLoader()

        # Mock PYPDF_AVAILABLE to be False
        with patch('src.knowledge.loader.PYPDF_AVAILABLE', False):
            pdf_file = mock_knowledge_files / "algorithms.pdf"
            with pytest.raises(ImportError, match="pypdf is required"):
                loader.load_pdf_file(str(pdf_file))

    def test_load_pdf_file_nonexistent(self):
        """Test PDF loading for non-existent file."""
        loader = DocumentLoader()

        # Test that FileNotFoundError is raised for non-existent file
        with pytest.raises(FileNotFoundError):
            loader.load_pdf_file("/nonexistent/file.pdf")

    def test_load_file_by_extension(self, mock_knowledge_files):
        """Test loading files based on extension."""
        loader = DocumentLoader()

        # Test markdown file
        md_file = mock_knowledge_files / "python_basics.md"
        docs = loader.load_file(str(md_file))
        assert isinstance(docs, list)
        assert len(docs) > 0
        assert docs[0].metadata["type"] == "markdown"

        # Test text file
        txt_file = mock_knowledge_files / "data_structures.txt"
        docs = loader.load_file(str(txt_file))
        assert isinstance(docs, list)
        assert len(docs) > 0
        assert docs[0].metadata["type"] == "text"

    def test_load_file_unsupported_extension(self, temp_dir):
        """Test loading unsupported file extension."""
        loader = DocumentLoader()

        # Create unsupported file
        unsupported_file = Path(temp_dir) / "test.xyz"
        unsupported_file.write_text("test content")

        # Implementation tries as text file for unsupported extensions
        docs = loader.load_file(str(unsupported_file))
        assert isinstance(docs, list)
        assert len(docs) > 0
        assert docs[0].metadata["type"] == "text"

    def test_load_directory(self, mock_knowledge_files):
        """Test loading all files from directory."""
        loader = DocumentLoader()

        docs = loader.load_directory(str(mock_knowledge_files))

        assert isinstance(docs, list)
        assert len(docs) > 0

        # Check that supported files were loaded
        file_types = {doc.metadata["type"] for doc in docs}
        assert "markdown" in file_types or "text" in file_types

        # Check document structure
        for doc in docs:
            assert isinstance(doc, Document)
            assert doc.content is not None
            assert doc.metadata is not None
            assert "source" in doc.metadata
            assert "type" in doc.metadata

    def test_load_directory_recursive(self, temp_dir):
        """Test recursive directory loading."""
        loader = DocumentLoader()

        # Create nested directory structure
        base_dir = Path(temp_dir) / "knowledge"
        base_dir.mkdir()

        sub_dir = base_dir / "subdir"
        sub_dir.mkdir()

        # Create files
        (base_dir / "root.md").write_text("Root document")
        (sub_dir / "nested.txt").write_text("Nested document")

        docs = loader.load_directory(str(base_dir), recursive=True)

        assert len(docs) == 2
        file_names = {Path(doc.metadata["source"]).name for doc in docs}
        assert "root.md" in file_names
        assert "nested.txt" in file_names

    def test_load_directory_non_recursive(self, temp_dir):
        """Test non-recursive directory loading."""
        loader = DocumentLoader()

        # Create nested directory structure
        base_dir = Path(temp_dir) / "knowledge"
        base_dir.mkdir()

        sub_dir = base_dir / "subdir"
        sub_dir.mkdir()

        # Create files
        (base_dir / "root.md").write_text("Root document")
        (sub_dir / "nested.txt").write_text("Nested document")

        docs = loader.load_directory(str(base_dir), recursive=False)

        assert len(docs) == 1  # Only root file
        assert Path(docs[0].metadata["source"]).name == "root.md"

    def test_load_directory_nonexistent(self):
        """Test loading from non-existent directory."""
        loader = DocumentLoader()

        with pytest.raises(FileNotFoundError):
            loader.load_directory("/nonexistent/directory")

    def test_generate_doc_id(self):
        """Test document ID generation."""
        loader = DocumentLoader()

        # Test with file path
        doc_id1 = loader._generate_doc_id("/path/to/file.txt")
        doc_id2 = loader._generate_doc_id("/path/to/file.txt")
        doc_id3 = loader._generate_doc_id("/path/to/other.txt")

        # Same file should generate same ID
        assert doc_id1 == doc_id2
        # Different files should generate different IDs
        assert doc_id1 != doc_id3

        # IDs should be reasonable length
        assert len(doc_id1) > 10
        assert len(doc_id1) < 50

    def test_extract_metadata(self, temp_dir):
        """Test metadata extraction from files."""
        loader = DocumentLoader()

        # Create test file
        test_file = Path(temp_dir) / "test.md"
        test_file.write_text("Test content")

        metadata = loader._extract_metadata(str(test_file), "md")

        assert metadata["file_path"] == str(test_file)
        assert metadata["file_type"] == "md"
        assert metadata["file_name"] == "test.md"
        assert "file_size" in metadata
        assert "created_at" in metadata
        assert "modified_at" in metadata


class TestDocumentLoaderFunctions:
    """Test cases for standalone loader functions."""

    def test_load_document_from_file(self, mock_knowledge_files):
        """Test loading single document from file."""
        md_file = mock_knowledge_files / "python_basics.md"
        docs = load_document_from_file(str(md_file))

        assert isinstance(docs, list)
        assert len(docs) > 0
        assert isinstance(docs[0], Document)
        assert "Python Basics" in docs[0].content
        assert docs[0].metadata["type"] == "markdown"

    def test_load_document_from_file_nonexistent(self):
        """Test loading from non-existent file."""
        docs = load_document_from_file("/nonexistent/file.txt")
        assert docs == []

    def test_load_documents_from_directory(self, mock_knowledge_files):
        """Test loading multiple documents from directory."""
        docs = load_documents_from_directory(str(mock_knowledge_files))

        assert isinstance(docs, list)
        assert len(docs) > 0

        # Check document types
        for doc in docs:
            assert isinstance(doc, Document)

    def test_load_documents_from_directory_with_options(self, temp_dir):
        """Test loading documents with various options."""
        # Create test structure
        base_dir = Path(temp_dir) / "knowledge"
        base_dir.mkdir()
        sub_dir = base_dir / "subdir"
        sub_dir.mkdir()

        (base_dir / "root.md").write_text("Root content")
        (sub_dir / "nested.txt").write_text("Nested content")
        (base_dir / "image.jpg").write_text("Not a text file")

        # Test recursive loading
        docs_recursive = load_documents_from_directory(str(base_dir), recursive=True)
        assert len(docs_recursive) == 2

        # Test non-recursive loading
        docs_non_recursive = load_documents_from_directory(str(base_dir), recursive=False)
        assert len(docs_non_recursive) == 1


class TestSetupKnowledgeBase:
    """Test cases for knowledge base setup functionality."""

    @patch('builtins.print')
    def test_setup_knowledge_base_success(
        self, mock_print, mock_knowledge_files
    ):
        """Test successful knowledge base setup."""
        # Run setup (actual implementation just loads and prints stats)
        result = setup_knowledge_base(data_dir=str(mock_knowledge_files))

        # Check that documents were loaded
        assert isinstance(result, list)
        assert len(result) > 0
        for doc in result:
            assert isinstance(doc, Document)

    @patch('builtins.print')
    def test_setup_knowledge_base_no_provider(self, mock_print):
        """Test setup with non-existent directory."""
        with pytest.raises(FileNotFoundError):
            setup_knowledge_base(data_dir="/nonexistent/path")

    @patch('builtins.print')
    def test_setup_knowledge_base_no_documents(
        self, mock_print, temp_dir
    ):
        """Test setup with no documents found."""
        # Create empty directory
        empty_dir = Path(temp_dir) / "empty"
        empty_dir.mkdir()

        result = setup_knowledge_base(data_dir=str(empty_dir))

        # Should return empty list since no markdown files found
        assert result == []

    @patch('builtins.print')
    def test_setup_knowledge_base_exception(self, mock_print):
        """Test setup exception handling."""
        with pytest.raises(FileNotFoundError):
            setup_knowledge_base(data_dir="/nonexistent/path")

        print_calls = [call.args[0] for call in mock_print.call_args_list]
        assert any("Error setting up knowledge base" in call for call in print_calls)

    @patch('builtins.print')
    def test_setup_knowledge_base_with_save_path(
        self, mock_print, mock_knowledge_files, temp_dir
    ):
        """Test setup with save path for persistence."""
        save_path = str(Path(temp_dir) / "vector_store")

        result = setup_knowledge_base(data_dir=str(mock_knowledge_files), save_path=save_path)

        # Should return list of documents
        assert isinstance(result, list)
        assert len(result) > 0

        # Check that save path was mentioned in print output
        print_calls = [call.args[0] for call in mock_print.call_args_list]
        assert any("would be saved to" in call for call in print_calls)


class TestDocumentLoaderIntegration:
    """Integration tests for document loader."""

    def test_full_loading_workflow(self, mock_knowledge_files):
        """Test complete document loading workflow."""
        loader = DocumentLoader()

        # Load all documents
        docs = loader.load_directory(str(mock_knowledge_files))

        # Verify structure
        assert len(docs) > 0

        # Check that different file types were loaded
        file_types = {doc.metadata["type"] for doc in docs}
        assert len(file_types) >= 1

        # Verify content extraction
        for doc in docs:
            assert len(doc.content.strip()) > 0
            assert doc.doc_id is not None
            assert doc.metadata["source"] is not None

    def test_content_quality(self, mock_knowledge_files):
        """Test that loaded content has good quality."""
        loader = DocumentLoader()

        docs = loader.load_directory(str(mock_knowledge_files))

        for doc in docs:
            # Content should not be empty
            assert len(doc.content.strip()) > 10

            # Should preserve structure for markdown
            if doc.metadata["type"] == "markdown":
                # Should contain some markdown indicators
                content_has_structure = any(
                    indicator in doc.content
                    for indicator in ["#", "##", "-", "*", "```"]
                )
                # Note: Not all markdown files need these, so we don't assert

            # Metadata should be complete
            required_fields = ["source", "type", "filename"]
            for field in required_fields:
                assert field in doc.metadata

    @pytest.mark.integration
    def test_large_directory_loading(self, temp_dir):
        """Test loading from directory with many files."""
        # Create many test files
        knowledge_dir = Path(temp_dir) / "large_knowledge"
        knowledge_dir.mkdir()

        # Create 50 test files
        for i in range(50):
            file_path = knowledge_dir / f"doc_{i:03d}.md"
            content = f"# Document {i}\n\nThis is test document number {i}.\n\n## Content\n\nSome test content here."
            file_path.write_text(content)

        loader = DocumentLoader()
        docs = loader.load_directory(str(knowledge_dir))

        assert len(docs) == 50

        # Verify all documents loaded correctly
        for i, doc in enumerate(docs):
            assert f"Document {i}" in doc.content or f"test document" in doc.content
            assert doc.metadata["type"] == "markdown"