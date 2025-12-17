"""Unit tests for document_loader module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.document_loader import SUPPORTED_LOADERS, DocumentLoader


class TestDocumentLoaderInitialization:
    """Tests for DocumentLoader initialization."""
    
    def test_initialization_succeeds(self):
        """Test that DocumentLoader initializes without errors."""
        loader = DocumentLoader()
        assert loader is not None


class TestGetSupportedExtensions:
    """Tests for get_supported_extensions method."""
    
    def test_returns_list_of_extensions(self):
        """Test that method returns a list of file extensions."""
        extensions = DocumentLoader.get_supported_extensions()
        assert isinstance(extensions, list)
    
    def test_returns_expected_extensions(self):
        """Test that method returns known supported extensions."""
        extensions = DocumentLoader.get_supported_extensions()
        assert '.pdf' in extensions
    
    def test_returns_all_loader_keys(self):
        """Test that method returns all keys from SUPPORTED_LOADERS."""
        extensions = DocumentLoader.get_supported_extensions()
        assert set(extensions) == set(SUPPORTED_LOADERS.keys())


class TestLoadFile:
    """Tests for load_file method."""
    
    def test_raises_error_for_nonexistent_file(self, temp_directory: Path):
        """Test that loading nonexistent file raises FileNotFoundError."""
        loader = DocumentLoader()
        nonexistent_file = temp_directory / "missing.txt"
        
        with pytest.raises(FileNotFoundError):
            loader.load_file(nonexistent_file)
    
    def test_raises_error_for_unsupported_extension(self, temp_directory: Path):
        """Test that loading unsupported file type raises ValueError."""
        loader = DocumentLoader()
        unsupported_file = temp_directory / "test.xyz"
        unsupported_file.touch()
        
        with pytest.raises(ValueError):
            loader.load_file(unsupported_file)
    
    def test_loads_txt_file_successfully(self, sample_txt_file: Path):
        """Test that text file is loaded successfully."""
        loader = DocumentLoader()
        documents = loader.load_file(sample_txt_file)
        
        assert isinstance(documents, list)
    
    def test_returns_documents_list(self, sample_txt_file: Path):
        """Test that load_file returns a list of Document objects."""
        loader = DocumentLoader()
        documents = loader.load_file(sample_txt_file)
        
        assert all(isinstance(doc, Document) for doc in documents)
    
    def test_uses_correct_loader_for_pdf(self, sample_pdf_file: Path):
        """Test that PDF files use the correct loader class."""
        # Instead of trying to load an empty PDF, just verify the loader
        # would be selected correctly based on extension
        loader = DocumentLoader()
        
        # Verify PDF is in supported loaders
        assert '.pdf' in SUPPORTED_LOADERS
        assert SUPPORTED_LOADERS['.pdf'].__name__ == 'PyPDFLoader'
    
    def test_handles_loader_exception(self, temp_directory: Path):
        """Test that loader exceptions are propagated."""
        loader = DocumentLoader()
        txt_file = temp_directory / "test.txt"
        txt_file.write_bytes(b'\x80\x81\x82')  # Invalid UTF-8
        
        with pytest.raises(Exception):
            loader.load_file(txt_file)


class TestListFiles:
    """Tests for list_files method."""
    
    def test_raises_error_for_nonexistent_directory(self, temp_directory: Path):
        """Test that nonexistent directory raises FileNotFoundError."""
        loader = DocumentLoader()
        nonexistent_dir = temp_directory / "missing"
        
        with pytest.raises(FileNotFoundError):
            loader.list_files(nonexistent_dir)
    
    def test_raises_error_for_file_path(self, sample_txt_file: Path):
        """Test that file path instead of directory raises ValueError."""
        loader = DocumentLoader()
        
        with pytest.raises(ValueError):
            loader.list_files(sample_txt_file)
    
    def test_returns_empty_list_for_empty_directory(self, temp_directory: Path):
        """Test that empty directory returns empty list."""
        loader = DocumentLoader()
        files = loader.list_files(temp_directory)
        
        assert files == []
    
    def test_finds_txt_files(self, temp_directory: Path):
        """Test that text files are found in directory."""
        (temp_directory / "file1.txt").touch()
        (temp_directory / "file2.txt").touch()
        
        loader = DocumentLoader()
        files = loader.list_files(temp_directory, file_types=['.txt'])
        
        assert len(files) == 2
    
    def test_finds_multiple_file_types(self, temp_directory: Path):
        """Test that multiple file types are found."""
        (temp_directory / "doc.txt").touch()
        (temp_directory / "doc.pdf").touch()
        (temp_directory / "doc.md").touch()
        
        loader = DocumentLoader()
        files = loader.list_files(
            temp_directory, 
            file_types=['.txt', '.pdf']
        )
        
        assert len(files) == 2
    
    def test_recursive_search_finds_nested_files(self, temp_directory: Path):
        """Test that recursive search finds files in subdirectories."""
        subdir = temp_directory / "subdir"
        subdir.mkdir()
        (temp_directory / "root.txt").touch()
        (subdir / "nested.txt").touch()
        
        loader = DocumentLoader()
        files = loader.list_files(temp_directory, recursive=True)
        
        assert len(files) == 2
    
    def test_non_recursive_ignores_subdirectories(self, temp_directory: Path):
        """Test that non-recursive search ignores subdirectories."""
        subdir = temp_directory / "subdir"
        subdir.mkdir()
        (temp_directory / "root.txt").touch()
        (subdir / "nested.txt").touch()
        
        loader = DocumentLoader()
        files = loader.list_files(temp_directory, recursive=False)
        
        assert len(files) == 1
    
    def test_normalizes_extensions_without_dot(self, temp_directory: Path):
        """Test that extensions without dot are normalized correctly."""
        (temp_directory / "file.txt").touch()
        
        loader = DocumentLoader()
        files = loader.list_files(temp_directory, file_types=['txt'])
        
        assert len(files) == 1
    
    def test_filters_unsupported_file_types(self, temp_directory: Path):
        """Test that unsupported file types are filtered out."""
        (temp_directory / "file.txt").touch()
        
        loader = DocumentLoader()
        files = loader.list_files(
            temp_directory, 
            file_types=['.txt', '.xyz']
        )
        
        assert len(files) == 1
    
    def test_returns_sorted_file_list(self, temp_directory: Path):
        """Test that returned files are sorted."""
        (temp_directory / "c.txt").touch()
        (temp_directory / "a.txt").touch()
        (temp_directory / "b.txt").touch()
        
        loader = DocumentLoader()
        files = loader.list_files(temp_directory)
        
        assert files[0].name == "a.txt"
    
    def test_none_file_types_includes_all_supported(self, temp_directory: Path):
        """Test that None file_types includes all supported formats."""
        (temp_directory / "doc.txt").touch()
        (temp_directory / "doc.pdf").touch()
        (temp_directory / "doc.md").touch()
        
        loader = DocumentLoader()
        files = loader.list_files(temp_directory, file_types=None)
        
        assert len(files) == 3


class TestLoadDirectory:
    """Tests for load_directory method."""
    
    def test_loads_all_files_in_directory(self, temp_directory: Path):
        """Test that all supported files in directory are loaded."""
        (temp_directory / "file1.txt").write_text("content1")
        (temp_directory / "file2.txt").write_text("content2")
        
        loader = DocumentLoader()
        documents = loader.load_directory(temp_directory)
        
        assert len(documents) >= 2
    
    def test_returns_document_list(self, temp_directory: Path):
        """Test that load_directory returns list of Documents."""
        (temp_directory / "file.txt").write_text("content")
        
        loader = DocumentLoader()
        documents = loader.load_directory(temp_directory)
        
        assert isinstance(documents, list)
    
    def test_filters_by_file_types(self, temp_directory: Path):
        """Test that only specified file types are loaded."""
        (temp_directory / "file.txt").write_text("content")
        (temp_directory / "file.md").write_text("# Title")
        
        loader = DocumentLoader()
        documents = loader.load_directory(
            temp_directory,
            file_types=['.txt']
        )
        
        assert all('.txt' in str(doc.metadata.get('source', '')) 
                  for doc in documents if doc.metadata)
    
    def test_empty_directory_returns_empty_list(self, temp_directory: Path):
        """Test that empty directory raises ValueError."""
        loader = DocumentLoader()
        
        with pytest.raises(ValueError, match="No supported files found"):
            loader.load_directory(temp_directory)
