"""Pytest configuration and shared fixtures for RAG system tests."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from langchain_core.documents import Document


@pytest.fixture
def mock_api_key() -> str:
    """
    Provide a mock OpenAI API key for testing.
    
    Returns:
        Mock API key string
    """
    return "sk-test-mock-api-key-12345"


@pytest.fixture
def set_test_api_key(mock_api_key: str) -> Generator[str, None, None]:
    """
    Set mock API key in environment for test duration.
    
    Args:
        mock_api_key: Mock API key fixture
    
    Yields:
        Mock API key string
    """
    original_key = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = mock_api_key
    yield mock_api_key
    if original_key:
        os.environ["OPENAI_API_KEY"] = original_key
    else:
        del os.environ["OPENAI_API_KEY"]


@pytest.fixture
def temp_directory() -> Generator[Path, None, None]:
    """
    Create a temporary directory for test files.
    
    Yields:
        Path to temporary directory
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_txt_file(temp_directory: Path) -> Path:
    """
    Create a sample text file for testing.
    
    Args:
        temp_directory: Temporary directory fixture
    
    Returns:
        Path to sample text file
    """
    file_path = temp_directory / "sample.txt"
    content = "This is a sample text file for testing.\nIt has multiple lines.\n"
    file_path.write_text(content)
    return file_path


@pytest.fixture
def sample_pdf_file(temp_directory: Path) -> Path:
    """
    Create a mock PDF file path for testing.
    
    Note: Returns a path without creating actual PDF content.
    Tests using this should mock the loader.
    
    Args:
        temp_directory: Temporary directory fixture
    
    Returns:
        Path where PDF would exist
    """
    return temp_directory / "sample.pdf"


@pytest.fixture
def sample_documents() -> list[Document]:
    """
    Create sample Document objects for testing.
    
    Returns:
        List of sample Document objects
    """
    return [
        Document(
            page_content="This is the first document about machine learning.",
            metadata={"source": "doc1.txt", "page": 0}
        ),
        Document(
            page_content="This is the second document about artificial intelligence.",
            metadata={"source": "doc2.txt", "page": 0}
        ),
        Document(
            page_content="This is the third document about deep learning and neural networks.",
            metadata={"source": "doc3.txt", "page": 0}
        ),
    ]


@pytest.fixture
def long_document() -> Document:
    """
    Create a long Document object for chunking tests.
    
    Returns:
        Document with long content
    """
    content = " ".join([f"This is sentence number {i}." for i in range(200)])
    return Document(
        page_content=content,
        metadata={"source": "long_doc.txt"}
    )


@pytest.fixture
def sample_query() -> str:
    """
    Provide a sample query string for testing.
    
    Returns:
        Sample query string
    """
    return "What is machine learning?"


@pytest.fixture
def temp_vector_store_path(temp_directory: Path) -> Path:
    """
    Provide path for temporary vector store.
    
    Args:
        temp_directory: Temporary directory fixture
    
    Returns:
        Path for vector store database
    """
    return temp_directory / "test_chroma_db"
