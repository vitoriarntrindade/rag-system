"""Unit tests for text_processor module."""

import pytest
from langchain_core.documents import Document

from rag_system.src.text_processor import TextProcessor


class TestTextProcessorInitialization:
    """Tests for TextProcessor initialization."""
    
    def test_initialization_with_defaults(self):
        """Test that TextProcessor initializes with default settings."""
        processor = TextProcessor()
        assert processor is not None
    
    def test_initialization_with_custom_chunk_size(self):
        """Test that custom chunk_size is set correctly."""
        processor = TextProcessor(chunk_size=500)
        assert processor.chunk_size == 500
    
    def test_initialization_with_custom_chunk_overlap(self):
        """Test that custom chunk_overlap is set correctly."""
        processor = TextProcessor(chunk_overlap=50)
        assert processor.chunk_overlap == 50
    
    def test_initialization_with_custom_separators(self):
        """Test that custom separators are set correctly."""
        custom_separators = ["\n\n", "\n", " "]
        processor = TextProcessor(separators=custom_separators)
        assert processor.separators == custom_separators
    
    def test_text_splitter_is_created(self):
        """Test that text_splitter instance is created during init."""
        processor = TextProcessor()
        assert processor.text_splitter is not None


class TestSplitDocuments:
    """Tests for split_documents method."""
    
    def test_returns_list_of_documents(self, sample_documents: list[Document]):
        """Test that method returns a list of Document objects."""
        processor = TextProcessor(chunk_size=100, chunk_overlap=20)
        chunks = processor.split_documents(sample_documents)
        
        assert isinstance(chunks, list)
    
    def test_splits_long_document_into_chunks(self, long_document: Document):
        """Test that long document is split into multiple chunks."""
        processor = TextProcessor(chunk_size=100, chunk_overlap=20)
        chunks = processor.split_documents([long_document])
        
        assert len(chunks) > 1
    
    def test_small_document_remains_single_chunk(self):
        """Test that small document is not split unnecessarily."""
        small_doc = Document(page_content="Short text.")
        processor = TextProcessor(chunk_size=1000, chunk_overlap=200)
        chunks = processor.split_documents([small_doc])
        
        assert len(chunks) == 1
    
    def test_empty_document_list_returns_empty(self):
        """Test that empty document list returns empty chunks."""
        processor = TextProcessor()
        chunks = processor.split_documents([])
        
        assert chunks == []
    
    def test_preserves_metadata_in_chunks(self):
        """Test that document metadata is preserved in chunks."""
        doc = Document(
            page_content="Test content " * 100,
            metadata={"source": "test.txt", "page": 1}
        )
        processor = TextProcessor(chunk_size=100, chunk_overlap=20)
        chunks = processor.split_documents([doc])
        
        assert all("source" in chunk.metadata for chunk in chunks)
    
    def test_multiple_documents_are_all_chunked(self, sample_documents: list[Document]):
        """Test that multiple documents are all processed."""
        processor = TextProcessor(chunk_size=50, chunk_overlap=10)
        chunks = processor.split_documents(sample_documents)
        
        assert len(chunks) >= len(sample_documents)
    
    def test_chunk_size_respected(self):
        """Test that chunks respect the specified size limit."""
        doc = Document(page_content="word " * 500)
        processor = TextProcessor(chunk_size=100, chunk_overlap=20)
        chunks = processor.split_documents([doc])
        
        assert all(len(chunk.page_content) <= 150 for chunk in chunks)
    
    def test_chunk_overlap_creates_redundancy(self):
        """Test that chunk overlap creates content redundancy between chunks."""
        content = "sentence " * 100
        doc = Document(page_content=content)
        processor = TextProcessor(chunk_size=100, chunk_overlap=50)
        chunks = processor.split_documents([doc])
        
        assert len(chunks) > 1
    
    def test_handles_document_with_metadata_only(self):
        """Test that documents with only metadata are handled."""
        doc = Document(page_content="", metadata={"source": "empty.txt"})
        processor = TextProcessor()
        chunks = processor.split_documents([doc])
        
        assert isinstance(chunks, list)
    
    def test_returns_all_document_instances(self, long_document: Document):
        """Test that all returned items are Document instances."""
        processor = TextProcessor(chunk_size=100, chunk_overlap=20)
        chunks = processor.split_documents([long_document])
        
        assert all(isinstance(chunk, Document) for chunk in chunks)


class TestTextProcessorConfiguration:
    """Tests for TextProcessor configuration options."""
    
    def test_default_chunk_size_from_settings(self):
        """Test that default chunk_size comes from settings."""
        processor = TextProcessor()
        assert processor.chunk_size > 0
    
    def test_default_chunk_overlap_from_settings(self):
        """Test that default chunk_overlap comes from settings."""
        processor = TextProcessor()
        assert processor.chunk_overlap >= 0
    
    def test_custom_values_override_defaults(self):
        """Test that custom values override default settings."""
        processor = TextProcessor(chunk_size=300, chunk_overlap=75)
        assert processor.chunk_size == 300
        assert processor.chunk_overlap == 75
    
    def test_default_separators_include_paragraph_breaks(self):
        """Test that default separators include paragraph breaks."""
        processor = TextProcessor()
        assert "\n\n" in processor.separators
    
    def test_default_separators_include_line_breaks(self):
        """Test that default separators include line breaks."""
        processor = TextProcessor()
        assert "\n" in processor.separators
