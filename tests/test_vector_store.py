"""Unit tests for vector_store module."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain_core.documents import Document

from src.vector_store import VectorStore


class TestVectorStoreInitialization:
    """Tests for VectorStore initialization."""
    
    def test_initialization_with_defaults(self, set_test_api_key: str):
        """Test that VectorStore initializes with default settings."""
        store = VectorStore()
        assert store is not None
    
    def test_initialization_with_custom_path(
        self, 
        set_test_api_key: str,
        temp_vector_store_path: Path
    ):
        """Test that custom persist_directory is set correctly."""
        store = VectorStore(persist_directory=temp_vector_store_path)
        assert store.persist_directory == temp_vector_store_path
    
    def test_initialization_with_custom_model(self, set_test_api_key: str):
        """Test that custom embedding_model is set correctly."""
        store = VectorStore(embedding_model="text-embedding-ada-002")
        assert store.embedding_model_name == "text-embedding-ada-002"
    
    def test_embeddings_instance_is_created(self, set_test_api_key: str):
        """Test that embeddings instance is created during init."""
        store = VectorStore()
        assert store.embeddings is not None
    
    def test_vectorstore_is_initially_none(self, set_test_api_key: str):
        """Test that vectorstore is None before loading or creating."""
        store = VectorStore()
        assert store.vectorstore is None


class TestCreateFromDocuments:
    """Tests for create_from_documents method."""
    
    @patch('src.vector_store.Chroma')
    def test_creates_vectorstore_from_documents(
        self,
        mock_chroma: MagicMock,
        set_test_api_key: str,
        sample_documents: list[Document],
        temp_vector_store_path: Path
    ):
        """Test that vectorstore is created from documents."""
        mock_vectorstore = Mock()
        mock_chroma.from_documents.return_value = mock_vectorstore
        
        store = VectorStore(persist_directory=temp_vector_store_path)
        result = store.create_from_documents(sample_documents)
        
        assert result is not None
    
    @patch('src.vector_store.Chroma')
    def test_calls_chroma_from_documents(
        self,
        mock_chroma: MagicMock,
        set_test_api_key: str,
        sample_documents: list[Document],
        temp_vector_store_path: Path
    ):
        """Test that Chroma.from_documents is called with correct args."""
        mock_vectorstore = Mock()
        mock_chroma.from_documents.return_value = mock_vectorstore
        
        store = VectorStore(persist_directory=temp_vector_store_path)
        store.create_from_documents(sample_documents)
        
        mock_chroma.from_documents.assert_called_once()
    
    @patch('src.vector_store.Chroma')
    def test_creates_persist_directory(
        self,
        mock_chroma: MagicMock,
        set_test_api_key: str,
        sample_documents: list[Document],
        temp_directory: Path
    ):
        """Test that persist directory is created if it doesn't exist."""
        nested_path = temp_directory / "nested" / "vector_db"
        mock_vectorstore = Mock()
        mock_chroma.from_documents.return_value = mock_vectorstore
        
        store = VectorStore(persist_directory=nested_path)
        store.create_from_documents(sample_documents)
        
        assert nested_path.parent.exists()
    
    @patch('src.vector_store.Chroma')
    def test_sets_vectorstore_attribute(
        self,
        mock_chroma: MagicMock,
        set_test_api_key: str,
        sample_documents: list[Document],
        temp_vector_store_path: Path
    ):
        """Test that vectorstore attribute is set after creation."""
        mock_vectorstore = Mock()
        mock_chroma.from_documents.return_value = mock_vectorstore
        
        store = VectorStore(persist_directory=temp_vector_store_path)
        store.create_from_documents(sample_documents)
        
        assert store.vectorstore == mock_vectorstore


class TestLoadExisting:
    """Tests for load_existing method."""
    
    def test_raises_error_for_nonexistent_store(
        self,
        set_test_api_key: str,
        temp_directory: Path
    ):
        """Test that loading nonexistent store raises FileNotFoundError."""
        nonexistent_path = temp_directory / "nonexistent_db"
        store = VectorStore(persist_directory=nonexistent_path)
        
        with pytest.raises(FileNotFoundError):
            store.load_existing()
    
    @patch('src.vector_store.Chroma')
    def test_loads_existing_vectorstore(
        self,
        mock_chroma: MagicMock,
        set_test_api_key: str,
        temp_vector_store_path: Path
    ):
        """Test that existing vectorstore is loaded successfully."""
        temp_vector_store_path.mkdir(parents=True)
        mock_vectorstore = Mock()
        mock_chroma.return_value = mock_vectorstore
        
        store = VectorStore(persist_directory=temp_vector_store_path)
        result = store.load_existing()
        
        assert result is not None
    
    @patch('src.vector_store.Chroma')
    def test_calls_chroma_constructor(
        self,
        mock_chroma: MagicMock,
        set_test_api_key: str,
        temp_vector_store_path: Path
    ):
        """Test that Chroma constructor is called when loading."""
        temp_vector_store_path.mkdir(parents=True)
        mock_vectorstore = Mock()
        mock_chroma.return_value = mock_vectorstore
        
        store = VectorStore(persist_directory=temp_vector_store_path)
        store.load_existing()
        
        mock_chroma.assert_called_once()
    
    @patch('src.vector_store.Chroma')
    def test_sets_vectorstore_attribute(
        self,
        mock_chroma: MagicMock,
        set_test_api_key: str,
        temp_vector_store_path: Path
    ):
        """Test that vectorstore attribute is set after loading."""
        temp_vector_store_path.mkdir(parents=True)
        mock_vectorstore = Mock()
        mock_chroma.return_value = mock_vectorstore
        
        store = VectorStore(persist_directory=temp_vector_store_path)
        store.load_existing()
        
        assert store.vectorstore == mock_vectorstore


class TestGetOrCreate:
    """Tests for get_or_create method."""
    
    @patch('src.vector_store.Chroma')
    def test_loads_existing_if_available(
        self,
        mock_chroma: MagicMock,
        set_test_api_key: str,
        temp_vector_store_path: Path
    ):
        """Test that existing store is loaded if it exists."""
        temp_vector_store_path.mkdir(parents=True)
        mock_vectorstore = Mock()
        mock_chroma.return_value = mock_vectorstore
        
        store = VectorStore(persist_directory=temp_vector_store_path)
        result = store.get_or_create()
        
        assert result is not None
    
    @patch('src.vector_store.Chroma')
    def test_creates_new_if_not_exists(
        self,
        mock_chroma: MagicMock,
        set_test_api_key: str,
        sample_documents: list[Document],
        temp_vector_store_path: Path
    ):
        """Test that new store is created if it doesn't exist."""
        mock_vectorstore = Mock()
        mock_chroma.from_documents.return_value = mock_vectorstore
        
        store = VectorStore(persist_directory=temp_vector_store_path)
        result = store.get_or_create(documents=sample_documents)
        
        assert result is not None
    
    def test_raises_error_if_no_documents_provided(
        self,
        set_test_api_key: str,
        temp_vector_store_path: Path
    ):
        """Test that error is raised if store doesn't exist and no documents."""
        store = VectorStore(persist_directory=temp_vector_store_path)
        
        with pytest.raises(ValueError):
            store.get_or_create()


class TestSimilaritySearch:
    """Tests for similarity_search method."""
    
    @patch('src.vector_store.Chroma')
    def test_returns_list_of_documents(
        self,
        mock_chroma: MagicMock,
        set_test_api_key: str,
        sample_query: str,
        temp_vector_store_path: Path
    ):
        """Test that similarity_search returns list of Documents."""
        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search.return_value = [
            Document(page_content="result")
        ]
        mock_chroma.return_value = mock_vectorstore
        
        store = VectorStore(persist_directory=temp_vector_store_path)
        store.vectorstore = mock_vectorstore
        results = store.similarity_search(sample_query)
        
        assert isinstance(results, list)
    
    @patch('src.vector_store.Chroma')
    def test_calls_vectorstore_similarity_search(
        self,
        mock_chroma: MagicMock,
        set_test_api_key: str,
        sample_query: str,
        temp_vector_store_path: Path
    ):
        """Test that vectorstore.similarity_search is called."""
        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search.return_value = []
        
        store = VectorStore(persist_directory=temp_vector_store_path)
        store.vectorstore = mock_vectorstore
        store.similarity_search(sample_query, k=5)
        
        mock_vectorstore.similarity_search.assert_called_once()
    
    @patch('src.vector_store.Chroma')
    def test_passes_k_parameter(
        self,
        mock_chroma: MagicMock,
        set_test_api_key: str,
        sample_query: str,
        temp_vector_store_path: Path
    ):
        """Test that k parameter is passed to similarity_search."""
        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search.return_value = []
        
        store = VectorStore(persist_directory=temp_vector_store_path)
        store.vectorstore = mock_vectorstore
        store.similarity_search(sample_query, k=3)
        
        call_args = mock_vectorstore.similarity_search.call_args
        assert call_args[1]['k'] == 3
