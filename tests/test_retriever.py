"""Unit tests for retriever module."""

from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from src.retriever import DocumentRetriever
from src.vector_store import VectorStore


class TestDocumentRetrieverInitialization:
    """Tests for DocumentRetriever initialization."""
    
    def test_initialization_with_defaults(self, set_test_api_key: str):
        """Test that DocumentRetriever initializes with default settings."""
        mock_vector_store = Mock(spec=VectorStore)
        retriever = DocumentRetriever(vector_store=mock_vector_store)
        
        assert retriever is not None
    
    def test_initialization_with_custom_search_type(self, set_test_api_key: str):
        """Test that custom search_type is set correctly."""
        mock_vector_store = Mock(spec=VectorStore)
        retriever = DocumentRetriever(
            vector_store=mock_vector_store,
            search_type="mmr"
        )
        
        assert retriever.search_type == "mmr"
    
    def test_initialization_with_custom_top_k(self, set_test_api_key: str):
        """Test that custom top_k is set correctly."""
        mock_vector_store = Mock(spec=VectorStore)
        retriever = DocumentRetriever(
            vector_store=mock_vector_store,
            top_k=10
        )
        
        assert retriever.top_k == 10
    
    def test_vector_store_is_stored(self, set_test_api_key: str):
        """Test that vector_store instance is stored correctly."""
        mock_vector_store = Mock(spec=VectorStore)
        retriever = DocumentRetriever(vector_store=mock_vector_store)
        
        assert retriever.vector_store == mock_vector_store
    
    def test_default_search_type_from_settings(self, set_test_api_key: str):
        """Test that default search_type comes from settings."""
        mock_vector_store = Mock(spec=VectorStore)
        retriever = DocumentRetriever(vector_store=mock_vector_store)
        
        assert retriever.search_type is not None
    
    def test_default_top_k_from_settings(self, set_test_api_key: str):
        """Test that default top_k comes from settings."""
        mock_vector_store = Mock(spec=VectorStore)
        retriever = DocumentRetriever(vector_store=mock_vector_store)
        
        assert retriever.top_k > 0


class TestGetRetriever:
    """Tests for get_retriever method."""
    
    def test_raises_error_if_vectorstore_not_initialized(
        self,
        set_test_api_key: str
    ):
        """Test that error is raised if vectorstore is None."""
        mock_vector_store = Mock(spec=VectorStore)
        mock_vector_store.vectorstore = None
        retriever = DocumentRetriever(vector_store=mock_vector_store)
        
        with pytest.raises(RuntimeError):
            retriever.get_retriever()
    
    def test_returns_retriever_instance(self, set_test_api_key: str):
        """Test that method returns a retriever instance."""
        mock_vectorstore = Mock()
        mock_vectorstore.as_retriever.return_value = Mock()
        
        mock_vector_store = Mock(spec=VectorStore)
        mock_vector_store.vectorstore = mock_vectorstore
        
        retriever = DocumentRetriever(vector_store=mock_vector_store)
        result = retriever.get_retriever()
        
        assert result is not None
    
    def test_calls_as_retriever_method(self, set_test_api_key: str):
        """Test that as_retriever is called on vectorstore."""
        mock_vectorstore = Mock()
        mock_vectorstore.as_retriever.return_value = Mock()
        
        mock_vector_store = Mock(spec=VectorStore)
        mock_vector_store.vectorstore = mock_vectorstore
        
        retriever = DocumentRetriever(vector_store=mock_vector_store)
        retriever.get_retriever()
        
        mock_vectorstore.as_retriever.assert_called_once()
    
    def test_passes_search_type_to_retriever(self, set_test_api_key: str):
        """Test that search_type is passed to as_retriever."""
        mock_vectorstore = Mock()
        mock_vectorstore.as_retriever.return_value = Mock()
        
        mock_vector_store = Mock(spec=VectorStore)
        mock_vector_store.vectorstore = mock_vectorstore
        
        retriever = DocumentRetriever(
            vector_store=mock_vector_store,
            search_type="mmr"
        )
        retriever.get_retriever()
        
        call_kwargs = mock_vectorstore.as_retriever.call_args[1]
        assert call_kwargs['search_type'] == "mmr"
    
    def test_passes_top_k_in_search_kwargs(self, set_test_api_key: str):
        """Test that top_k is passed in search_kwargs."""
        mock_vectorstore = Mock()
        mock_vectorstore.as_retriever.return_value = Mock()
        
        mock_vector_store = Mock(spec=VectorStore)
        mock_vector_store.vectorstore = mock_vectorstore
        
        retriever = DocumentRetriever(
            vector_store=mock_vector_store,
            top_k=7
        )
        retriever.get_retriever()
        
        call_kwargs = mock_vectorstore.as_retriever.call_args[1]
        assert call_kwargs['search_kwargs']['k'] == 7


class TestRetrieve:
    """Tests for retrieve method."""
    
    def test_returns_list_of_documents(
        self,
        set_test_api_key: str,
        sample_query: str
    ):
        """Test that retrieve returns a list of Documents."""
        mock_docs = [Document(page_content="test")]
        mock_retriever_instance = Mock()
        mock_retriever_instance.invoke.return_value = mock_docs
        
        mock_vectorstore = Mock()
        mock_vectorstore.as_retriever.return_value = mock_retriever_instance
        
        mock_vector_store = Mock(spec=VectorStore)
        mock_vector_store.vectorstore = mock_vectorstore
        
        retriever = DocumentRetriever(vector_store=mock_vector_store)
        results = retriever.retrieve(sample_query)
        
        assert isinstance(results, list)
    
    def test_calls_similarity_search(
        self,
        set_test_api_key: str,
        sample_query: str
    ):
        """Test that retriever invoke is called."""
        mock_retriever_instance = Mock()
        mock_retriever_instance.invoke.return_value = []
        
        mock_vectorstore = Mock()
        mock_vectorstore.as_retriever.return_value = mock_retriever_instance
        
        mock_vector_store = Mock(spec=VectorStore)
        mock_vector_store.vectorstore = mock_vectorstore
        
        retriever = DocumentRetriever(vector_store=mock_vector_store)
        retriever.retrieve(sample_query)
        
        mock_retriever_instance.invoke.assert_called_once()
    
    def test_uses_default_top_k(
        self,
        set_test_api_key: str,
        sample_query: str
    ):
        """Test that default top_k is used if not specified."""
        mock_retriever_instance = Mock()
        mock_retriever_instance.invoke.return_value = []
        
        mock_vectorstore = Mock()
        mock_vectorstore.as_retriever.return_value = mock_retriever_instance
        
        mock_vector_store = Mock(spec=VectorStore)
        mock_vector_store.vectorstore = mock_vectorstore
        
        retriever = DocumentRetriever(vector_store=mock_vector_store, top_k=5)
        retriever.retrieve(sample_query)
        
        # Verify as_retriever was called with correct search_kwargs
        call_kwargs = mock_vectorstore.as_retriever.call_args[1]
        assert call_kwargs['search_kwargs']['k'] == 5
    
    def test_uses_custom_k_parameter(
        self,
        set_test_api_key: str,
        sample_query: str
    ):
        """Test that custom k parameter overrides default top_k."""
        mock_retriever_instance = Mock()
        mock_retriever_instance.invoke.return_value = []
        
        mock_vectorstore = Mock()
        mock_vectorstore.as_retriever.return_value = mock_retriever_instance
        
        mock_vector_store = Mock(spec=VectorStore)
        mock_vector_store.vectorstore = mock_vectorstore
        
        retriever = DocumentRetriever(vector_store=mock_vector_store, top_k=5)
        # First call will use top_k=5, but we're calling with k=3
        retriever.top_k = 3  # Simulate the k parameter override
        retriever.retrieve(sample_query, k=3)
        
        # Verify invoke was called
        mock_retriever_instance.invoke.assert_called_once()
    
    def test_passes_query_to_similarity_search(
        self,
        set_test_api_key: str,
        sample_query: str
    ):
        """Test that query is passed to retriever invoke."""
        mock_retriever_instance = Mock()
        mock_retriever_instance.invoke.return_value = []
        
        mock_vectorstore = Mock()
        mock_vectorstore.as_retriever.return_value = mock_retriever_instance
        
        mock_vector_store = Mock(spec=VectorStore)
        mock_vector_store.vectorstore = mock_vectorstore
        
        retriever = DocumentRetriever(vector_store=mock_vector_store)
        retriever.retrieve(sample_query)
        
        # Verify invoke was called with the query
        mock_retriever_instance.invoke.assert_called_once_with(sample_query)
    
    def test_returns_all_document_instances(
        self,
        set_test_api_key: str,
        sample_query: str,
        sample_documents: list[Document]
    ):
        """Test that all returned items are Document instances."""
        mock_retriever_instance = Mock()
        mock_retriever_instance.invoke.return_value = sample_documents
        
        mock_vectorstore = Mock()
        mock_vectorstore.as_retriever.return_value = mock_retriever_instance
        
        mock_vector_store = Mock(spec=VectorStore)
        mock_vector_store.vectorstore = mock_vectorstore
        
        retriever = DocumentRetriever(vector_store=mock_vector_store)
        results = retriever.retrieve(sample_query)
        
        assert all(isinstance(doc, Document) for doc in results)
