"""Unit tests for vector_store module."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain_core.documents import Document

from src.ports.embedding_provider import BaseEmbeddingProvider
from src.vector_store import VectorStore


class TestVectorStoreInitialization:
    """Tests for VectorStore initialization."""

    def test_initialization_requires_embedding_provider(self):
        """Test that VectorStore requires an embedding_provider."""
        with pytest.raises(TypeError):
            VectorStore()  # type: ignore

    def test_initialization_with_provider(
        self,
        mock_embedding_provider: BaseEmbeddingProvider,
        temp_vector_store_path: Path,
    ):
        """Test that VectorStore initializes with a provider and path."""
        store = VectorStore(
            embedding_provider=mock_embedding_provider,
            persist_directory=temp_vector_store_path,
        )
        assert store is not None

    def test_stores_embedding_provider(
        self,
        mock_embedding_provider: BaseEmbeddingProvider,
        temp_vector_store_path: Path,
    ):
        """Test that the injected provider is stored."""
        store = VectorStore(
            embedding_provider=mock_embedding_provider,
            persist_directory=temp_vector_store_path,
        )
        assert store.embedding_provider is mock_embedding_provider

    def test_initialization_with_custom_path(
        self,
        mock_embedding_provider: BaseEmbeddingProvider,
        temp_vector_store_path: Path,
    ):
        """Test that custom persist_directory is set correctly."""
        store = VectorStore(
            embedding_provider=mock_embedding_provider,
            persist_directory=temp_vector_store_path,
        )
        assert store.persist_directory == temp_vector_store_path

    def test_vectorstore_is_initially_none(
        self,
        mock_embedding_provider: BaseEmbeddingProvider,
        temp_vector_store_path: Path,
    ):
        """Test that vectorstore is None before loading or creating."""
        store = VectorStore(
            embedding_provider=mock_embedding_provider,
            persist_directory=temp_vector_store_path,
        )
        assert store.vectorstore is None


class TestCreateFromDocuments:
    """Tests for create_from_documents method."""

    @patch("src.vector_store.Chroma")
    def test_creates_vectorstore_from_documents(
        self,
        mock_chroma: MagicMock,
        mock_embedding_provider: BaseEmbeddingProvider,
        sample_documents: list[Document],
        temp_vector_store_path: Path,
    ):
        """Test that vectorstore is created from documents."""
        mock_chroma.from_documents.return_value = Mock()
        store = VectorStore(
            embedding_provider=mock_embedding_provider,
            persist_directory=temp_vector_store_path,
        )
        result = store.create_from_documents(sample_documents)
        assert result is not None

    @patch("src.vector_store.Chroma")
    def test_calls_chroma_from_documents(
        self,
        mock_chroma: MagicMock,
        mock_embedding_provider: BaseEmbeddingProvider,
        sample_documents: list[Document],
        temp_vector_store_path: Path,
    ):
        """Test that Chroma.from_documents is called."""
        mock_chroma.from_documents.return_value = Mock()
        store = VectorStore(
            embedding_provider=mock_embedding_provider,
            persist_directory=temp_vector_store_path,
        )
        store.create_from_documents(sample_documents)
        mock_chroma.from_documents.assert_called_once()

    @patch("src.vector_store.Chroma")
    def test_chroma_receives_embedding_provider(
        self,
        mock_chroma: MagicMock,
        mock_embedding_provider: BaseEmbeddingProvider,
        sample_documents: list[Document],
        temp_vector_store_path: Path,
    ):
        """Test that the embedding provider is forwarded to Chroma."""
        mock_chroma.from_documents.return_value = Mock()
        store = VectorStore(
            embedding_provider=mock_embedding_provider,
            persist_directory=temp_vector_store_path,
        )
        store.create_from_documents(sample_documents)
        call_kwargs = mock_chroma.from_documents.call_args[1]
        assert call_kwargs["embedding"] is mock_embedding_provider

    @patch("src.vector_store.Chroma")
    def test_creates_persist_directory(
        self,
        mock_chroma: MagicMock,
        mock_embedding_provider: BaseEmbeddingProvider,
        sample_documents: list[Document],
        temp_directory: Path,
    ):
        """Test that nested persist directory is created if it doesn't exist."""
        nested_path = temp_directory / "nested" / "vector_db"
        mock_chroma.from_documents.return_value = Mock()
        store = VectorStore(
            embedding_provider=mock_embedding_provider,
            persist_directory=nested_path,
        )
        store.create_from_documents(sample_documents)
        assert nested_path.parent.exists()

    @patch("src.vector_store.Chroma")
    def test_sets_vectorstore_attribute(
        self,
        mock_chroma: MagicMock,
        mock_embedding_provider: BaseEmbeddingProvider,
        sample_documents: list[Document],
        temp_vector_store_path: Path,
    ):
        """Test that vectorstore attribute is set after creation."""
        mock_vs = Mock()
        mock_chroma.from_documents.return_value = mock_vs
        store = VectorStore(
            embedding_provider=mock_embedding_provider,
            persist_directory=temp_vector_store_path,
        )
        store.create_from_documents(sample_documents)
        assert store.vectorstore is mock_vs


class TestLoadExisting:
    """Tests for load_existing method."""

    def test_raises_error_for_nonexistent_store(
        self,
        mock_embedding_provider: BaseEmbeddingProvider,
        temp_directory: Path,
    ):
        """Test that loading a nonexistent store raises FileNotFoundError."""
        store = VectorStore(
            embedding_provider=mock_embedding_provider,
            persist_directory=temp_directory / "nonexistent_db",
        )
        with pytest.raises(FileNotFoundError):
            store.load_existing()

    @patch("src.vector_store.Chroma")
    def test_loads_existing_vectorstore(
        self,
        mock_chroma: MagicMock,
        mock_embedding_provider: BaseEmbeddingProvider,
        temp_vector_store_path: Path,
    ):
        """Test that existing vectorstore is loaded successfully."""
        temp_vector_store_path.mkdir(parents=True)
        mock_chroma.return_value = Mock()
        store = VectorStore(
            embedding_provider=mock_embedding_provider,
            persist_directory=temp_vector_store_path,
        )
        result = store.load_existing()
        assert result is not None

    @patch("src.vector_store.Chroma")
    def test_calls_chroma_constructor(
        self,
        mock_chroma: MagicMock,
        mock_embedding_provider: BaseEmbeddingProvider,
        temp_vector_store_path: Path,
    ):
        """Test that Chroma constructor is called when loading."""
        temp_vector_store_path.mkdir(parents=True)
        mock_chroma.return_value = Mock()
        store = VectorStore(
            embedding_provider=mock_embedding_provider,
            persist_directory=temp_vector_store_path,
        )
        store.load_existing()
        mock_chroma.assert_called_once()

    @patch("src.vector_store.Chroma")
    def test_sets_vectorstore_attribute(
        self,
        mock_chroma: MagicMock,
        mock_embedding_provider: BaseEmbeddingProvider,
        temp_vector_store_path: Path,
    ):
        """Test that vectorstore attribute is set after loading."""
        temp_vector_store_path.mkdir(parents=True)
        mock_vs = Mock()
        mock_chroma.return_value = mock_vs
        store = VectorStore(
            embedding_provider=mock_embedding_provider,
            persist_directory=temp_vector_store_path,
        )
        store.load_existing()
        assert store.vectorstore is mock_vs


class TestGetOrCreate:
    """Tests for get_or_create method."""

    @patch("src.vector_store.Chroma")
    def test_loads_existing_if_available(
        self,
        mock_chroma: MagicMock,
        mock_embedding_provider: BaseEmbeddingProvider,
        temp_vector_store_path: Path,
    ):
        """Test that existing store is loaded when directory exists."""
        temp_vector_store_path.mkdir(parents=True)
        mock_chroma.return_value = Mock()
        store = VectorStore(
            embedding_provider=mock_embedding_provider,
            persist_directory=temp_vector_store_path,
        )
        result = store.get_or_create()
        assert result is not None

    @patch("src.vector_store.Chroma")
    def test_creates_new_if_not_exists(
        self,
        mock_chroma: MagicMock,
        mock_embedding_provider: BaseEmbeddingProvider,
        sample_documents: list[Document],
        temp_vector_store_path: Path,
    ):
        """Test that new store is created when directory does not exist."""
        mock_chroma.from_documents.return_value = Mock()
        store = VectorStore(
            embedding_provider=mock_embedding_provider,
            persist_directory=temp_vector_store_path,
        )
        result = store.get_or_create(documents=sample_documents)
        assert result is not None

    def test_raises_error_if_no_documents_provided(
        self,
        mock_embedding_provider: BaseEmbeddingProvider,
        temp_vector_store_path: Path,
    ):
        """Test that ValueError is raised when store doesn't exist and no docs given."""
        store = VectorStore(
            embedding_provider=mock_embedding_provider,
            persist_directory=temp_vector_store_path,
        )
        with pytest.raises(ValueError):
            store.get_or_create()


class TestSimilaritySearch:
    """Tests for similarity_search method."""

    def test_returns_list_of_documents(
        self,
        mock_embedding_provider: BaseEmbeddingProvider,
        sample_query: str,
        temp_vector_store_path: Path,
    ):
        """Test that similarity_search returns a list of Documents."""
        mock_vs = Mock()
        mock_vs.similarity_search.return_value = [Document(page_content="result")]
        store = VectorStore(
            embedding_provider=mock_embedding_provider,
            persist_directory=temp_vector_store_path,
        )
        store.vectorstore = mock_vs
        results = store.similarity_search(sample_query)
        assert isinstance(results, list)

    def test_calls_vectorstore_similarity_search(
        self,
        mock_embedding_provider: BaseEmbeddingProvider,
        sample_query: str,
        temp_vector_store_path: Path,
    ):
        """Test that vectorstore.similarity_search is called."""
        mock_vs = Mock()
        mock_vs.similarity_search.return_value = []
        store = VectorStore(
            embedding_provider=mock_embedding_provider,
            persist_directory=temp_vector_store_path,
        )
        store.vectorstore = mock_vs
        store.similarity_search(sample_query, k=5)
        mock_vs.similarity_search.assert_called_once()

    def test_passes_k_parameter(
        self,
        mock_embedding_provider: BaseEmbeddingProvider,
        sample_query: str,
        temp_vector_store_path: Path,
    ):
        """Test that the k parameter is forwarded to similarity_search."""
        mock_vs = Mock()
        mock_vs.similarity_search.return_value = []
        store = VectorStore(
            embedding_provider=mock_embedding_provider,
            persist_directory=temp_vector_store_path,
        )
        store.vectorstore = mock_vs
        store.similarity_search(sample_query, k=3)
        call_kwargs = mock_vs.similarity_search.call_args[1]
        assert call_kwargs["k"] == 3

    def test_raises_if_not_initialized(
        self,
        mock_embedding_provider: BaseEmbeddingProvider,
        sample_query: str,
        temp_vector_store_path: Path,
    ):
        """Test that RuntimeError is raised when vectorstore is None."""
        store = VectorStore(
            embedding_provider=mock_embedding_provider,
            persist_directory=temp_vector_store_path,
        )
        with pytest.raises(RuntimeError):
            store.similarity_search(sample_query)



