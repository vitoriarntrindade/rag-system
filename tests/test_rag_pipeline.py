"""Unit tests for rag_pipeline module."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from src.ports.embedding_provider import BaseEmbeddingProvider
from src.ports.llm_provider import BaseLLMProvider
from src.rag_pipeline import RAGPipeline


def make_pipeline(
    mock_llm_provider: BaseLLMProvider,
    mock_embedding_provider: BaseEmbeddingProvider,
    **kwargs,
) -> RAGPipeline:
    """Helper: build a RAGPipeline with mocked providers (no real API call)."""
    return RAGPipeline(
        api_key="",
        llm_provider=mock_llm_provider,
        embedding_provider=mock_embedding_provider,
        **kwargs,
    )


class TestRAGPipelineInitialization:
    """Tests for RAGPipeline initialization."""

    def test_raises_error_for_missing_api_key_without_providers(self):
        """Test that missing api_key raises ValueError when providers are not injected."""
        with pytest.raises(ValueError):
            RAGPipeline(api_key="")

    def test_raises_error_for_none_api_key_without_providers(self):
        """Test that None api_key raises ValueError when providers are not injected."""
        with pytest.raises(ValueError):
            RAGPipeline(api_key=None)  # type: ignore

    def test_initialization_with_injected_providers(
        self,
        mock_llm_provider: BaseLLMProvider,
        mock_embedding_provider: BaseEmbeddingProvider,
    ):
        """Test that pipeline initializes when both providers are injected."""
        pipeline = make_pipeline(mock_llm_provider, mock_embedding_provider)
        assert pipeline is not None

    def test_document_loader_is_created(
        self,
        mock_llm_provider: BaseLLMProvider,
        mock_embedding_provider: BaseEmbeddingProvider,
    ):
        """Test that document_loader instance is created."""
        pipeline = make_pipeline(mock_llm_provider, mock_embedding_provider)
        assert pipeline.document_loader is not None

    def test_text_processor_is_created(
        self,
        mock_llm_provider: BaseLLMProvider,
        mock_embedding_provider: BaseEmbeddingProvider,
    ):
        """Test that text_processor instance is created."""
        pipeline = make_pipeline(mock_llm_provider, mock_embedding_provider)
        assert pipeline.text_processor is not None

    def test_vector_store_is_created(
        self,
        mock_llm_provider: BaseLLMProvider,
        mock_embedding_provider: BaseEmbeddingProvider,
    ):
        """Test that vector_store instance is created."""
        pipeline = make_pipeline(mock_llm_provider, mock_embedding_provider)
        assert pipeline.vector_store is not None

    def test_generator_is_created(
        self,
        mock_llm_provider: BaseLLMProvider,
        mock_embedding_provider: BaseEmbeddingProvider,
    ):
        """Test that generator instance is created."""
        pipeline = make_pipeline(mock_llm_provider, mock_embedding_provider)
        assert pipeline.generator is not None

    def test_retriever_initially_none(
        self,
        mock_llm_provider: BaseLLMProvider,
        mock_embedding_provider: BaseEmbeddingProvider,
    ):
        """Test that retriever is None before initialization."""
        pipeline = make_pipeline(mock_llm_provider, mock_embedding_provider)
        assert pipeline.retriever is None

    def test_is_initialized_flag_starts_false(
        self,
        mock_llm_provider: BaseLLMProvider,
        mock_embedding_provider: BaseEmbeddingProvider,
    ):
        """Test that _is_initialized flag starts as False."""
        pipeline = make_pipeline(mock_llm_provider, mock_embedding_provider)
        assert pipeline._is_initialized is False

    def test_providers_reach_components(
        self,
        mock_llm_provider: BaseLLMProvider,
        mock_embedding_provider: BaseEmbeddingProvider,
    ):
        """Test that injected providers are forwarded to generator and vector_store."""
        pipeline = make_pipeline(mock_llm_provider, mock_embedding_provider)
        assert pipeline.generator.llm_provider is mock_llm_provider
        assert pipeline.vector_store.embedding_provider is mock_embedding_provider

    def test_initialization_with_custom_vector_store_path(
        self,
        mock_llm_provider: BaseLLMProvider,
        mock_embedding_provider: BaseEmbeddingProvider,
        temp_vector_store_path: Path,
    ):
        """Test that custom vector_store_path is set on vector_store."""
        pipeline = make_pipeline(
            mock_llm_provider,
            mock_embedding_provider,
            vector_store_path=temp_vector_store_path,
        )
        assert pipeline.vector_store.persist_directory == temp_vector_store_path

    def test_initialization_with_custom_chunk_size(
        self,
        mock_llm_provider: BaseLLMProvider,
        mock_embedding_provider: BaseEmbeddingProvider,
    ):
        """Test that custom chunk_size is passed to text_processor."""
        pipeline = make_pipeline(
            mock_llm_provider, mock_embedding_provider, chunk_size=500
        )
        assert pipeline.text_processor.chunk_size == 500

    def test_initialization_with_custom_chunk_overlap(
        self,
        mock_llm_provider: BaseLLMProvider,
        mock_embedding_provider: BaseEmbeddingProvider,
    ):
        """Test that custom chunk_overlap is passed to text_processor."""
        pipeline = make_pipeline(
            mock_llm_provider, mock_embedding_provider, chunk_overlap=50
        )
        assert pipeline.text_processor.chunk_overlap == 50


class TestIngestDocuments:
    """Tests for ingest_documents method."""

    def test_raises_error_without_file_or_directory(
        self,
        mock_llm_provider: BaseLLMProvider,
        mock_embedding_provider: BaseEmbeddingProvider,
        temp_vector_store_path: Path,
    ):
        """Test that error is raised when neither file nor directory is provided."""
        pipeline = make_pipeline(
            mock_llm_provider,
            mock_embedding_provider,
            vector_store_path=temp_vector_store_path,
        )
        with pytest.raises(ValueError, match="Either file_path or directory must be provided"):
            pipeline.ingest_documents()

    @patch("src.rag_pipeline.VectorStore")
    @patch("src.rag_pipeline.TextProcessor")
    def test_loads_single_file(
        self,
        mock_text_processor_class: Mock,
        mock_vector_store_class: Mock,
        mock_llm_provider: BaseLLMProvider,
        mock_embedding_provider: BaseEmbeddingProvider,
        sample_txt_file: Path,
    ):
        """Test that single file ingestion sets _is_initialized."""
        mock_text_proc = Mock()
        mock_text_proc.split_documents.return_value = [Document(page_content="chunk")]
        mock_text_processor_class.return_value = mock_text_proc

        mock_vector = Mock()
        mock_vector.persist_directory = Mock()
        mock_vector.persist_directory.exists.return_value = False
        mock_vector.create_from_documents.return_value = Mock()
        mock_vector.vectorstore = Mock()
        mock_vector_store_class.return_value = mock_vector

        pipeline = make_pipeline(mock_llm_provider, mock_embedding_provider)
        pipeline.ingest_documents(file_path=sample_txt_file)

        assert pipeline._is_initialized

    @patch("src.rag_pipeline.VectorStore")
    @patch("src.rag_pipeline.TextProcessor")
    @patch("src.rag_pipeline.DocumentLoader")
    def test_loads_directory(
        self,
        mock_loader_class: Mock,
        mock_text_processor_class: Mock,
        mock_vector_store_class: Mock,
        mock_llm_provider: BaseLLMProvider,
        mock_embedding_provider: BaseEmbeddingProvider,
        temp_directory: Path,
    ):
        """Test that directory ingestion calls load_documents."""
        mock_loader = Mock()
        mock_loader.load_documents.return_value = [Document(page_content="doc")]
        mock_loader_class.return_value = mock_loader

        mock_text_proc = Mock()
        mock_text_proc.split_documents.return_value = [Document(page_content="chunk")]
        mock_text_processor_class.return_value = mock_text_proc

        mock_vector = Mock()
        mock_vector.persist_directory = temp_directory / "vector_db"
        mock_vector.create_from_documents.return_value = Mock()
        mock_vector_store_class.return_value = mock_vector

        pipeline = make_pipeline(mock_llm_provider, mock_embedding_provider)
        pipeline.ingest_documents(directory=temp_directory)

        mock_loader.load_documents.assert_called_once()

    @patch("src.rag_pipeline.VectorStore")
    @patch("src.rag_pipeline.TextProcessor")
    def test_creates_retriever_after_ingestion(
        self,
        mock_text_processor_class: Mock,
        mock_vector_store_class: Mock,
        mock_llm_provider: BaseLLMProvider,
        mock_embedding_provider: BaseEmbeddingProvider,
        sample_txt_file: Path,
    ):
        """Test that retriever is created after successful ingestion."""
        mock_text_proc = Mock()
        mock_text_proc.split_documents.return_value = [Document(page_content="chunk")]
        mock_text_processor_class.return_value = mock_text_proc

        mock_vector = Mock()
        mock_vector.persist_directory = Mock()
        mock_vector.persist_directory.exists.return_value = False
        mock_vector.create_from_documents.return_value = Mock()
        mock_vector.vectorstore = Mock()
        mock_vector_store_class.return_value = mock_vector

        pipeline = make_pipeline(mock_llm_provider, mock_embedding_provider)
        pipeline.ingest_documents(file_path=sample_txt_file)

        assert pipeline.retriever is not None


class TestQuery:
    """Tests for query method."""

    def test_raises_error_when_not_initialized(
        self,
        mock_llm_provider: BaseLLMProvider,
        mock_embedding_provider: BaseEmbeddingProvider,
    ):
        """Test that RuntimeError is raised when querying uninitialized pipeline."""
        pipeline = make_pipeline(mock_llm_provider, mock_embedding_provider)
        with pytest.raises(RuntimeError):
            pipeline.query("test query")

    @patch("src.rag_pipeline.DocumentRetriever")
    def test_returns_tuple(
        self,
        mock_retriever_class: Mock,
        mock_llm_provider: BaseLLMProvider,
        mock_embedding_provider: BaseEmbeddingProvider,
        sample_query: str,
    ):
        """Test that query returns a tuple."""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = [Document(page_content="result")]
        mock_retriever_class.return_value = mock_retriever
        mock_llm_provider.generate.return_value = "answer"  # type: ignore

        pipeline = make_pipeline(mock_llm_provider, mock_embedding_provider)
        pipeline._is_initialized = True
        pipeline.retriever = mock_retriever

        result = pipeline.query(sample_query)
        assert isinstance(result, tuple)

    @patch("src.rag_pipeline.DocumentRetriever")
    def test_returns_answer_and_sources(
        self,
        mock_retriever_class: Mock,
        mock_llm_provider: BaseLLMProvider,
        mock_embedding_provider: BaseEmbeddingProvider,
        sample_query: str,
    ):
        """Test that query returns the answer from the provider and sources."""
        source_doc = Document(page_content="source")
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = [source_doc]
        mock_retriever_class.return_value = mock_retriever
        mock_llm_provider.generate.return_value = "Generated answer"  # type: ignore

        pipeline = make_pipeline(mock_llm_provider, mock_embedding_provider)
        pipeline._is_initialized = True
        pipeline.retriever = mock_retriever

        answer, sources = pipeline.query(sample_query)
        assert answer == "Generated answer"
        assert isinstance(sources, list)


class TestInteractiveChat:
    """Tests for interactive_chat method."""

    def test_raises_error_when_not_initialized(
        self,
        mock_llm_provider: BaseLLMProvider,
        mock_embedding_provider: BaseEmbeddingProvider,
    ):
        """Test that RuntimeError is raised when chatting with uninitialized pipeline."""
        pipeline = make_pipeline(mock_llm_provider, mock_embedding_provider)
        with pytest.raises(RuntimeError):
            pipeline.interactive_chat()


class TestRAGPipelineIntegration:
    """Integration tests for RAGPipeline workflow."""

    @patch("src.rag_pipeline.DocumentRetriever")
    def test_complete_workflow(
        self,
        mock_retriever_class: Mock,
        mock_llm_provider: BaseLLMProvider,
        mock_embedding_provider: BaseEmbeddingProvider,
        sample_query: str,
    ):
        """Test complete workflow: init → mark initialized → query."""
        source_doc = Document(page_content="result")
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = [source_doc]
        mock_retriever_class.return_value = mock_retriever
        mock_llm_provider.generate.return_value = "answer"  # type: ignore

        pipeline = make_pipeline(mock_llm_provider, mock_embedding_provider)
        pipeline._is_initialized = True
        pipeline.retriever = mock_retriever

        answer, sources = pipeline.query(sample_query)
        assert isinstance(answer, str)
        assert isinstance(sources, list)



