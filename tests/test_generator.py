"""Unit tests for generator module."""

import pytest
from langchain_core.documents import Document

from src.generator import ResponseGenerator
from src.ports.llm_provider import BaseLLMProvider


class TestResponseGeneratorInitialization:
    """Tests for ResponseGenerator initialization."""

    def test_initialization_requires_llm_provider(self):
        """Test that ResponseGenerator requires llm_provider."""
        with pytest.raises(TypeError):
            ResponseGenerator()  # type: ignore

    def test_initialization_with_provider(self, mock_llm_provider: BaseLLMProvider):
        """Test that ResponseGenerator initializes with a provider."""
        generator = ResponseGenerator(llm_provider=mock_llm_provider)
        assert generator is not None

    def test_stores_provider(self, mock_llm_provider: BaseLLMProvider):
        """Test that the injected provider is stored."""
        generator = ResponseGenerator(llm_provider=mock_llm_provider)
        assert generator.llm_provider is mock_llm_provider

    def test_default_system_prompt_is_set(self, mock_llm_provider: BaseLLMProvider):
        """Test that default system prompt is set when none provided."""
        generator = ResponseGenerator(llm_provider=mock_llm_provider)
        assert generator.system_prompt_template is not None

    def test_default_system_prompt_contains_context_placeholder(
        self, mock_llm_provider: BaseLLMProvider
    ):
        """Test that default system prompt contains {context} placeholder."""
        generator = ResponseGenerator(llm_provider=mock_llm_provider)
        assert "{context}" in generator.system_prompt_template

    def test_custom_system_prompt_is_stored(self, mock_llm_provider: BaseLLMProvider):
        """Test that a custom system prompt overrides the default."""
        custom_prompt = "Custom instructions: {context}"
        generator = ResponseGenerator(
            llm_provider=mock_llm_provider,
            system_prompt=custom_prompt,
        )
        assert generator.system_prompt_template == custom_prompt


class TestFormatContext:
    """Tests for _format_context method."""

    def test_formats_single_document(
        self,
        mock_llm_provider: BaseLLMProvider,
        sample_documents: list[Document],
    ):
        """Test that a single document is formatted correctly."""
        generator = ResponseGenerator(llm_provider=mock_llm_provider)
        context = generator._format_context([sample_documents[0]])
        assert isinstance(context, str)

    def test_includes_document_content(
        self,
        mock_llm_provider: BaseLLMProvider,
        sample_documents: list[Document],
    ):
        """Test that document content appears in formatted context."""
        generator = ResponseGenerator(llm_provider=mock_llm_provider)
        context = generator._format_context(sample_documents[:1])
        assert sample_documents[0].page_content in context

    def test_numbers_sources_sequentially(
        self,
        mock_llm_provider: BaseLLMProvider,
        sample_documents: list[Document],
    ):
        """Test that sources are numbered sequentially."""
        generator = ResponseGenerator(llm_provider=mock_llm_provider)
        context = generator._format_context(sample_documents)
        assert "Source 1:" in context
        assert "Source 2:" in context

    def test_includes_all_documents(
        self,
        mock_llm_provider: BaseLLMProvider,
        sample_documents: list[Document],
    ):
        """Test that all document contents are included."""
        generator = ResponseGenerator(llm_provider=mock_llm_provider)
        context = generator._format_context(sample_documents)
        for doc in sample_documents:
            assert doc.page_content in context

    def test_empty_documents_returns_empty_string(
        self, mock_llm_provider: BaseLLMProvider
    ):
        """Test that empty document list returns empty string."""
        generator = ResponseGenerator(llm_provider=mock_llm_provider)
        assert generator._format_context([]) == ""


class TestGenerate:
    """Tests for generate method."""

    def test_returns_tuple(
        self,
        mock_llm_provider: BaseLLMProvider,
        sample_query: str,
        sample_documents: list[Document],
    ):
        """Test that generate returns a (str, list) tuple."""
        generator = ResponseGenerator(llm_provider=mock_llm_provider)
        result = generator.generate(sample_query, sample_documents)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_answer_string(
        self,
        mock_llm_provider: BaseLLMProvider,
        sample_query: str,
        sample_documents: list[Document],
    ):
        """Test that first element of tuple is a string answer."""
        generator = ResponseGenerator(llm_provider=mock_llm_provider)
        answer, _ = generator.generate(sample_query, sample_documents)
        assert isinstance(answer, str)

    def test_returns_sources_unchanged(
        self,
        mock_llm_provider: BaseLLMProvider,
        sample_query: str,
        sample_documents: list[Document],
    ):
        """Test that source documents are returned unchanged."""
        generator = ResponseGenerator(llm_provider=mock_llm_provider)
        _, sources = generator.generate(sample_query, sample_documents)
        assert sources == sample_documents

    def test_delegates_to_provider(
        self,
        mock_llm_provider: BaseLLMProvider,
        sample_query: str,
        sample_documents: list[Document],
    ):
        """Test that generate delegates to llm_provider.generate."""
        generator = ResponseGenerator(llm_provider=mock_llm_provider)
        generator.generate(sample_query, sample_documents)
        mock_llm_provider.generate.assert_called_once()  # type: ignore

    def test_provider_receives_query_in_user_message(
        self,
        mock_llm_provider: BaseLLMProvider,
        sample_query: str,
        sample_documents: list[Document],
    ):
        """Test that the query appears in the user_message sent to the provider."""
        generator = ResponseGenerator(llm_provider=mock_llm_provider)
        generator.generate(sample_query, sample_documents)
        _, user_message = mock_llm_provider.generate.call_args[0]  # type: ignore
        assert sample_query in user_message

    def test_provider_receives_context_in_system_prompt(
        self,
        mock_llm_provider: BaseLLMProvider,
        sample_query: str,
        sample_documents: list[Document],
    ):
        """Test that document content appears in the system_prompt sent to the provider."""
        generator = ResponseGenerator(llm_provider=mock_llm_provider)
        generator.generate(sample_query, sample_documents)
        system_prompt, _ = mock_llm_provider.generate.call_args[0]  # type: ignore
        assert sample_documents[0].page_content in system_prompt

    def test_answer_comes_from_provider(
        self,
        mock_llm_provider: BaseLLMProvider,
        sample_query: str,
        sample_documents: list[Document],
    ):
        """Test that the returned answer is exactly what the provider returns."""
        mock_llm_provider.generate.return_value = "Expected answer"  # type: ignore
        generator = ResponseGenerator(llm_provider=mock_llm_provider)
        answer, _ = generator.generate(sample_query, sample_documents)
        assert answer == "Expected answer"

    def test_propagates_provider_exception(
        self,
        mock_llm_provider: BaseLLMProvider,
        sample_query: str,
        sample_documents: list[Document],
    ):
        """Test that exceptions from the provider are re-raised."""
        mock_llm_provider.generate.side_effect = RuntimeError("provider error")  # type: ignore
        generator = ResponseGenerator(llm_provider=mock_llm_provider)
        with pytest.raises(RuntimeError, match="provider error"):
            generator.generate(sample_query, sample_documents)

