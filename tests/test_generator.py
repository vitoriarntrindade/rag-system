"""Unit tests for generator module."""

from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from rag_system.src.generator import ResponseGenerator


class TestResponseGeneratorInitialization:
    """Tests for ResponseGenerator initialization."""
    
    def test_initialization_with_defaults(self, set_test_api_key: str):
        """Test that ResponseGenerator initializes with default settings."""
        generator = ResponseGenerator()
        assert generator is not None
    
    def test_initialization_with_custom_model(self, set_test_api_key: str):
        """Test that custom model is set correctly."""
        generator = ResponseGenerator(model="gpt-4")
        assert generator.model_name == "gpt-4"
    
    def test_initialization_with_custom_temperature(self, set_test_api_key: str):
        """Test that custom temperature is set correctly."""
        generator = ResponseGenerator(temperature=0.7)
        assert generator.temperature == 0.7
    
    def test_initialization_with_custom_system_prompt(self, set_test_api_key: str):
        """Test that custom system_prompt is set correctly."""
        custom_prompt = "Custom instructions: {context}"
        generator = ResponseGenerator(system_prompt=custom_prompt)
        assert generator.system_prompt_template == custom_prompt
    
    def test_llm_instance_is_created(self, set_test_api_key: str):
        """Test that LLM instance is created during init."""
        generator = ResponseGenerator()
        assert generator.llm is not None
    
    def test_default_system_prompt_is_set(self, set_test_api_key: str):
        """Test that default system prompt is set."""
        generator = ResponseGenerator()
        assert generator.system_prompt_template is not None
    
    def test_default_system_prompt_contains_context_placeholder(
        self,
        set_test_api_key: str
    ):
        """Test that default system prompt contains context placeholder."""
        generator = ResponseGenerator()
        assert "{context}" in generator.system_prompt_template


class TestFormatContext:
    """Tests for _format_context method."""
    
    def test_formats_single_document(
        self,
        set_test_api_key: str,
        sample_documents: list[Document]
    ):
        """Test that single document is formatted correctly."""
        generator = ResponseGenerator()
        context = generator._format_context([sample_documents[0]])
        
        assert isinstance(context, str)
    
    def test_includes_document_content(
        self,
        set_test_api_key: str,
        sample_documents: list[Document]
    ):
        """Test that document content is included in formatted context."""
        generator = ResponseGenerator()
        context = generator._format_context(sample_documents[:1])
        
        assert sample_documents[0].page_content in context
    
    def test_formats_multiple_documents(
        self,
        set_test_api_key: str,
        sample_documents: list[Document]
    ):
        """Test that multiple documents are formatted correctly."""
        generator = ResponseGenerator()
        context = generator._format_context(sample_documents)
        
        assert "Source 1:" in context
    
    def test_includes_all_documents(
        self,
        set_test_api_key: str,
        sample_documents: list[Document]
    ):
        """Test that all document contents are included."""
        generator = ResponseGenerator()
        context = generator._format_context(sample_documents)
        
        for doc in sample_documents:
            assert doc.page_content in context
    
    def test_numbers_sources_sequentially(
        self,
        set_test_api_key: str,
        sample_documents: list[Document]
    ):
        """Test that sources are numbered sequentially."""
        generator = ResponseGenerator()
        context = generator._format_context(sample_documents)
        
        assert "Source 1:" in context
        assert "Source 2:" in context
    
    def test_empty_documents_returns_empty_string(self, set_test_api_key: str):
        """Test that empty document list returns empty string."""
        generator = ResponseGenerator()
        context = generator._format_context([])
        
        assert context == ""


class TestGenerate:
    """Tests for generate method."""
    
    @patch('src.generator.ChatOpenAI')
    def test_returns_tuple(
        self,
        mock_chat: Mock,
        set_test_api_key: str,
        sample_query: str,
        sample_documents: list[Document]
    ):
        """Test that generate returns a tuple."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="Test answer")
        mock_chat.return_value = mock_llm
        
        generator = ResponseGenerator()
        generator.llm = mock_llm
        result = generator.generate(sample_query, sample_documents)
        
        assert isinstance(result, tuple)
    
    @patch('src.generator.ChatOpenAI')
    def test_returns_answer_and_sources(
        self,
        mock_chat: Mock,
        set_test_api_key: str,
        sample_query: str,
        sample_documents: list[Document]
    ):
        """Test that generate returns answer string and source documents."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="Test answer")
        mock_chat.return_value = mock_llm
        
        generator = ResponseGenerator()
        generator.llm = mock_llm
        answer, sources = generator.generate(sample_query, sample_documents)
        
        assert isinstance(answer, str)
        assert isinstance(sources, list)
    
    @patch('src.generator.ChatOpenAI')
    def test_calls_llm_invoke(
        self,
        mock_chat: Mock,
        set_test_api_key: str,
        sample_query: str,
        sample_documents: list[Document]
    ):
        """Test that LLM invoke method is called."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="Test answer")
        mock_chat.return_value = mock_llm
        
        generator = ResponseGenerator()
        generator.llm = mock_llm
        generator.generate(sample_query, sample_documents)
        
        mock_llm.invoke.assert_called_once()
    
    @patch('src.generator.ChatOpenAI')
    def test_includes_query_in_messages(
        self,
        mock_chat: Mock,
        set_test_api_key: str,
        sample_query: str,
        sample_documents: list[Document]
    ):
        """Test that query is included in messages sent to LLM."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="Test answer")
        mock_chat.return_value = mock_llm
        
        generator = ResponseGenerator()
        generator.llm = mock_llm
        generator.generate(sample_query, sample_documents)
        
        call_args = mock_llm.invoke.call_args[0][0]
        messages_str = str(call_args)
        assert sample_query in messages_str
    
    @patch('src.generator.ChatOpenAI')
    def test_includes_context_in_messages(
        self,
        mock_chat: Mock,
        set_test_api_key: str,
        sample_query: str,
        sample_documents: list[Document]
    ):
        """Test that document context is included in messages."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="Test answer")
        mock_chat.return_value = mock_llm
        
        generator = ResponseGenerator()
        generator.llm = mock_llm
        generator.generate(sample_query, sample_documents)
        
        call_args = mock_llm.invoke.call_args[0][0]
        messages_str = str(call_args)
        assert sample_documents[0].page_content in messages_str
    
    @patch('src.generator.ChatOpenAI')
    def test_returns_sources_unchanged(
        self,
        mock_chat: Mock,
        set_test_api_key: str,
        sample_query: str,
        sample_documents: list[Document]
    ):
        """Test that source documents are returned unchanged."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="Test answer")
        mock_chat.return_value = mock_llm
        
        generator = ResponseGenerator()
        generator.llm = mock_llm
        answer, sources = generator.generate(sample_query, sample_documents)
        
        assert sources == sample_documents
    
    @patch('src.generator.ChatOpenAI')
    def test_handles_llm_exception(
        self,
        mock_chat: Mock,
        set_test_api_key: str,
        sample_query: str,
        sample_documents: list[Document]
    ):
        """Test that LLM exceptions are properly propagated."""
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("LLM error")
        mock_chat.return_value = mock_llm
        
        generator = ResponseGenerator()
        generator.llm = mock_llm
        
        with pytest.raises(Exception):
            generator.generate(sample_query, sample_documents)
    
    @patch('src.generator.ChatOpenAI')
    def test_extracts_content_from_response(
        self,
        mock_chat: Mock,
        set_test_api_key: str,
        sample_query: str,
        sample_documents: list[Document]
    ):
        """Test that answer content is extracted from LLM response."""
        mock_llm = Mock()
        expected_answer = "This is the generated answer"
        mock_llm.invoke.return_value = AIMessage(content=expected_answer)
        mock_chat.return_value = mock_llm
        
        generator = ResponseGenerator()
        generator.llm = mock_llm
        answer, sources = generator.generate(sample_query, sample_documents)
        
        assert answer == expected_answer


class TestResponseGeneratorConfiguration:
    """Tests for ResponseGenerator configuration options."""
    
    def test_default_model_from_settings(self, set_test_api_key: str):
        """Test that default model comes from settings."""
        generator = ResponseGenerator()
        assert generator.model_name is not None
    
    def test_default_temperature_from_settings(self, set_test_api_key: str):
        """Test that default temperature comes from settings."""
        generator = ResponseGenerator()
        assert generator.temperature is not None
    
    def test_custom_values_override_defaults(self, set_test_api_key: str):
        """Test that custom values override default settings."""
        generator = ResponseGenerator(model="gpt-4", temperature=0.9)
        assert generator.model_name == "gpt-4"
        assert generator.temperature == 0.9
