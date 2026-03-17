"""Unit tests for OpenAI LLM and Embedding adapters."""

from typing import List
from unittest.mock import MagicMock, patch

import pytest

from src.adapters.embeddings.openai_embeddings import OpenAIEmbeddingAdapter
from src.adapters.llm.openai_llm import OpenAILLMAdapter
from src.ports.embedding_provider import BaseEmbeddingProvider
from src.ports.llm_provider import BaseLLMProvider


# ---------------------------------------------------------------------------
# OpenAILLMAdapter
# ---------------------------------------------------------------------------

class TestOpenAILLMAdapter:
    """Tests for OpenAILLMAdapter."""

    def _make_adapter(self) -> tuple[OpenAILLMAdapter, MagicMock]:
        """Return (adapter, mock_llm) with ChatOpenAI patched."""
        with patch("src.adapters.llm.openai_llm.ChatOpenAI") as mock_cls:
            mock_llm = MagicMock()
            mock_cls.return_value = mock_llm
            adapter = OpenAILLMAdapter(
                model="gpt-3.5-turbo",
                temperature=0.3,
                api_key="test-key",
            )
        adapter._llm = mock_llm
        return adapter, mock_llm

    # ── contract ────────────────────────────────────────────────────────────

    def test_is_base_llm_provider_subclass(self):
        """Adapter must satisfy the BaseLLMProvider port."""
        assert issubclass(OpenAILLMAdapter, BaseLLMProvider)

    def test_instance_is_base_llm_provider(self):
        """Instance must be recognised as a BaseLLMProvider."""
        with patch("src.adapters.llm.openai_llm.ChatOpenAI"):
            adapter = OpenAILLMAdapter("gpt-3.5-turbo", 0.3, "key")
        assert isinstance(adapter, BaseLLMProvider)

    # ── initialisation ──────────────────────────────────────────────────────

    def test_initialisation_creates_chat_openai(self):
        """__init__ must pass correct args to ChatOpenAI."""
        with patch("src.adapters.llm.openai_llm.ChatOpenAI") as mock_cls:
            OpenAILLMAdapter(
                model="gpt-4",
                temperature=0.9,
                api_key="sk-abc",
            )
        mock_cls.assert_called_once_with(model="gpt-4", temperature=0.9, api_key="sk-abc")

    # ── generate ────────────────────────────────────────────────────────────

    def test_generate_returns_string(self):
        """generate() must return the content string from the response."""
        adapter, mock_llm = self._make_adapter()
        mock_llm.invoke.return_value = MagicMock(content="Hello!")
        result = adapter.generate("You are helpful.", "What is 2+2?")
        assert result == "Hello!"

    def test_generate_calls_invoke(self):
        """generate() must call invoke on the underlying LLM exactly once."""
        adapter, mock_llm = self._make_adapter()
        mock_llm.invoke.return_value = MagicMock(content="ok")
        adapter.generate("sys", "user")
        mock_llm.invoke.assert_called_once()

    def test_generate_passes_system_and_human_messages(self):
        """generate() must build SystemMessage + HumanMessage and pass both."""
        from langchain_core.messages import HumanMessage, SystemMessage

        adapter, mock_llm = self._make_adapter()
        mock_llm.invoke.return_value = MagicMock(content="ok")
        adapter.generate("sys prompt", "user msg")

        (messages,), _ = mock_llm.invoke.call_args
        assert any(isinstance(m, SystemMessage) for m in messages)
        assert any(isinstance(m, HumanMessage) for m in messages)
        system = next(m for m in messages if isinstance(m, SystemMessage))
        human = next(m for m in messages if isinstance(m, HumanMessage))
        assert system.content == "sys prompt"
        assert human.content == "user msg"

    def test_generate_propagates_exception(self):
        """generate() must re-raise any exception from the underlying LLM."""
        adapter, mock_llm = self._make_adapter()
        mock_llm.invoke.side_effect = RuntimeError("API error")
        with pytest.raises(RuntimeError, match="API error"):
            adapter.generate("sys", "user")


# ---------------------------------------------------------------------------
# OpenAIEmbeddingAdapter
# ---------------------------------------------------------------------------

class TestOpenAIEmbeddingAdapter:
    """Tests for OpenAIEmbeddingAdapter."""

    def _make_adapter(self) -> tuple[OpenAIEmbeddingAdapter, MagicMock]:
        """Return (adapter, mock_embeddings) with OpenAIEmbeddings patched."""
        with patch("src.adapters.embeddings.openai_embeddings.OpenAIEmbeddings") as mock_cls:
            mock_emb = MagicMock()
            mock_cls.return_value = mock_emb
            adapter = OpenAIEmbeddingAdapter(
                model="text-embedding-3-small",
                api_key="test-key",
            )
        adapter._embeddings = mock_emb
        return adapter, mock_emb

    # ── contract ────────────────────────────────────────────────────────────

    def test_is_base_embedding_provider_subclass(self):
        """Adapter must satisfy the BaseEmbeddingProvider port."""
        assert issubclass(OpenAIEmbeddingAdapter, BaseEmbeddingProvider)

    def test_instance_is_base_embedding_provider(self):
        """Instance must be recognised as a BaseEmbeddingProvider."""
        with patch("src.adapters.embeddings.openai_embeddings.OpenAIEmbeddings"):
            adapter = OpenAIEmbeddingAdapter("text-embedding-3-small", "key")
        assert isinstance(adapter, BaseEmbeddingProvider)

    # ── initialisation ──────────────────────────────────────────────────────

    def test_initialisation_creates_openai_embeddings(self):
        """__init__ must pass correct args to OpenAIEmbeddings."""
        with patch("src.adapters.embeddings.openai_embeddings.OpenAIEmbeddings") as mock_cls:
            OpenAIEmbeddingAdapter(model="text-embedding-ada-002", api_key="sk-xyz")
        mock_cls.assert_called_once_with(model="text-embedding-ada-002", api_key="sk-xyz")

    # ── embed_documents ─────────────────────────────────────────────────────

    def test_embed_documents_returns_list_of_vectors(self):
        """embed_documents() must return a list of float lists."""
        adapter, mock_emb = self._make_adapter()
        mock_emb.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
        result = adapter.embed_documents(["doc1", "doc2"])
        assert result == [[0.1, 0.2], [0.3, 0.4]]

    def test_embed_documents_delegates_to_underlying(self):
        """embed_documents() must delegate to the underlying embeddings object."""
        adapter, mock_emb = self._make_adapter()
        mock_emb.embed_documents.return_value = [[0.0]]
        adapter.embed_documents(["hello"])
        mock_emb.embed_documents.assert_called_once_with(["hello"])

    def test_embed_documents_empty_list(self):
        """embed_documents() must handle an empty list gracefully."""
        adapter, mock_emb = self._make_adapter()
        mock_emb.embed_documents.return_value = []
        result = adapter.embed_documents([])
        assert result == []

    # ── embed_query ─────────────────────────────────────────────────────────

    def test_embed_query_returns_vector(self):
        """embed_query() must return a list of floats."""
        adapter, mock_emb = self._make_adapter()
        mock_emb.embed_query.return_value = [0.1, 0.2, 0.3]
        result = adapter.embed_query("what is AI?")
        assert result == [0.1, 0.2, 0.3]

    def test_embed_query_delegates_to_underlying(self):
        """embed_query() must delegate to the underlying embeddings object."""
        adapter, mock_emb = self._make_adapter()
        mock_emb.embed_query.return_value = [0.0]
        adapter.embed_query("hello")
        mock_emb.embed_query.assert_called_once_with("hello")

    def test_embed_query_returns_list_of_floats(self):
        """embed_query() result must be a list of floats."""
        adapter, mock_emb = self._make_adapter()
        mock_emb.embed_query.return_value = [0.5, 0.6]
        result = adapter.embed_query("test")
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)
