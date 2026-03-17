"""Unit tests for provider_factory module."""

from unittest.mock import MagicMock, patch

import pytest

from config.settings import Settings
from src.factories.provider_factory import create_embedding_provider, create_llm_provider
from src.ports.embedding_provider import BaseEmbeddingProvider
from src.ports.llm_provider import BaseLLMProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _settings(**kwargs) -> Settings:
    """Return a Settings instance with safe defaults for tests."""
    return Settings(
        llm_provider=kwargs.get("llm_provider", "openai"),
        embedding_provider=kwargs.get("embedding_provider", "openai"),
        openai_chat_model=kwargs.get("openai_chat_model", "gpt-3.5-turbo"),
        openai_embedding_model=kwargs.get("openai_embedding_model", "text-embedding-3-small"),
        openai_temperature=kwargs.get("openai_temperature", 0.3),
    )


# ---------------------------------------------------------------------------
# create_llm_provider
# ---------------------------------------------------------------------------

class TestCreateLLMProvider:
    """Tests for create_llm_provider factory function."""

    def test_returns_base_llm_provider_instance(self):
        """create_llm_provider must return a BaseLLMProvider subclass."""
        with patch("src.adapters.llm.openai_llm.ChatOpenAI"):
            provider = create_llm_provider(_settings(), api_key="test-key")
        assert isinstance(provider, BaseLLMProvider)

    def test_openai_provider_uses_correct_model(self):
        """Adapter must be created with the model from settings."""
        with patch("src.adapters.llm.openai_llm.ChatOpenAI") as mock_chat:
            create_llm_provider(
                _settings(openai_chat_model="gpt-4"),
                api_key="test-key",
            )
        _, kwargs = mock_chat.call_args
        assert kwargs["model"] == "gpt-4"

    def test_openai_provider_uses_correct_temperature(self):
        """Adapter must be created with the temperature from settings."""
        with patch("src.adapters.llm.openai_llm.ChatOpenAI") as mock_chat:
            create_llm_provider(
                _settings(openai_temperature=0.7),
                api_key="test-key",
            )
        _, kwargs = mock_chat.call_args
        assert kwargs["temperature"] == 0.7

    def test_openai_provider_uses_correct_api_key(self):
        """Adapter must be created with the api_key passed explicitly."""
        with patch("src.adapters.llm.openai_llm.ChatOpenAI") as mock_chat:
            create_llm_provider(_settings(), api_key="sk-secret")
        _, kwargs = mock_chat.call_args
        assert kwargs["api_key"] == "sk-secret"

    def test_unsupported_llm_provider_raises_value_error(self):
        """Unknown provider names must raise ValueError with a clear message."""
        settings = _settings(llm_provider="unknown-provider")
        with pytest.raises(ValueError, match="unknown-provider"):
            create_llm_provider(settings, api_key="test-key")

    def test_provider_name_is_case_insensitive(self):
        """Provider name comparison must be case-insensitive."""
        with patch("src.adapters.llm.openai_llm.ChatOpenAI"):
            provider = create_llm_provider(
                _settings(llm_provider="OpenAI"),
                api_key="test-key",
            )
        assert isinstance(provider, BaseLLMProvider)


# ---------------------------------------------------------------------------
# create_embedding_provider
# ---------------------------------------------------------------------------

class TestCreateEmbeddingProvider:
    """Tests for create_embedding_provider factory function."""

    def test_returns_base_embedding_provider_instance(self):
        """create_embedding_provider must return a BaseEmbeddingProvider subclass."""
        with patch("src.adapters.embeddings.openai_embeddings.OpenAIEmbeddings"):
            provider = create_embedding_provider(_settings(), api_key="test-key")
        assert isinstance(provider, BaseEmbeddingProvider)

    def test_openai_provider_uses_correct_model(self):
        """Adapter must be created with the embedding model from settings."""
        with patch("src.adapters.embeddings.openai_embeddings.OpenAIEmbeddings") as mock_emb:
            create_embedding_provider(
                _settings(openai_embedding_model="text-embedding-ada-002"),
                api_key="test-key",
            )
        _, kwargs = mock_emb.call_args
        assert kwargs["model"] == "text-embedding-ada-002"

    def test_openai_provider_uses_correct_api_key(self):
        """Adapter must be created with the api_key passed explicitly."""
        with patch("src.adapters.embeddings.openai_embeddings.OpenAIEmbeddings") as mock_emb:
            create_embedding_provider(_settings(), api_key="sk-embed-secret")
        _, kwargs = mock_emb.call_args
        assert kwargs["api_key"] == "sk-embed-secret"

    def test_unsupported_embedding_provider_raises_value_error(self):
        """Unknown provider names must raise ValueError with a clear message."""
        settings = _settings(embedding_provider="unknown-provider")
        with pytest.raises(ValueError, match="unknown-provider"):
            create_embedding_provider(settings, api_key="test-key")

    def test_provider_name_is_case_insensitive(self):
        """Provider name comparison must be case-insensitive."""
        with patch("src.adapters.embeddings.openai_embeddings.OpenAIEmbeddings"):
            provider = create_embedding_provider(
                _settings(embedding_provider="OpenAI"),
                api_key="test-key",
            )
        assert isinstance(provider, BaseEmbeddingProvider)
