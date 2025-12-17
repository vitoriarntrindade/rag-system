"""Unit tests for settings module."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from config.settings import Settings, get_settings


class TestSettingsInitialization:
    """Tests for Settings initialization."""
    
    def test_initialization_with_defaults(self):
        """Test that Settings initializes with default values."""
        settings = Settings()
        assert settings is not None
    
    def test_openai_api_key_is_optional(self):
        """Test that openai_api_key can be None."""
        settings = Settings(openai_api_key=None)
        assert settings.openai_api_key is None
    
    def test_openai_api_key_accepts_string(self):
        """Test that openai_api_key accepts string value."""
        settings = Settings(openai_api_key="test-key")
        assert settings.openai_api_key == "test-key"
    
    def test_default_embedding_model(self):
        """Test that default embedding_model is set."""
        settings = Settings()
        assert settings.openai_embedding_model is not None
    
    def test_default_chat_model(self):
        """Test that default chat_model is set."""
        settings = Settings()
        assert settings.openai_chat_model is not None
    
    def test_default_temperature(self):
        """Test that default temperature is set."""
        settings = Settings()
        assert settings.openai_temperature >= 0.0
    
    def test_temperature_is_float(self):
        """Test that temperature is a float."""
        settings = Settings()
        assert isinstance(settings.openai_temperature, float)
    
    def test_default_chunk_size(self):
        """Test that default chunk_size is positive."""
        settings = Settings()
        assert settings.chunk_size > 0
    
    def test_default_chunk_overlap(self):
        """Test that default chunk_overlap is non-negative."""
        settings = Settings()
        assert settings.chunk_overlap >= 0
    
    def test_default_retrieval_top_k(self):
        """Test that default retrieval_top_k is positive."""
        settings = Settings()
        assert settings.retrieval_top_k > 0
    
    def test_default_retrieval_search_type(self):
        """Test that default retrieval_search_type is set."""
        settings = Settings()
        assert settings.retrieval_search_type is not None


class TestSettingsCustomValues:
    """Tests for Settings with custom values."""
    
    def test_custom_embedding_model(self):
        """Test that custom embedding_model is set correctly."""
        settings = Settings(openai_embedding_model="text-embedding-ada-002")
        assert settings.openai_embedding_model == "text-embedding-ada-002"
    
    def test_custom_chat_model(self):
        """Test that custom chat_model is set correctly."""
        settings = Settings(openai_chat_model="gpt-4")
        assert settings.openai_chat_model == "gpt-4"
    
    def test_custom_temperature(self):
        """Test that custom temperature is set correctly."""
        settings = Settings(openai_temperature=0.7)
        assert settings.openai_temperature == 0.7
    
    def test_custom_chunk_size(self):
        """Test that custom chunk_size is set correctly."""
        settings = Settings(chunk_size=500)
        assert settings.chunk_size == 500
    
    def test_custom_chunk_overlap(self):
        """Test that custom chunk_overlap is set correctly."""
        settings = Settings(chunk_overlap=100)
        assert settings.chunk_overlap == 100
    
    def test_custom_retrieval_top_k(self):
        """Test that custom retrieval_top_k is set correctly."""
        settings = Settings(retrieval_top_k=10)
        assert settings.retrieval_top_k == 10
    
    def test_custom_retrieval_search_type(self):
        """Test that custom retrieval_search_type is set correctly."""
        settings = Settings(retrieval_search_type="mmr")
        assert settings.retrieval_search_type == "mmr"


class TestSettingsValidation:
    """Tests for Settings validation rules."""
    
    def test_temperature_minimum_value(self):
        """Test that temperature minimum value is enforced."""
        settings = Settings(openai_temperature=0.0)
        assert settings.openai_temperature == 0.0
    
    def test_temperature_maximum_value(self):
        """Test that temperature maximum value is enforced."""
        settings = Settings(openai_temperature=2.0)
        assert settings.openai_temperature == 2.0
    
    def test_chunk_size_positive(self):
        """Test that chunk_size must be positive."""
        with pytest.raises(Exception):
            Settings(chunk_size=0)
    
    def test_chunk_overlap_non_negative(self):
        """Test that chunk_overlap can be zero."""
        settings = Settings(chunk_overlap=0)
        assert settings.chunk_overlap == 0
    
    def test_retrieval_top_k_positive(self):
        """Test that retrieval_top_k must be positive."""
        with pytest.raises(Exception):
            Settings(retrieval_top_k=0)


class TestSettingsPaths:
    """Tests for Settings path properties."""
    
    def test_base_dir_is_path(self):
        """Test that base_dir is a Path object."""
        settings = Settings()
        assert isinstance(settings.base_dir, Path)
    
    def test_data_dir_is_path(self):
        """Test that data_dir is a Path object."""
        settings = Settings()
        assert isinstance(settings.data_dir, Path)
    
    def test_db_dir_is_path(self):
        """Test that db_dir is a Path object."""
        settings = Settings()
        assert isinstance(settings.db_dir, Path)
    
    def test_logs_dir_is_path(self):
        """Test that logs_dir is a Path object."""
        settings = Settings()
        assert isinstance(settings.logs_dir, Path)
    
    def test_vector_store_path_is_path(self):
        """Test that vector_store_path is a Path object."""
        settings = Settings()
        assert isinstance(settings.vector_store_path, Path)
    
    def test_data_dir_is_subdirectory_of_base(self):
        """Test that data_dir is under base_dir."""
        settings = Settings()
        assert settings.data_dir.parent == settings.base_dir
    
    def test_db_dir_is_subdirectory_of_base(self):
        """Test that db_dir is under base_dir."""
        settings = Settings()
        assert settings.db_dir.parent == settings.base_dir
    
    def test_logs_dir_is_subdirectory_of_base(self):
        """Test that logs_dir is under base_dir."""
        settings = Settings()
        assert settings.logs_dir.parent == settings.base_dir
    
    def test_vector_store_path_is_under_db_dir(self):
        """Test that vector_store_path is under db_dir."""
        settings = Settings()
        assert settings.vector_store_path.parent == settings.db_dir


class TestSettingsLogging:
    """Tests for Settings logging configuration."""
    
    def test_default_log_level(self):
        """Test that default log_level is set."""
        settings = Settings()
        assert settings.log_level is not None
    
    def test_default_log_to_file(self):
        """Test that default log_to_file is set."""
        settings = Settings()
        assert isinstance(settings.log_to_file, bool)
    
    def test_custom_log_level(self):
        """Test that custom log_level is set correctly."""
        settings = Settings(log_level="DEBUG")
        assert settings.log_level == "DEBUG"
    
    def test_custom_log_to_file(self):
        """Test that custom log_to_file is set correctly."""
        settings = Settings(log_to_file=False)
        assert settings.log_to_file is False


class TestGetSettings:
    """Tests for get_settings factory function."""
    
    def test_returns_settings_instance(self):
        """Test that get_settings returns a Settings instance."""
        settings = get_settings()
        assert isinstance(settings, Settings)
    
    def test_accepts_override_parameters(self):
        """Test that get_settings accepts override parameters."""
        settings = get_settings(chunk_size=500, chunk_overlap=100)
        assert settings.chunk_size == 500
        assert settings.chunk_overlap == 100
    
    def test_creates_new_instance_each_call(self):
        """Test that get_settings creates new instance each time."""
        settings1 = get_settings(chunk_size=500)
        settings2 = get_settings(chunk_size=600)
        assert settings1.chunk_size != settings2.chunk_size
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "env-test-key"})
    def test_loads_from_environment(self):
        """Test that get_settings loads values from environment."""
        settings = get_settings()
        assert settings.openai_api_key == "env-test-key"
    
    def test_override_takes_precedence(self):
        """Test that override parameters take precedence."""
        settings = get_settings(
            openai_api_key="override-key",
            chunk_size=750
        )
        assert settings.openai_api_key == "override-key"
        assert settings.chunk_size == 750


class TestSettingsEnvironmentVariables:
    """Tests for Settings loading from environment variables."""
    
    @patch.dict(os.environ, {"OPENAI_EMBEDDING_MODEL": "custom-embedding"})
    def test_loads_embedding_model_from_env(self):
        """Test that embedding_model is loaded from environment."""
        settings = Settings()
        assert settings.openai_embedding_model == "custom-embedding"
    
    @patch.dict(os.environ, {"OPENAI_CHAT_MODEL": "custom-chat"})
    def test_loads_chat_model_from_env(self):
        """Test that chat_model is loaded from environment."""
        settings = Settings()
        assert settings.openai_chat_model == "custom-chat"
    
    @patch.dict(os.environ, {"OPENAI_TEMPERATURE": "0.8"})
    def test_loads_temperature_from_env(self):
        """Test that temperature is loaded from environment."""
        settings = Settings()
        assert settings.openai_temperature == 0.8
    
    @patch.dict(os.environ, {"CHUNK_SIZE": "800"})
    def test_loads_chunk_size_from_env(self):
        """Test that chunk_size is loaded from environment."""
        settings = Settings()
        assert settings.chunk_size == 800
    
    @patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"})
    def test_loads_log_level_from_env(self):
        """Test that log_level is loaded from environment."""
        settings = Settings()
        assert settings.log_level == "DEBUG"
