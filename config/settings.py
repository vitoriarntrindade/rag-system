"""Configuration management for RAG system using Pydantic Settings."""

import os
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    This class uses Pydantic Settings to validate and manage configuration
    from environment variables with type safety and default values.
    
    Note: openai_api_key is now optional here and should be injected
    when creating RAGPipeline instances for better security and flexibility.
    """
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(
        default=None, 
        description="OpenAI API key for embeddings and LLM (optional, prefer injection)"
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model"
    )
    openai_chat_model: str = Field(
        default="gpt-3.5-turbo",
        description="OpenAI chat model for generation"
    )
    openai_temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM generation"
    )
    
    # Text Processing Configuration
    chunk_size: int = Field(
        default=1000,
        gt=0,
        description="Size of text chunks for document splitting"
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        description="Overlap between consecutive chunks"
    )
    
    # Retrieval Configuration
    retrieval_top_k: int = Field(
        default=5,
        gt=0,
        description="Number of documents to retrieve"
    )
    retrieval_search_type: str = Field(
        default="similarity",
        description="Type of search to perform (similarity, mmr, etc.)"
    )
    
    # Paths Configuration
    base_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent,
        description="Base directory of the project"
    )
    
    @property
    def data_dir(self) -> Path:
        """Directory for storing documents."""
        return self.base_dir / "data"
    
    @property
    def db_dir(self) -> Path:
        """Directory for vector store database."""
        return self.base_dir / "db"
    
    @property
    def logs_dir(self) -> Path:
        """Directory for log files."""
        return self.base_dir / "logs"
    
    @property
    def vector_store_path(self) -> Path:
        """Path to the vector store database."""
        return self.db_dir / "chroma_db"
    
    # Logging Configuration
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    log_to_file: bool = Field(
        default=True,
        description="Whether to log to file in addition to console"
    )
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def get_settings(**overrides) -> Settings:
    """
    Factory function to create Settings instance with optional overrides.
    
    Args:
        **overrides: Keyword arguments to override default settings
    
    Returns:
        Settings instance
    
    Example:
        settings = get_settings(openai_api_key="sk-...")
    """
    return Settings(**overrides)
