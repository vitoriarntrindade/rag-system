"""OpenAI adapter for the embedding provider port."""

from typing import List

from langchain_openai import OpenAIEmbeddings

from src.ports.embedding_provider import BaseEmbeddingProvider
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class OpenAIEmbeddingAdapter(BaseEmbeddingProvider):
    """
    Concrete implementation of BaseEmbeddingProvider using OpenAI's
    embedding models.

    This adapter is the *only* place in the codebase that imports
    langchain_openai for embedding purposes. ChromaDB receives this
    object through the BaseEmbeddingProvider interface, so neither
    vector_store.py nor the pipeline need to know it is OpenAI.
    """

    def __init__(self, model: str, api_key: str) -> None:
        """
        Initialize the OpenAI embedding adapter.

        Args:
            model:   OpenAI embedding model name (e.g. "text-embedding-3-small").
            api_key: OpenAI API key. Passed explicitly to keep credentials
                     out of global environment state.
        """
        self._embeddings = OpenAIEmbeddings(
            model=model,
            api_key=api_key,
        )
        logger.info(f"OpenAIEmbeddingAdapter ready (model={model})")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of documents.

        Called during ingestion to build the vector store index.

        Args:
            texts: List of document strings to embed.

        Returns:
            List of embedding vectors, one per input text.
        """
        logger.debug(f"Embedding {len(texts)} document(s) via OpenAI")
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Generate an embedding for a single query string.

        Called at retrieval time before searching the vector store.

        Args:
            text: The user's query string.

        Returns:
            A single embedding vector.
        """
        logger.debug("Embedding query via OpenAI")
        return self._embeddings.embed_query(text)

    def __repr__(self) -> str:  # pragma: no cover
        return f"OpenAIEmbeddingAdapter(model={self._embeddings.model!r})"
