"""Port (interface) for embedding providers."""

from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingProvider(ABC):
    """
    Contract that any embedding provider must fulfill.

    The core (vector_store, pipeline) depends only on this interface,
    never on a concrete implementation such as OpenAIEmbeddings or OllamaEmbeddings.
    """

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (one per input text).
        """
        ...

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Generate an embedding for a single query string.

        Used at retrieval time to embed the user's question before
        searching the vector store.

        Args:
            text: The query string to embed.

        Returns:
            A single embedding vector.
        """
        ...
