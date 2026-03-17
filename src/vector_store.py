"""Vector store module for embeddings and similarity search."""

from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src.ports.embedding_provider import BaseEmbeddingProvider
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class VectorStore:
    """
    Manages embeddings and vector store operations.
    
    This class handles creating embeddings from documents and storing them
    in a vector database for efficient similarity search.
    """
    
    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider,
        persist_directory: Optional[Path] = None,
    ):
        """
        Initialize the vector store.

        Args:
            embedding_provider: Any object that implements BaseEmbeddingProvider.
                                The vector store does not know (or care) whether
                                it is OpenAI, Ollama, HuggingFace, etc.
            persist_directory:  Directory to persist the Chroma database.
                                If None, must be set before any operation.
        """
        self.embedding_provider = embedding_provider
        self.persist_directory = persist_directory
        self.vectorstore: Optional[Chroma] = None

        if persist_directory is None:
            logger.warning(
                "VectorStore created without persist_directory. "
                "You must set it before calling create_from_documents() or load_existing()."
            )

        logger.info("VectorStore initialized")
    
    def create_from_documents(self, documents: List[Document]) -> Chroma:
        """
        Create a new vector store from documents.
        
        Args:
            documents: List of Document objects to embed and store
        
        Returns:
            Chroma vector store instance
        """
        logger.info(f"Creating vector store from {len(documents)} documents")
        
        self.persist_directory.parent.mkdir(parents=True, exist_ok=True)
        
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_provider,
            persist_directory=str(self.persist_directory)
        )
        
        logger.info(f"Vector store created and persisted to {self.persist_directory}")
        return self.vectorstore
    
    def load_existing(self) -> Chroma:
        """
        Load an existing vector store from disk.
        
        Returns:
            Chroma vector store instance
        
        Raises:
            FileNotFoundError: If the vector store does not exist
        """
        if not self.persist_directory.exists():
            logger.error(f"Vector store not found at {self.persist_directory}")
            raise FileNotFoundError(
                f"Vector store not found at {self.persist_directory}"
            )
        
        logger.info(f"Loading existing vector store from {self.persist_directory}")
        
        self.vectorstore = Chroma(
            persist_directory=str(self.persist_directory),
            embedding_function=self.embedding_provider
        )
        
        logger.info("Vector store loaded successfully")
        return self.vectorstore
    
    def get_or_create(self, documents: Optional[List[Document]] = None) -> Chroma:
        """
        Get existing vector store or create a new one if it doesn't exist.
        
        Args:
            documents: Documents to use if creating a new store.
                      Required if store doesn't exist
        
        Returns:
            Chroma vector store instance
        
        Raises:
            ValueError: If store doesn't exist and no documents provided
        """
        if self.persist_directory.exists():
            return self.load_existing()
        else:
            if documents is None:
                raise ValueError(
                    "Documents must be provided to create a new vector store"
                )
            return self.create_from_documents(documents)
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
    ) -> List[Document]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Query string to search for
            k: Number of results to return. Defaults to 5.

        Returns:
            List of most similar Document objects
        
        Raises:
            RuntimeError: If vector store is not initialized
        """
        if self.vectorstore is None:
            raise RuntimeError("Vector store not initialized. Call load_existing() or create_from_documents() first")

        logger.debug(f"Performing similarity search for query with k={k}")
        
        results = self.vectorstore.similarity_search(query, k=k)
        
        logger.debug(f"Found {len(results)} similar documents")
        return results
