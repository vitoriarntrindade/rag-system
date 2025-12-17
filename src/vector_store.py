"""Vector store module for embeddings and similarity search."""

from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from config.settings import get_settings
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
        persist_directory: Optional[Path] = None,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist the vector store.
                             If None, uses default settings
            embedding_model: Name of the OpenAI embedding model.
                           If None, uses default settings
        """
        settings = get_settings()
        self.persist_directory = persist_directory or settings.vector_store_path
        self.embedding_model_name = embedding_model or settings.openai_embedding_model
        
        logger.info(f"Initializing VectorStore with model: {self.embedding_model_name}")
        
        self.embeddings = OpenAIEmbeddings(model=self.embedding_model_name)
        self.vectorstore: Optional[Chroma] = None
    
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
            embedding=self.embeddings,
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
            embedding_function=self.embeddings
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
        k: int = None
    ) -> List[Document]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Query string to search for
            k: Number of results to return. If None, uses default settings
        
        Returns:
            List of most similar Document objects
        
        Raises:
            RuntimeError: If vector store is not initialized
        """
        if self.vectorstore is None:
            raise RuntimeError("Vector store not initialized. Call load_existing() or create_from_documents() first")
        
        if k is None:
            settings = get_settings()
            k = settings.retrieval_top_k
        
        logger.debug(f"Performing similarity search for query with k={k}")
        
        results = self.vectorstore.similarity_search(query, k=k)
        
        logger.debug(f"Found {len(results)} similar documents")
        return results
