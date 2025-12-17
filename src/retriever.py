"""Retriever module for document retrieval operations."""

from typing import List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag_system.config.settings import get_settings
from rag_system.src.utils.logger import setup_logger
from rag_system.src.vector_store import VectorStore

logger = setup_logger(__name__)


class DocumentRetriever:
    """
    Handles document retrieval operations.
    
    This class provides an interface for retrieving relevant documents
    based on a query using the vector store.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        search_type: str = None,
        top_k: int = None
    ):
        """
        Initialize the document retriever.
        
        Args:
            vector_store: VectorStore instance to use for retrieval
            search_type: Type of search (similarity, mmr, etc.).
                        If None, uses default settings
            top_k: Number of documents to retrieve. If None, uses default settings
        """
        settings = get_settings()
        self.vector_store = vector_store
        self.search_type = search_type or settings.retrieval_search_type
        self.top_k = top_k or settings.retrieval_top_k
        
        logger.info(
            f"DocumentRetriever initialized with search_type={self.search_type}, "
            f"top_k={self.top_k}"
        )
    
    def get_retriever(self) -> BaseRetriever:
        """
        Get a LangChain retriever instance.
        
        Returns:
            BaseRetriever configured with current settings
        """
        if self.vector_store.vectorstore is None:
            raise RuntimeError("Vector store not initialized")
        
        retriever = self.vector_store.vectorstore.as_retriever(
            search_type=self.search_type,
            search_kwargs={"k": self.top_k}
        )
        
        logger.debug("Created retriever instance")
        return retriever
    
    def retrieve(self, query: str, k: int = None) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query string to search for
            k: Number of documents to retrieve. If None, uses instance value
        
        Returns:
            List of relevant Document objects
        """
        k = k or self.top_k
        
        logger.info(f"Retrieving documents for query (k={k})")
        logger.debug(f"Query: {query[:100]}...")
        
        retriever = self.get_retriever()
        documents = retriever.invoke(query)
        
        logger.info(f"Retrieved {len(documents)} documents")
        return documents
