"""Main RAG Pipeline orchestrator."""

import os
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from rag_system.config.settings import Settings, get_settings
from rag_system.src.document_loader import DocumentLoader
from rag_system.src.generator import ResponseGenerator
from rag_system.src.retriever import DocumentRetriever
from rag_system.src.text_processor import TextProcessor
from rag_system.src.utils.logger import setup_logger
from rag_system.src.vector_store import VectorStore

logger = setup_logger(__name__)


class RAGPipeline:
    """
    Main RAG (Retrieval-Augmented Generation) pipeline orchestrator.
    
    This class integrates all components of the RAG system:
    - Document loading
    - Text processing and chunking
    - Vector storage and embeddings
    - Document retrieval
    - Response generation
    
    Security Note:
        The OpenAI API key must be provided explicitly when creating instances
        of this class to ensure proper key management and avoid accidental exposure.
    """
    
    def __init__(
        self,
        openai_api_key: str,
        vector_store_path: Optional[Path] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        retrieval_top_k: Optional[int] = None,
        settings: Optional[Settings] = None
    ):
        """
        Initialize the RAG pipeline with all components.
        
        Args:
            openai_api_key: OpenAI API key (required). This ensures explicit
                           key management and prevents accidental exposure.
            vector_store_path: Path to vector store. If None, uses settings
            chunk_size: Size of text chunks. If None, uses settings
            chunk_overlap: Overlap between chunks. If None, uses settings
            retrieval_top_k: Number of docs to retrieve. If None, uses settings
            settings: Custom Settings instance. If None, creates default settings
        
        Raises:
            ValueError: If openai_api_key is empty or None
        
        Example:
            >>> import os
            >>> api_key = os.getenv("OPENAI_API_KEY")
            >>> pipeline = RAGPipeline(openai_api_key=api_key)
        """
        if not openai_api_key:
            error_msg = (
                "OpenAI API key is required. Please provide it explicitly:\n"
                "  pipeline = RAGPipeline(openai_api_key='your-key')\n"
                "or from environment:\n"
                "  pipeline = RAGPipeline(openai_api_key=os.getenv('OPENAI_API_KEY'))"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("Initializing RAG Pipeline")
        
        # Set OpenAI API key in environment for langchain components
        os.environ['OPENAI_API_KEY'] = openai_api_key
        
        # Load or create settings
        self.settings = settings or get_settings()
        
        # Initialize components
        self.document_loader = DocumentLoader()
        self.text_processor = TextProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.vector_store = VectorStore(persist_directory=vector_store_path)
        self.retriever: Optional[DocumentRetriever] = None
        self.generator = ResponseGenerator()
        
        self._is_initialized = False
        
        logger.info("RAG Pipeline initialized successfully")
    
    def ingest_documents(
        self,
        file_path: Optional[Path] = None,
        directory: Optional[Path] = None,
        file_types: Optional[List[str]] = None,
        force_recreate: bool = False,
        recursive: bool = True
    ) -> None:
        """
        Ingest documents into the RAG system.
        
        This method:
        1. Loads documents from the specified source
        2. Splits them into chunks
        3. Creates embeddings and stores in vector database
        
        Args:
            file_path: Path to a single document file
            directory: Path to a directory of documents
            file_types: List of file extensions to include (e.g., ['pdf', 'txt', 'docx']).
                       If None, loads all supported types
            force_recreate: If True, recreates vector store even if it exists
            recursive: If True, search subdirectories when loading from directory
        
        Raises:
            ValueError: If neither file_path nor directory is provided
        """
        logger.info("Starting document ingestion")
        
        # Check if vector store already exists
        if not force_recreate and self.vector_store.persist_directory.exists():
            logger.info("Vector store already exists. Loading existing store.")
            self.vector_store.load_existing()
            self.retriever = DocumentRetriever(self.vector_store)
            self._is_initialized = True
            return
        
        # Load documents
        documents = self.document_loader.load_documents(
            file_path=file_path,
            directory=directory,
            file_types=file_types,
            recursive=recursive
        )
        
        # Split into chunks
        chunks = self.text_processor.split_documents(documents)
        
        # Create vector store
        self.vector_store.create_from_documents(chunks)
        
        # Initialize retriever
        self.retriever = DocumentRetriever(self.vector_store)
        
        self._is_initialized = True
        logger.info("Document ingestion completed successfully")
    
    def load_existing_index(self) -> None:
        """
        Load an existing vector store index.
        
        Raises:
            FileNotFoundError: If vector store doesn't exist
        """
        logger.info("Loading existing vector store")
        
        self.vector_store.load_existing()
        self.retriever = DocumentRetriever(self.vector_store)
        self._is_initialized = True
        
        logger.info("Existing vector store loaded successfully")
    
    def query(
        self,
        question: str,
        return_sources: bool = True
    ) -> Tuple[str, Optional[List[Document]]]:
        """
        Query the RAG system with a question.
        
        Args:
            question: The question to ask
            return_sources: Whether to return source documents
        
        Returns:
            Tuple of (answer string, list of source documents if requested)
        
        Raises:
            RuntimeError: If pipeline is not initialized
        """
        if not self._is_initialized:
            raise RuntimeError(
                "Pipeline not initialized. Call ingest_documents() or "
                "load_existing_index() first"
            )
        
        logger.info(f"Processing query: {question[:100]}...")
        
        # Retrieve relevant documents
        relevant_docs = self.retriever.retrieve(question)
        
        # Generate response
        answer, sources = self.generator.generate(question, relevant_docs)
        
        logger.info("Query processed successfully")
        
        if return_sources:
            return answer, sources
        else:
            return answer, None
    
    def interactive_chat(self) -> None:
        """
        Start an interactive chat session.
        
        Allows users to ask multiple questions in a conversational manner.
        Type 'quit', 'exit', or 'stop' to end the session.
        """
        if not self._is_initialized:
            raise RuntimeError(
                "Pipeline not initialized. Call ingest_documents() or "
                "load_existing_index() first"
            )
        
        logger.info("Starting interactive chat session")
        print("\n" + "="*60)
        print("RAG System - Interactive Chat")
        print("="*60)
        print("Ask me anything! Type 'quit', 'exit', or 'stop' to end.\n")
        
        while True:
            try:
                user_input = input("Your question: ").strip()
                
                if user_input.lower() in ["quit", "exit", "stop"]:
                    print("\nThank you for using the RAG system! Goodbye.")
                    logger.info("Interactive chat session ended")
                    break
                
                if not user_input:
                    print("Please enter a question.\n")
                    continue
                
                # Process query
                answer, sources = self.query(user_input)
                
                # Display answer
                print(f"\n{'='*60}")
                print(f"ANSWER:\n{answer}")
                
                # Display sources
                if sources:
                    print(f"\n{'-'*60}")
                    print(f"SOURCES ({len(sources)} documents):")
                    for i, doc in enumerate(sources[:3], 1):
                        print(f"\nSource {i}:")
                        preview = doc.page_content[:200].replace("\n", " ")
                        print(f"  {preview}...")
                        if "page" in doc.metadata:
                            print(f"  Page: {doc.metadata['page']}")
                
                print(f"{'='*60}\n")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                logger.info("Interactive chat session interrupted")
                break
            except Exception as e:
                logger.error(f"Error during chat: {e}")
                print(f"\nError: {e}\n")
