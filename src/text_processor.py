"""Text processing module for document chunking and splitting."""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import get_settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class TextProcessor:
    """
    Handles text processing operations including document chunking.
    
    This class is responsible for splitting documents into smaller,
    manageable chunks for embedding and retrieval.
    """
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        separators: List[str] = None
    ):
        """
        Initialize the text processor.
        
        Args:
            chunk_size: Size of each text chunk. If None, uses default settings
            chunk_overlap: Overlap between chunks. If None, uses default settings
            separators: List of separators for splitting. If None, uses default
        """
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.separators = separators or [
            "\n\nChapter",  # Split by chapters first
            "\n\n",         # Then by paragraphs
            "\n",           # Then by lines
            " ",            # Finally by spaces
            "",
        ]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            add_start_index=True,
        )
        
        logger.info(
            f"TextProcessor initialized with chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}"
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.
        
        Args:
            documents: List of Document objects to split
        
        Returns:
            List of chunked Document objects
        """
        logger.info(f"Splitting {len(documents)} documents into chunks")
        
        chunks = self.text_splitter.split_documents(documents)
        
        avg_chunk_size = (
            sum(len(chunk.page_content) for chunk in chunks) // len(chunks)
            if chunks else 0
        )
        
        logger.info(
            f"Created {len(chunks)} chunks with average size of "
            f"{avg_chunk_size} characters"
        )
        
        return chunks
