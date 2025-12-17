"""Document loader module for ingesting various document formats."""

from pathlib import Path
from typing import List, Optional, Union

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document

from rag_system.src.utils.logger import setup_logger

logger = setup_logger(__name__)


# Supported file types and their corresponding loaders
SUPPORTED_LOADERS = {
    '.pdf': PyPDFLoader,
    '.txt': TextLoader,
    '.md': UnstructuredMarkdownLoader,
    '.docx': UnstructuredWordDocumentLoader,
    '.doc': UnstructuredWordDocumentLoader,
}


class DocumentLoader:
    """
    Handles document loading operations.
    
    This class is responsible for loading documents from various sources,
    with support for multiple file formats (PDF, TXT, DOCX, MD, etc.).
    """
    
    def __init__(self):
        """Initialize the document loader."""
        logger.info("DocumentLoader initialized")
        logger.info(f"Supported file types: {', '.join(SUPPORTED_LOADERS.keys())}")
    
    @staticmethod
    def get_supported_extensions() -> List[str]:
        """
        Get list of supported file extensions.
        
        Returns:
            List of supported file extensions (e.g., ['.pdf', '.txt', '.docx'])
        """
        return list(SUPPORTED_LOADERS.keys())
    
    def load_file(self, file_path: Path) -> List[Document]:
        """
        Load a document file based on its extension.
        
        Args:
            file_path: Path to the file to load
        
        Returns:
            List of Document objects
        
        Raises:
            FileNotFoundError: If the specified file does not exist
            ValueError: If the file type is not supported
            Exception: If there's an error loading the file
        """
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found at {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        if file_extension not in SUPPORTED_LOADERS:
            logger.error(f"Unsupported file type: {file_extension}")
            raise ValueError(
                f"Unsupported file type: {file_extension}. "
                f"Supported types: {', '.join(SUPPORTED_LOADERS.keys())}"
            )
        
        try:
            logger.info(f"Loading {file_extension} document from: {file_path}")
            loader_class = SUPPORTED_LOADERS[file_extension]
            loader = loader_class(str(file_path))
            documents = loader.load()
            logger.info(f"Successfully loaded {len(documents)} document(s) from {file_path.name}")
            return documents
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            raise
    
    def list_files(
        self,
        directory: Path,
        file_types: Optional[List[str]] = None,
        recursive: bool = True
    ) -> List[Path]:
        """
        List all supported files in a directory.
        
        Args:
            directory: Path to the directory to search
            file_types: List of file extensions to include (e.g., ['.pdf', '.txt']).
                       If None, includes all supported types
            recursive: If True, search subdirectories recursively
        
        Returns:
            List of Path objects for matching files
        
        Raises:
            FileNotFoundError: If the directory does not exist
        """
        if not directory.exists():
            logger.error(f"Directory not found: {directory}")
            raise FileNotFoundError(f"Directory not found at {directory}")
        
        if not directory.is_dir():
            logger.error(f"Path is not a directory: {directory}")
            raise ValueError(f"Path is not a directory: {directory}")
        
        # Determine which file types to search for
        if file_types is None:
            file_types = list(SUPPORTED_LOADERS.keys())
        else:
            # Normalize extensions (ensure they start with '.')
            file_types = [
                ext if ext.startswith('.') else f'.{ext}' 
                for ext in file_types
            ]
            # Validate file types
            unsupported = [ft for ft in file_types if ft not in SUPPORTED_LOADERS]
            if unsupported:
                logger.warning(f"Unsupported file types will be ignored: {unsupported}")
                file_types = [ft for ft in file_types if ft in SUPPORTED_LOADERS]
        
        # Find all matching files
        all_files = []
        for file_type in file_types:
            pattern = f"*{file_type}"
            if recursive:
                files = list(directory.rglob(pattern))
            else:
                files = list(directory.glob(pattern))
            all_files.extend(files)
        
        # Sort for consistent ordering
        all_files.sort()
        
        logger.info(f"Found {len(all_files)} file(s) matching types {file_types}")
        return all_files
    
    def load_directory(
        self,
        directory: Path,
        file_types: Optional[List[str]] = None,
        recursive: bool = True
    ) -> List[Document]:
        """
        Load all documents from a directory.
        
        Args:
            directory: Path to the directory containing documents
            file_types: List of file extensions to include (e.g., ['.pdf', '.txt']).
                       If None, loads all supported types
            recursive: If True, search subdirectories recursively
        
        Returns:
            List of all loaded Document objects from all files found
        
        Raises:
            FileNotFoundError: If the directory does not exist
            ValueError: If no files are found in the directory
        """
        # List all matching files
        files = self.list_files(directory, file_types, recursive)
        
        if not files:
            logger.warning(f"No files found in {directory}")
            raise ValueError(f"No supported files found in directory: {directory}")
        
        logger.info(f"Loading {len(files)} file(s) from {directory}")
        
        # Load all documents
        all_documents = []
        successful_loads = 0
        failed_loads = 0
        
        for file_path in files:
            try:
                logger.info(f"Loading: {file_path.name}")
                documents = self.load_file(file_path)
                all_documents.extend(documents)
                successful_loads += 1
                logger.debug(f"Loaded {len(documents)} document(s) from {file_path.name}")
            except Exception as e:
                failed_loads += 1
                logger.error(f"Failed to load {file_path.name}: {e}")
                # Continue processing other files
                continue
        
        logger.info(
            f"Directory loading complete: {successful_loads} successful, "
            f"{failed_loads} failed, {len(all_documents)} total document(s)"
        )
        
        if not all_documents:
            raise ValueError("No documents could be loaded from any files")
        
        return all_documents
    
    def load_documents(
        self,
        file_path: Optional[Path] = None,
        directory: Optional[Path] = None,
        file_types: Optional[List[str]] = None,
        recursive: bool = True
    ) -> List[Document]:
        """
        Load documents from a file or directory.
        
        Args:
            file_path: Path to a single document file
            directory: Path to a directory containing documents
            file_types: List of file extensions to include when loading from directory
                       (e.g., ['.pdf', '.txt']). If None, loads all supported types
            recursive: If True and loading from directory, search subdirectories
        
        Returns:
            List of loaded Document objects
        
        Raises:
            ValueError: If neither file_path nor directory is provided
        """
        if file_path:
            return self.load_file(file_path)
        elif directory:
            return self.load_directory(directory, file_types=file_types, recursive=recursive)
        else:
            raise ValueError("Either file_path or directory must be provided")
