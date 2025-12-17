"""RAG System - A modular Retrieval-Augmented Generation system."""

__version__ = "0.1.0"

# Main exports available at package level
__all__ = [
    "RAGPipeline",
    "DocumentLoader",
    "ResponseGenerator",
    "DocumentRetriever",
    "TextProcessor",
    "VectorStore",
]


def __getattr__(name):
    """Lazy import to avoid loading heavy dependencies unless needed."""
    if name == "RAGPipeline":
        from src.rag_pipeline import RAGPipeline
        return RAGPipeline
    elif name == "DocumentLoader":
        from src.document_loader import DocumentLoader
        return DocumentLoader
    elif name == "ResponseGenerator":
        from src.generator import ResponseGenerator
        return ResponseGenerator
    elif name == "DocumentRetriever":
        from src.retriever import DocumentRetriever
        return DocumentRetriever
    elif name == "TextProcessor":
        from src.text_processor import TextProcessor
        return TextProcessor
    elif name == "VectorStore":
        from src.vector_store import VectorStore
        return VectorStore
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")