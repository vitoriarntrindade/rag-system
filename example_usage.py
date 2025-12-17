"""
Example: How to use RAG System as a library with explicit API key injection

This example demonstrates the secure way to use the RAG system
by explicitly passing the API key instead of relying on global state.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import RAGPipeline
from rag_system.src.rag_pipeline import RAGPipeline


def main():
    """Example of using RAG system with proper API key management."""
    
    # Method 1: Load API key from environment (Recommended for production)
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Please set it in your environment or .env file"
        )
    
    # Initialize pipeline with explicit API key injection
    print("Initializing RAG Pipeline...")
    pipeline = RAGPipeline(openai_api_key=api_key)
    
    # Example 1: Ingest a single document
    print("\n=== Example 1: Ingest Single Document ===")
    pipeline.ingest_documents(
        file_path=Path("data/understanding_climate_change.pdf")
    )
    
    # Example 2: Query the system
    print("\n=== Example 2: Query System ===")
    answer, sources = pipeline.query("What is climate change?")
    print(f"Answer: {answer}")
    print(f"Number of sources: {len(sources)}")
    
    # Example 3: Ingest directory with specific file types
    print("\n=== Example 3: Ingest Multiple Files ===")
    pipeline.ingest_documents(
        directory=Path("data/"),
        file_types=['.pdf', '.txt'],
        recursive=True,
        force_recreate=True
    )
    
    # Example 4: Interactive chat
    print("\n=== Example 4: Interactive Chat ===")
    print("Starting interactive chat (type 'quit' to exit)...")
    pipeline.interactive_chat()


if __name__ == "__main__":
    main()
