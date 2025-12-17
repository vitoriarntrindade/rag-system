"""Main entry point for the RAG System CLI."""

import argparse
import os
import sys
from pathlib import Path

from rag_system.src.rag_pipeline import RAGPipeline
from rag_system.src.utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    """Main CLI entry point for the RAG system."""
    # Load API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable is not set.")
        print("\nPlease set it before running:")
        print("  export OPENAI_API_KEY='your-api-key'")
        print("or create a .env file with:")
        print("  OPENAI_API_KEY=your-api-key")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(
        description="RAG (Retrieval-Augmented Generation) System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest a single document (auto-detects format)
  python main.py ingest --file data/document.pdf
  python main.py ingest --file data/notes.txt
  
  # List available files in a directory
  python main.py ingest --directory data/ --list-files
  
  # Ingest all supported files from a directory (recursive by default)
  python main.py ingest --directory data/
  
  # Ingest only specific file types
  python main.py ingest --directory data/ --file-types pdf txt
  python main.py ingest --directory data/ --file-types docx md
  
  # Ingest files from a directory (non-recursive)
  python main.py ingest --directory data/ --no-recursive
  
  # Ask a single question
  python main.py query "What is the main topic?"
  
  # Start interactive chat
  python main.py chat
  
  # Force recreate the vector store
  python main.py ingest --file data/document.pdf --force
  
Supported file types: PDF, TXT, DOCX, DOC, MD
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Ingest command
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest documents into the system"
    )
    ingest_parser.add_argument(
        "--file",
        type=Path,
        help="Path to a single document file (PDF, TXT, DOCX, MD, etc.)"
    )
    ingest_parser.add_argument(
        "--directory",
        type=Path,
        help="Path to a directory containing documents"
    )
    ingest_parser.add_argument(
        "--file-types",
        nargs="+",
        help="File types to include when loading from directory (e.g., pdf txt docx). "
             "If not specified, loads all supported types"
    )
    ingest_parser.add_argument(
        "--list-files",
        action="store_true",
        help="List available files in directory without processing"
    )
    ingest_parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Search subdirectories recursively when loading from directory (default: True)"
    )
    ingest_parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Do not search subdirectories when loading from directory"
    )
    ingest_parser.add_argument(
        "--force",
        action="store_true",
        help="Force recreate vector store even if it exists"
    )
    
    # Query command
    query_parser = subparsers.add_parser(
        "query",
        help="Ask a single question to the system"
    )
    query_parser.add_argument(
        "question",
        type=str,
        help="The question to ask"
    )
    query_parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Don't display source documents"
    )
    
    # Chat command
    chat_parser = subparsers.add_parser(
        "chat",
        help="Start an interactive chat session"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        # Initialize pipeline with API key
        pipeline = RAGPipeline(openai_api_key=api_key)
        
        if args.command == "ingest":
            if not args.file and not args.directory:
                logger.error("Either --file or --directory must be provided")
                print("Error: Either --file or --directory must be provided")
                sys.exit(1)
            
            # Handle --list-files option
            if args.list_files:
                if not args.directory:
                    print("Error: --list-files requires --directory")
                    sys.exit(1)
                
                from src.document_loader import DocumentLoader
                doc_loader = DocumentLoader()
                
                # Parse file types if provided
                file_types = None
                if args.file_types:
                    file_types = [f'.{ft}' if not ft.startswith('.') else ft 
                                  for ft in args.file_types]
                
                try:
                    files = doc_loader.list_files(
                        args.directory,
                        file_types=file_types,
                        recursive=args.recursive
                    )
                    
                    print(f"\n{'='*60}")
                    print(f"Files found in {args.directory}")
                    print(f"{'='*60}")
                    
                    if files:
                        # Group files by extension
                        from collections import defaultdict
                        files_by_type = defaultdict(list)
                        for f in files:
                            files_by_type[f.suffix].append(f)
                        
                        total_files = 0
                        for ext, file_list in sorted(files_by_type.items()):
                            print(f"\n{ext.upper()} files ({len(file_list)}):")
                            for f in sorted(file_list):
                                print(f"  - {f.relative_to(args.directory)}")
                            total_files += len(file_list)
                        
                        print(f"\n{'='*60}")
                        print(f"Total: {total_files} file(s)")
                        print(f"{'='*60}\n")
                    else:
                        print("No files found matching the criteria.\n")
                
                except Exception as e:
                    print(f"Error listing files: {e}")
                    sys.exit(1)
                
                sys.exit(0)
            
            # Parse file types if provided
            file_types = None
            if args.file_types:
                file_types = [f'.{ft}' if not ft.startswith('.') else ft 
                              for ft in args.file_types]
                logger.info(f"File types filter: {file_types}")
            
            logger.info(f"Starting ingestion process")
            pipeline.ingest_documents(
                file_path=args.file,
                directory=args.directory,
                file_types=file_types,
                force_recreate=args.force,
                recursive=args.recursive
            )
            print("✅ Document ingestion completed successfully!")
        
        elif args.command == "query":
            # Load existing index
            logger.info("Loading existing vector store for query")
            pipeline.load_existing_index()
            
            # Process query
            answer, sources = pipeline.query(
                args.question,
                return_sources=not args.no_sources
            )
            
            # Display results
            print(f"\n{'='*60}")
            print(f"QUESTION: {args.question}")
            print(f"\n{'='*60}")
            print(f"ANSWER:\n{answer}")
            
            if sources and not args.no_sources:
                print(f"\n{'-'*60}")
                print(f"SOURCES ({len(sources)} documents):")
                for i, doc in enumerate(sources[:3], 1):
                    print(f"\nSource {i}:")
                    preview = doc.page_content[:200].replace("\n", " ")
                    print(f"  {preview}...")
                    if "page" in doc.metadata:
                        print(f"  Page: {doc.metadata['page']}")
            
            print(f"{'='*60}\n")
        
        elif args.command == "chat":
            # Load existing index
            logger.info("Loading existing vector store for chat")
            pipeline.load_existing_index()
            
            # Start interactive chat
            pipeline.interactive_chat()
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"\n❌ Error: {e}")
        print("\nMake sure you've ingested documents first:")
        print("  python main.py ingest --file data/your_document.pdf")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
