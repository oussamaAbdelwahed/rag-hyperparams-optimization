"""
Script to process Markdown document using Docling with hierarchical chunking
and store chunks in Pinecone vector database.

This script:
1. Uses Docling's LlamaIndex integration to parse Markdown files
2. Uses Docling's hierarchical chunking to preserve document structure
3. Embeds chunks using HuggingFace embeddings (dimension 768)
4. Stores embeddings in a dedicated Pinecone index

Usage:
    python process_docx_with_docling.py --doc-path chrono-docs/1.md --index-name chrono-docling-hierarchical
"""

import os
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any

# Docling imports
from docling.document_converter import DocumentConverter
from llama_index.readers.docling import DoclingReader
from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker

# LlamaIndex imports
from llama_index.core import Document
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore

# Pinecone imports
from pinecone import Pinecone, ServerlessSpec

# Rich for pretty console output
from rich.console import Console
from rich.progress import track

# Load environment variables
load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
EMBEDDING_DIMENSION = 768
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"  # HuggingFace model with 768 dimensions (better quality than bge-small)

console = Console()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process Markdown document with hierarchical chunking and store in Pinecone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_docx_with_docling.py --doc-path chrono-docs/1.md --index-name chrono-docling-hierarchical
  python process_docx_with_docling.py --doc-path chrono-docs/1.md --index-name my-index --embedding-model BAAI/bge-base-en-v1.5
        """
    )
    
    parser.add_argument(
        "--doc-path",
        required=True,
        help="Path to the Markdown document to process (e.g., chrono-docs/1.md)"
    )
    
    parser.add_argument(
        "--index-name",
        required=True,
        help="Name for the Pinecone index (e.g., chrono-docling-hierarchical)"
    )
    
    parser.add_argument(
        "--embedding-model",
        default=EMBEDDING_MODEL,
        help=f"HuggingFace embedding model to use (default: {EMBEDDING_MODEL})"
    )
    
    parser.add_argument(
        "--export-metadata",
        action="store_true",
        help="Export hierarchical metadata to JSON file for analysis"
    )
    
    return parser.parse_args()


def initialize_pinecone(index_name: str):
    """Initialize Pinecone client and create index if it doesn't exist."""
    console.print("[bold blue]Initializing Pinecone...[/bold blue]")
    
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not found in environment variables")
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists
    existing_indexes = pc.list_indexes()
    index_names = [idx.name for idx in existing_indexes]
    
    if index_name not in index_names:
        console.print(f"[yellow]Creating new index: {index_name}[/yellow]")
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        console.print(f"[green]Index '{index_name}' created successfully.[/green]")
    else:
        console.print(f"[green]Index '{index_name}' already exists.[/green]")
    
    return pc.Index(index_name)


def parse_and_chunk_with_docling(doc_path: str) -> List[Document]:
    """
    Parse Markdown document using Docling's LlamaIndex integration with hierarchical chunking.
    
    Args:
        doc_path: Path to the Markdown document
        
    Returns:
        List of LlamaIndex Document objects with hierarchical chunks
    """
    console.print(f"[bold blue]Parsing document with DoclingReader: {doc_path}[/bold blue]")
    
    # Verify file exists
    doc_file = Path(doc_path)
    if not doc_file.exists():
        raise FileNotFoundError(f"Document not found: {doc_path}")
    
    # Initialize DoclingReader with hierarchical chunker
    console.print("[yellow]Initializing DoclingReader with HierarchicalChunker...[/yellow]")
    reader = DoclingReader(
        chunker=HierarchicalChunker()
    )
    
    # Load documents - this will parse and chunk in one step
    console.print(f"[yellow]Loading and chunking document...[/yellow]")
    documents = reader.load_data(file_path=doc_file)
    
    console.print(f"[green]Generated {len(documents)} hierarchical chunks[/green]\n")
    
    # Preview first few chunks
    console.print("[dim]First 3 chunks preview:[/dim]")
    for i, doc in enumerate(documents[:3]):
        console.print(f"[dim]--- Chunk {i} (length: {len(doc.text)} chars) ---[/dim]")
        console.print(f"[dim]{doc.text[:200]}...[/dim]")
        console.print(f"[dim]Metadata: {doc.metadata}[/dim]\n")
    
    # Enhance metadata for Pinecone (add source file and ensure no null values)
    for idx, doc in enumerate(documents):
        # Add source file
        if "source_file" not in doc.metadata:
            doc.metadata["source_file"] = str(doc_path)
        
        # Add chunk index info
        doc.metadata["chunk_index"] = idx
        doc.metadata["total_chunks"] = len(documents)
        doc.metadata["chunk_type"] = "hierarchical"
        
        # Add sequential relationships
        if idx > 0:
            doc.metadata["prev_id"] = idx - 1
        if idx < len(documents) - 1:
            doc.metadata["next_id"] = idx + 1
        
        # Ensure doc_id is set
        if not hasattr(doc, 'doc_id') or not doc.doc_id:
            doc.doc_id = f"chunk_{idx}"
    
    console.print(f"[green]Enhanced metadata for {len(documents)} documents[/green]")
    console.print(f"[dim]Metadata includes: source_file, chunk_index, hierarchical structure info[/dim]\n")
    
    return documents


def create_llamaindex_documents(chunks, doc_path: str) -> List[Document]:
    """
    This function is now integrated into chunk_markdown_hierarchically.
    Kept for backward compatibility but not used.
    """
    return chunks


def export_hierarchical_metadata(documents: List[Document], output_path: str):
    """
    Export hierarchical metadata to JSON for analysis and future retrieval use.
    
    This creates a file that can be used by hierarchical retrievers to:
    - Navigate parent/child relationships
    - Understand document structure
    - Build tree-based retrieval strategies
    
    Args:
        documents: List of LlamaIndex Documents with hierarchical metadata
        output_path: Path to save the JSON file
    """
    console.print(f"[bold blue]Exporting hierarchical metadata to {output_path}...[/bold blue]")
    
    metadata_export = {
        "total_chunks": len(documents),
        "chunks": []
    }
    
    for doc in documents:
        chunk_info = {
            "doc_id": doc.doc_id if hasattr(doc, 'doc_id') else None,
            "text_preview": doc.text[:200] if len(doc.text) > 200 else doc.text,
            "text_length": len(doc.text),
            "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
        }
        metadata_export["chunks"].append(chunk_info)
    
    # Save to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_export, f, indent=2, ensure_ascii=False)
    
    console.print(f"[green]Metadata exported successfully to {output_path}[/green]\n")


def embed_and_store_in_pinecone(documents: List[Document], pinecone_index, embedding_model: str):
    """
    Embed documents and store them in Pinecone.
    
    Args:
        documents: List of LlamaIndex Document objects
        pinecone_index: Pinecone index instance
        embedding_model: HuggingFace model name for embeddings
    """
    console.print("[bold blue]Initializing embeddings and storing in Pinecone...[/bold blue]")
    
    # Initialize HuggingFace embeddings
    console.print(f"[yellow]Loading embedding model: {embedding_model}[/yellow]")
    embed_model = HuggingFaceEmbedding(model_name=embedding_model)
    
    # Create Pinecone vector store
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    
    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create index and embed documents
    console.print("[yellow]Embedding and storing documents (this may take a while)...[/yellow]")
    
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )
    
    console.print(f"[green]Successfully stored {len(documents)} chunks in Pinecone![/green]")
    
    return index


def main():
    """Main execution function."""
    args = parse_arguments()
    
    console.print("[bold green]Starting Docling + Hierarchical Chunking Pipeline[/bold green]\n")
    
    try:
        # Step 1: Initialize Pinecone
        pinecone_index = initialize_pinecone(args.index_name)
        
        # Step 2: Parse and chunk document with Docling (using LlamaIndex integration)
        documents = parse_and_chunk_with_docling(args.doc_path)
        
        # Step 3: Export metadata if requested
        if args.export_metadata:
            metadata_file = f"{args.index_name}_hierarchical_metadata.json"
            export_hierarchical_metadata(documents, metadata_file)
        
        # Step 4: Embed and store in Pinecone
        embed_and_store_in_pinecone(documents, pinecone_index, args.embedding_model)
        
        console.print("\n[bold green]✓ Pipeline completed successfully![/bold green]")
        console.print(f"[bold green]✓ Index: {args.index_name}[/bold green]")
        console.print(f"[bold green]✓ Total chunks: {len(documents)}[/bold green]")
        
    except Exception as e:
        console.print(f"\n[bold red]Error: {str(e)}[/bold red]")
        raise


if __name__ == "__main__":
    main()
