"""
Chunk and index markdown documents using MarkdownNodeParser.

This script reads markdown documents (e.g., from markitdown conversion), chunks them using
MarkdownNodeParser, and optionally indexes them to Pinecone for retrieval.

Usage:
    # Visualization mode (test + enrich + save to file)
    python chunk_and_index.py --test-mode --enrich-metadata --save-to-file
    
    # Production indexing to Pinecone
    python chunk_and_index.py --enrich-metadata
"""

import os
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "markitdown-parse-and-md-node-parser-split"  # to be created if not exists
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIMENSION = 1024  # bge-large-en-v1.5 has 1024 dimensions
MARKDOWN_INPUT_FILE = "chrono-specs-markitdown.md"


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Chunk and index markdown documents using MarkdownNodeParser',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualization mode (test + enrich + save to file)
  python chunk_and_index.py --test-mode --enrich-metadata --save-to-file
  
  # Production mode (index to Pinecone) with metadata enrichment
  python chunk_and_index.py --enrich-metadata
  
  # Test mode without metadata enrichment
  python chunk_and_index.py --test-mode
        """
    )
    
    parser.add_argument(
        '--test-mode',
        action='store_true',
        default=False,
        help='Run in test mode (skip embedding and indexing to Pinecone)'
    )
    
    parser.add_argument(
        '--enrich-metadata',
        action='store_true',
        default=False,
        help='Enable metadata enrichment for chunks'
    )
    
    parser.add_argument(
        '--save-to-file',
        action='store_true',
        default=False,
        help='Save enriched chunks to text file for inspection'
    )
    
    return parser.parse_args()


def initialize_pinecone():
    """Initialize Pinecone client and create index if it doesn't exist."""
    print("Initializing Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists
    existing_indexes = pc.list_indexes()
    index_names = [idx.name for idx in existing_indexes]
    
    if INDEX_NAME not in index_names:
        print(f"Creating new index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"Index '{INDEX_NAME}' created successfully.")
    else:
        print(f"Index '{INDEX_NAME}' already exists.")
    
    return pc.Index(INDEX_NAME)


def load_markdown_documents(input_file):
    """Load a markdown file.
    
    Args:
        input_file: Path to the markdown file
    
    Returns:
        List of LlamaIndex Document objects (single document)
    """
    print(f"Loading markdown document from: {input_file}")
    
    # Try the current directory first, then parent directory
    input_path = Path(input_file)
    if not input_path.exists():
        input_path = Path(__file__).parent / input_file
    if not input_path.exists():
        input_path = Path(__file__).parent.parent / input_file
    
    if not input_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {input_path}")
    
    print(f"  Loading {input_path.name}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # IMPORTANT: Add file metadata so MarkdownNodeParser can extract header hierarchy
    doc = Document(
        text=content
    )
    
    print(f"✓ Loaded {input_path.name} ({len(content)} characters)")
    return [doc]


def chunk_documents(documents):
    """Chunk documents using MarkdownNodeParser.
    
    Args:
        documents: List of Document objects
    
    Returns:
        List of Node objects
    """
    print("Chunking documents with MarkdownNodeParser...")
    
    # Initialize MarkdownNodeParser
    # This parser respects markdown structure (headers, sections, etc.)
    parser = MarkdownNodeParser(
        include_metadata=True,  # Include metadata from headers in chunks
        header_path_separator=" > ",  # Separator for header hierarchy in metadata
    )
    
    # Parse documents into nodes
    nodes = parser.get_nodes_from_documents(documents, show_progress=True)
    
    print(f"✓ Created {len(nodes)} chunks from {len(documents)} document(s)")
    
    # Print some statistics
    if nodes:
        chunk_sizes = [len(node.get_content()) for node in nodes]
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        print(f"  Average chunk size: {avg_size:.0f} characters")
        print(f"  Min chunk size: {min(chunk_sizes)} characters")
        print(f"  Max chunk size: {max(chunk_sizes)} characters")
    
    return nodes


def index_to_pinecone(nodes, pinecone_index):
    """Index nodes to Pinecone vector store.
    
    Args:
        nodes: List of Node objects to index
        pinecone_index: Pinecone index instance
    
    Returns:
        VectorStoreIndex instance
    """
    print("Setting up embedding model and indexing to Pinecone...")
    
    # Initialize embedding model
    embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL,
    )
    
    # Create vector store
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index
    )
    
    # Create storage context
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )
    
    # Create index from nodes
    print(f"Indexing {len(nodes)} chunks to Pinecone...")
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )
    
    print("✓ Documents successfully indexed in Pinecone!")
    return index


def save_chunks_metadata(nodes, output_file="chunks_metadata.json"):
    """Save chunk metadata for analysis and debugging.
    
    Args:
        nodes: List of Node objects
        output_file: Name of output JSON file
    """
    print(f"Saving chunk metadata to {output_file}...")
    
    output_path = Path(__file__).parent / output_file
    
    chunks_data = []
    for i, node in enumerate(nodes):
        chunk_info = {
            "chunk_id": i,
            "node_id": node.node_id,
            "text": node.get_content(),
            "text_length": len(node.get_content()),
            "metadata": node.metadata,
        }
        chunks_data.append(chunk_info)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved metadata for {len(chunks_data)} chunks")


def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_arguments()
    
    print("=" * 60)
    print("LlamaParse Markdown Chunking and Indexing Pipeline")
    print("=" * 60)
    print(f"Mode: {'TEST (no indexing)' if args.test_mode else 'PRODUCTION (with indexing)'}")
    print(f"Metadata enrichment: {'ENABLED' if args.enrich_metadata else 'DISABLED'}")
    print(f"Save to file: {'ENABLED' if args.save_to_file else 'DISABLED'}")
    print("=" * 60)
    print()
    
    try:
        # Step 1: Load markdown documents
        documents = load_markdown_documents(MARKDOWN_INPUT_FILE)
        print()
        
        # Step 2: Chunk documents using MarkdownNodeParser
        nodes = chunk_documents(documents)
        print()
        
        # Step 4: Save chunks metadata for analysis
        save_chunks_metadata(nodes)
        print()
        
        # Step 6: Index to Pinecone (skip in test mode)
        if not args.test_mode:
            pinecone_index = initialize_pinecone()
            print()
            index = index_to_pinecone(nodes, pinecone_index)
            print()
            
            print("=" * 60)
            print("Processing completed successfully!")
            print(f"Total chunks indexed: {len(nodes)}")
            print(f"Index name: {INDEX_NAME}")
            print(f"Embedding model: {EMBEDDING_MODEL}")
            print("=" * 60)
        else:
            print("=" * 60)
            print("TEST MODE - Processing completed successfully!")
            print(f"Total chunks created: {len(nodes)}")
            print(f"Metadata enrichment: {'YES' if args.enrich_metadata else 'NO'}")
            print("Note: Skipped embedding and indexing to Pinecone")
            print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())