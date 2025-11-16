"""
Chunk and index markdown documents using Docling HybridChunker.

This script reads markdown documents (e.g., from markitdown conversion), chunks them using
Docling's HybridChunker with tokenization-aware refinements, and optionally indexes 
them to Pinecone for retrieval.

Key Features:
- HybridChunker applies tokenization-aware refinements on top of hierarchical chunking
- Uses a tokenizer aligned with the embedding model (CRITICAL for RAG)
- Splits oversized chunks and merges undersized peer chunks intelligently
- Preserves document structure while respecting token boundaries

Differences from MarkdownNodeParser (chunk_and_index.py):
┌─────────────────────────┬──────────────────────────┬──────────────────────────┐
│ Feature                 │ MarkdownNodeParser       │ HybridChunker            │
├─────────────────────────┼──────────────────────────┼──────────────────────────┤
│ Chunking Strategy       │ Markdown structure only  │ Hybrid: structure +      │
│                         │ (headers, sections)      │ tokenization-aware       │
├─────────────────────────┼──────────────────────────┼──────────────────────────┤
│ Token Awareness         │ No                       │ Yes (uses tokenizer)     │
├─────────────────────────┼──────────────────────────┼──────────────────────────┤
│ Chunk Size Control      │ Character-based only     │ Token-based (more        │
│                         │                          │ accurate for LLMs)       │
├─────────────────────────┼──────────────────────────┼──────────────────────────┤
│ Oversized Chunks        │ May exceed token limits  │ Auto-splits when needed  │
├─────────────────────────┼──────────────────────────┼──────────────────────────┤
│ Undersized Chunks       │ Not handled              │ Auto-merges peers        │
├─────────────────────────┼──────────────────────────┼──────────────────────────┤
│ Embedding Alignment     │ No tokenizer sync        │ Same tokenizer as        │
│                         │                          │ embedding model          │
├─────────────────────────┼──────────────────────────┼──────────────────────────┤
│ Best For                │ Simple markdown docs     │ RAG systems, precise     │
│                         │                          │ token control            │
└─────────────────────────┴──────────────────────────┴──────────────────────────┘

Why HybridChunker for RAG?
- Ensures chunks fit within embedding model's context window
- Prevents token truncation during embedding
- Improves retrieval quality by aligning chunking with model tokenization
- Smarter chunk size optimization (merges small, splits large)

Important:
- The chunker and embedding model MUST use the same tokenizer for optimal retrieval
- Requires: pip install 'docling-core[chunking]' for HuggingFace tokenizers
- Or: pip install 'docling-core[chunking-openai]' for OpenAI tokenizers (tiktoken)

Usage:
    # Visualization mode (test + enrich + save to file)
    python chunk_and_index_with_docling_hybrid.py --test-mode --enrich-metadata --save-to-file
    
    # Production indexing to Pinecone
    python chunk_and_index_with_docling_hybrid.py --enrich-metadata

References:
- Docling Chunking: https://docling-project.github.io/docling/concepts/chunking/
- Hybrid Chunking Example: https://docling-project.github.io/docling/examples/hybrid_chunking/
"""

import os
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
from typing import List

# Try to import HybridChunker and tokenizer from docling
try:
    # Primary import path (recommended)
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
    from transformers import AutoTokenizer
    CHUNKER_SOURCE = "docling_core"
    HAS_HYBRID_CHUNKER = True
except ImportError as e:
    # If HybridChunker doesn't exist, we'll use HierarchicalChunker as fallback
    print(f"⚠️  Could not import HybridChunker: {e}")
    print("Install with: pip install 'docling-core[chunking]'")
    HybridChunker = None
    HuggingFaceTokenizer = None
    AutoTokenizer = None
    CHUNKER_SOURCE = "fallback"
    HAS_HYBRID_CHUNKER = False

# LlamaIndex imports
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore

# Docling imports
from docling.document_converter import DocumentConverter
try:
    from llama_index.node_parser.docling import DoclingNodeParser
    HAS_DOCLING_NODE_PARSER = True
except ImportError:
    from llama_index.readers.docling import DoclingReader
    from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker as FallbackChunker
    HAS_DOCLING_NODE_PARSER = False

# Pinecone imports
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "markitdown-docling-hybrid-chunker"  # to be created if not exists
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIMENSION = 1024  # bge-large-en-v1.5 has 1024 dimensions
MARKDOWN_INPUT_FILE = "chrono-specs-markitdown.md"

# HybridChunker configuration
MAX_TOKENS = 512  # Maximum tokens per chunk (should align with embedding model context window)
MERGE_PEERS = True  # Merge undersized peer chunks when possible


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Chunk and index markdown documents using Docling HybridChunker',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualization mode (test + enrich + save to file)
  python chunk_and_index_with_docling_hybrid.py --test-mode --enrich-metadata --save-to-file
  
  # Production mode (index to Pinecone) with metadata enrichment
  python chunk_and_index_with_docling_hybrid.py --enrich-metadata
  
  # Test mode without metadata enrichment
  python chunk_and_index_with_docling_hybrid.py --test-mode
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


def load_markdown_file(input_file):
    """Load a markdown file and return its content.
    
    Args:
        input_file: Path to the markdown file
    
    Returns:
        Tuple of (file_path, content)
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
    
    print(f"✓ Loaded {input_path.name} ({len(content)} characters)")
    return input_path, content


def chunk_with_docling_hybrid(file_path: Path) -> List[Document]:
    """Chunk markdown document using Docling's HybridChunker with proper tokenization.
    
    The HybridChunker uses a tokenizer aligned with the embedding model to ensure
    chunks respect token boundaries and fit within the model's context window.
    
    Args:
        file_path: Path to the markdown file
    
    Returns:
        List of Document objects (chunks)
    """
    print("\nChunking document with Docling HybridChunker...")
    print(f"Chunker source: {CHUNKER_SOURCE}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Max tokens per chunk: {MAX_TOKENS}")
    
    if HAS_HYBRID_CHUNKER and HybridChunker is not None:
        # Method 1: Use HybridChunker with proper tokenization (RECOMMENDED)
        print("Using DocumentConverter + HybridChunker with HuggingFace tokenizer...")
        
        from docling.document_converter import DocumentConverter
        
        # Step 1: Convert the document first
        print(f"Converting document with DocumentConverter...")
        converter = DocumentConverter()
        conv_result = converter.convert(file_path)
        dl_doc = conv_result.document
        
        print(f"Document converted: {len(dl_doc.texts)} text elements")
        
        # Step 2: Initialize tokenizer aligned with embedding model
        # This is CRITICAL for RAG: the chunker and embedding model must use the same tokenizer
        print(f"Loading tokenizer for embedding model: {EMBEDDING_MODEL}...")
        hf_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        
        tokenizer = HuggingFaceTokenizer(
            tokenizer=hf_tokenizer,
            max_tokens=MAX_TOKENS,  # Maximum tokens per chunk
        )
        
        # Step 3: Initialize HybridChunker with the tokenizer
        chunker = HybridChunker(
            tokenizer=tokenizer,
            merge_peers=MERGE_PEERS,  # Merge undersized peer chunks when possible
        )
        
        # Step 4: Chunk the DoclingDocument
        print("Chunking with HybridChunker...")
        chunk_iter = chunker.chunk(dl_doc=dl_doc)
        chunks = list(chunk_iter)
        
        print(f"Created {len(chunks)} chunks from HybridChunker")
        
        # Step 5: Convert chunks to LlamaIndex Document objects
        from llama_index.core import Document
        nodes = []
        for i, chunk in enumerate(chunks):
            # Use contextualize() to get the enriched text with metadata
            enriched_text = chunker.contextualize(chunk=chunk)
            doc = Document(
                text=enriched_text,
                metadata={}
            )
            nodes.append(doc)
        
        print(f"Converted to {len(nodes)} LlamaIndex Document objects")
        
    elif HAS_DOCLING_NODE_PARSER:
        # Method 2: Use DoclingNodeParser with HybridChunker (if available)
        print("Using DoclingNodeParser with HybridChunker...")
        
        # Initialize tokenizer
        hf_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        tokenizer = HuggingFaceTokenizer(
            tokenizer=hf_tokenizer,
            max_tokens=MAX_TOKENS,
        )
        
        # Initialize the chunker
        chunker = HybridChunker(
            tokenizer=tokenizer,
            merge_peers=MERGE_PEERS,
        )
        
        # Use DoclingNodeParser
        node_parser = DoclingNodeParser(chunker=chunker)
        
        # First, convert the document using DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(file_path)
        
        # Get the parsed document
        doc_content = result.document.export_to_markdown()
        
        # Create a LlamaIndex Document
        doc = Document(text=doc_content, metadata={"source_file": str(file_path)})
        
        # Parse into nodes
        nodes = node_parser.get_nodes_from_documents([doc], show_progress=True)
        
    else:
        # Method 3: Fallback to HierarchicalChunker if HybridChunker not available
        print("⚠️  HybridChunker not found!")
        print("Falling back to HierarchicalChunker...")
        print("Note: Install HybridChunker support with: pip install 'docling-core[chunking]'")
        
        from llama_index.readers.docling import DoclingReader
        from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker as FallbackChunker
        
        # Use HierarchicalChunker as fallback
        chunker = FallbackChunker()
        reader = DoclingReader(chunker=chunker)
        
        # Load and chunk in one step
        nodes = reader.load_data(file_path=file_path)
    
    print(f"✓ Created {len(nodes)} chunks from document")
    
    # Clear ALL metadata completely to avoid Pinecone's 40KB limit
    # IMPORTANT: Pinecone has a 40KB metadata limit per vector (40,960 bytes)
    # DoclingReader includes large metadata fields (headings, structure, etc.) that can exceed 1.6MB
    # 
    # Note: We clear metadata here, but we'll create completely new TextNode objects
    # before indexing to ensure no hidden metadata is carried over
    
    print("\nRemoving ALL metadata to comply with Pinecone's 40KB limit...")
    for idx, node in enumerate(nodes):
        # Set metadata to empty dict - NO metadata at all
        node.metadata = {}
    
    # Print some statistics
    if nodes:
        chunk_sizes = [len(node.get_content()) for node in nodes]
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        print(f"  Average chunk size: {avg_size:.0f} characters")
        print(f"  Min chunk size: {min(chunk_sizes)} characters")
        print(f"  Max chunk size: {max(chunk_sizes)} characters")
        
        # Check metadata sizes (Pinecone limit: 40KB per vector)
        metadata_sizes = []
        for node in nodes:
            metadata_json = json.dumps(node.metadata, ensure_ascii=False)
            metadata_size = len(metadata_json.encode('utf-8'))
            metadata_sizes.append(metadata_size)
        
        avg_metadata_size = sum(metadata_sizes) / len(metadata_sizes)
        max_metadata_size = max(metadata_sizes)
        print(f"  Average metadata size: {avg_metadata_size:.0f} bytes")
        print(f"  Max metadata size: {max_metadata_size} bytes")
        
        # Warning if metadata is too large
        PINECONE_METADATA_LIMIT = 40960  # 40KB
        if max_metadata_size > PINECONE_METADATA_LIMIT:
            print(f"  ⚠️  WARNING: Max metadata size ({max_metadata_size} bytes) exceeds Pinecone limit ({PINECONE_METADATA_LIMIT} bytes)!")
            print(f"  ⚠️  Indexing will fail. Reduce metadata size.")
        elif max_metadata_size > PINECONE_METADATA_LIMIT * 0.8:
            print(f"  ⚠️  WARNING: Max metadata size is close to Pinecone limit. Consider reducing.")
    
    return nodes


def index_to_pinecone(nodes, pinecone_index):
    """Index nodes to Pinecone vector store using LlamaIndex.
    
    Args:
        nodes: List of Node objects to index
        pinecone_index: Pinecone index instance
    
    Returns:
        VectorStoreIndex instance
    """
    print("\nSetting up embedding model and indexing to Pinecone...")
    
    # CRITICAL: LlamaIndex automatically stores text in a way that works with Pinecone
    # We just need to ensure our nodes have minimal metadata to stay under 40KB limit
    
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
    print(f"\nSaving chunk metadata to {output_file}...")
    
    output_path = Path(__file__).parent / output_file
    
    chunks_data = []
    for i, node in enumerate(nodes):
        chunk_info = {
            "chunk_id": i,
            "node_id": node.node_id,
            "text": node.get_content(),
            "text_length": len(node.get_content()),
            "metadata": node.metadata if hasattr(node, 'metadata') else {},
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
    print("Docling HybridChunker - Markdown Chunking and Indexing Pipeline")
    print("=" * 60)
    print(f"Mode: {'TEST (no indexing)' if args.test_mode else 'PRODUCTION (with indexing)'}")
    print(f"Metadata enrichment: {'ENABLED' if args.enrich_metadata else 'DISABLED'}")
    print(f"Save to file: {'ENABLED' if args.save_to_file else 'DISABLED'}")
    print("=" * 60)
    print()
    
    try:
        # Step 1: Load markdown file
        file_path, content = load_markdown_file(MARKDOWN_INPUT_FILE)
        print()
        
        # Step 2: Chunk documents using Docling HybridChunker
        nodes = chunk_with_docling_hybrid(file_path)
        print()
        
        # Step 3: Save chunks metadata for analysis
        save_chunks_metadata(nodes)
        print()
        
        # Step 4: Index to Pinecone (skip in test mode)
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
            print(f"Chunker type: {'HybridChunker' if HAS_HYBRID_CHUNKER else 'HierarchicalChunker (fallback)'}")
            if HAS_HYBRID_CHUNKER:
                print(f"Max tokens per chunk: {MAX_TOKENS}")
                print(f"Merge peers: {MERGE_PEERS}")
            print("=" * 60)
        else:
            print("=" * 60)
            print("TEST MODE - Processing completed successfully!")
            print(f"Total chunks created: {len(nodes)}")
            print(f"Metadata enrichment: {'YES' if args.enrich_metadata else 'NO'}")
            print(f"Chunker type: {'HybridChunker' if HAS_HYBRID_CHUNKER else 'HierarchicalChunker (fallback)'}")
            if HAS_HYBRID_CHUNKER:
                print(f"Max tokens per chunk: {MAX_TOKENS}")
                print(f"Merge peers: {MERGE_PEERS}")
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
