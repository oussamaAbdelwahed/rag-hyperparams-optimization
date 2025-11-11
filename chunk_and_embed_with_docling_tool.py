"""
Script to chunk knowledge base documents using Docling
and store them in Pinecone vector database.

Docling provides advanced document preprocessing that preserves
document semantics (sections, paragraphs, tables, etc.) for better RAG performance.
"""

import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.readers.docling import DoclingReader
from llama_index.node_parser.docling import DoclingNodeParser
from pinecone import Pinecone, ServerlessSpec
import json

# Load environment variables
load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "docling-tool-chunking-ats-chrono"
DOCUMENT_PATH = "Spec détaillées - Middleware XL EDS (ATS __ XL EDS __ CHRONOPOST).docx"

# Embedding model configuration
# Using bge-base for 768 dimensions (bge-small produces 384 dimensions)
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIMENSION = 768


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


def load_and_preprocess_document():
    """
    Load and preprocess the document using Docling.
    
    Docling provides advanced features:
    - Table extraction with markdown formatting
    - Smart header and footer handling
    - Section and paragraph structure preservation
    - Semantic chunking that respects document structure
    
    Returns:
        list: List of LlamaIndex Document objects
    """
    print(f"Loading document: {DOCUMENT_PATH}")
    
    if not os.path.exists(DOCUMENT_PATH):
        raise FileNotFoundError(f"Document not found: {DOCUMENT_PATH}")
    
    print("Initializing Docling reader with JSON export for rich format...")
    # Use JSON export to preserve rich document structure
    # This maintains semantic information like headings, sections, tables, etc.
    reader = DoclingReader(export_type=DoclingReader.ExportType.JSON)
    
    print("Processing document with Docling...")
    print("Features enabled:")
    print("  ✓ Table extraction with structure preservation")
    print("  ✓ Smart header/footer detection")
    print("  ✓ Section hierarchy preservation")
    print("  ✓ Paragraph boundary detection")
    
    # Load the document
    documents = reader.load_data(DOCUMENT_PATH)
    
    print(f"Successfully loaded {len(documents)} document(s)")
    
    return documents


def clean_metadata_for_pinecone(metadata):
    """
    Clean metadata to ensure it only contains Pinecone-compatible types
    and stays under the 40KB size limit (targeting 40000 bytes max).
    Pinecone only accepts: string, number, boolean, or list of strings.
    
    Args:
        metadata: Dictionary of metadata
        
    Returns:
        Cleaned metadata dictionary
    """
    # Keep only essential fields and ensure size limit
    MAX_METADATA_SIZE = 40000  # Stay under Pinecone's 40960 byte limit
    MAX_FIELD_SIZE = 500  # Max size for any single field value
    
    cleaned = {}
    
    # Priority fields to keep (in order of importance)
    priority_fields = ['file_name', 'file_path', 'page_no', 'headings', 'file_type']
    
    for key, value in metadata.items():
        # Skip complex nested objects entirely
        if isinstance(value, dict):
            # Only keep small, important dicts
            if key in ['origin'] and isinstance(value, dict):
                # Extract only simple fields from nested dicts
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (str, int, float, bool)):
                        field_name = f"{key}_{sub_key}"
                        if isinstance(sub_value, str) and len(sub_value) < MAX_FIELD_SIZE:
                            cleaned[field_name] = sub_value[:MAX_FIELD_SIZE]
                        elif isinstance(sub_value, (int, float, bool)):
                            cleaned[field_name] = sub_value
            continue
            
        elif isinstance(value, list):
            # Handle lists carefully
            if key == 'headings' and all(isinstance(item, str) for item in value):
                # Keep headings but limit size
                headings_str = ' > '.join(value[:5])  # Max 5 levels
                if len(headings_str) < MAX_FIELD_SIZE:
                    cleaned['headings'] = headings_str
            elif all(isinstance(item, (str, int, float, bool)) for item in value):
                # Convert to limited list of strings
                limited_list = [str(item)[:100] for item in value[:3]]  # Max 3 items, 100 chars each
                cleaned[key] = limited_list
            continue
            
        elif isinstance(value, (str, int, float, bool)):
            # Keep simple types but limit string sizes
            if isinstance(value, str):
                if len(value) < MAX_FIELD_SIZE:
                    cleaned[key] = value[:MAX_FIELD_SIZE]
            else:
                cleaned[key] = value
                
        elif value is None:
            continue
    
    # Calculate total size and trim if needed
    metadata_str = json.dumps(cleaned, ensure_ascii=False)
    metadata_size = len(metadata_str.encode('utf-8'))
    
    if metadata_size > MAX_METADATA_SIZE:
        # If still too large, keep only priority fields
        minimal_cleaned = {}
        for key in priority_fields:
            if key in cleaned:
                minimal_cleaned[key] = cleaned[key]
        
        # Check size again
        metadata_str = json.dumps(minimal_cleaned, ensure_ascii=False)
        metadata_size = len(metadata_str.encode('utf-8'))
        
        if metadata_size > MAX_METADATA_SIZE:
            # Last resort: keep only file_name and page_no
            minimal_cleaned = {
                'file_name': cleaned.get('file_name', 'unknown')[:100],
            }
            if 'page_no' in cleaned:
                minimal_cleaned['page_no'] = cleaned['page_no']
        
        return minimal_cleaned
    
    return cleaned


def chunk_and_store_documents(documents, pinecone_index):
    """
    Chunk documents using Docling's semantic chunker and store in Pinecone.
    
    The DoclingNodeParser is specifically designed to:
    - Respect document structure and semantics
    - Maintain section context
    - Preserve table formatting
    - Keep related content together
    
    Args:
        documents: List of LlamaIndex Document objects
        pinecone_index: Pinecone index instance
    
    Returns:
        VectorStoreIndex: The created index
    """
    print("Setting up embedding model...")
    print(f"Using {EMBEDDING_MODEL_NAME} (dimension: {EMBEDDING_DIMENSION})")
    embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL_NAME
    )
    
    print("Setting up Docling semantic chunker...")
    # DoclingNodeParser uses Docling's chunking that preserves document semantics
    node_parser = DoclingNodeParser()
    
    print("Parsing documents into nodes...")
    # Parse documents into nodes first
    nodes = node_parser.get_nodes_from_documents(documents)
    
    print(f"Cleaning metadata for {len(nodes)} nodes...")
    # Clean metadata for each node to ensure Pinecone compatibility
    max_size = 0
    for idx, node in enumerate(nodes):
        if hasattr(node, 'metadata') and node.metadata:
            original_size = len(json.dumps(node.metadata, ensure_ascii=False).encode('utf-8'))
            node.metadata = clean_metadata_for_pinecone(node.metadata)
            cleaned_size = len(json.dumps(node.metadata, ensure_ascii=False).encode('utf-8'))
            max_size = max(max_size, cleaned_size)
            
            if idx < 3:  # Show first 3 for debugging
                print(f"  Node {idx}: metadata size {original_size} -> {cleaned_size} bytes")
    
    print(f"✓ Max metadata size across all nodes: {max_size} bytes (limit: 40960)")
    if max_size > 40000:
        print(f"⚠️  WARNING: Metadata size {max_size} is close to or exceeds limit!")
    
    print("Setting up Pinecone vector store...")
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index
    )
    
    # Create storage context
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )
    
    print("Creating vector index with semantic chunking...")
    print("This process will:")
    print("  1. Generate embeddings for each chunk")
    print("  2. Store in Pinecone vector database")
    
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )
    
    print("Documents successfully chunked and stored in Pinecone!")
    return index


def main():
    """Main execution function."""
    print("=" * 70)
    print("Docling-based Knowledge Base Document Processing Pipeline")
    print("=" * 70)
    print()
    print("This pipeline uses Docling for advanced document preprocessing:")
    print("  • Preserves document structure (sections, paragraphs)")
    print("  • Extracts tables with markdown formatting")
    print("  • Smart header/footer handling")
    print("  • Semantic-aware chunking")
    print("=" * 70)
    print()
    
    try:
        # Step 1: Initialize Pinecone
        pinecone_index = initialize_pinecone()
        print()
        
        # Step 2: Load and preprocess document with Docling
        documents = load_and_preprocess_document()
        print()
        
        # Step 3: Chunk and store in Pinecone
        index = chunk_and_store_documents(documents, pinecone_index)
        
        print("\n" + "=" * 70)
        print("Processing completed successfully!")
        print(f"Index name: {INDEX_NAME}")
        print(f"Document: {DOCUMENT_PATH}")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
