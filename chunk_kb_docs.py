"""
Script to chunk knowledge base documents using Preprocess API
and store them in Pinecone vector database.
"""

import os
import time
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from pypreprocess import Preprocess

# Load environment variables
load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PREPROCESS_API_KEY = os.getenv("PREPROCESS_API_KEY")
INDEX_NAME = "enhanced-preprocess-tool-chunking-ats-chrono"
DOCUMENT_PATH = "Spec détaillées - Middleware XL EDS (ATS __ XL EDS __ CHRONOPOST).docx"

# Preprocessing parameters
PREPROCESS_OPTIONS = {
    "table_output_format": "markdown",
    "repeat_table_header": True,
    "merge": True,
    "keep_header": False,
    "smart_header": True,
    "keep_footer": False,
    "image_text": False,
}


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
            dimension=384,  # Dimension for BAAI/bge-base-en-v1.5 embeddings
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


def poll_for_chunks(preprocess, max_wait_minutes=120, check_interval=30):
    """
    Poll the Preprocess API for chunks, up to max_wait_minutes.
    
    Args:
        preprocess: Preprocess instance
        max_wait_minutes: Maximum minutes to wait (default: 120)
        check_interval: Seconds between checks (default: 30)
    
    Returns:
        List of chunks or None if timeout
    """
    print(f"Polling for chunks (up to {max_wait_minutes} minutes)...")
    print(f"Checking every {check_interval} seconds...")
    
    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60
    check_count = 0
    
    while True:
        elapsed = time.time() - start_time
        
        if elapsed > max_wait_seconds:
            print(f"Timeout: Document is still processing after {max_wait_minutes} minutes.")
            print("Try running the script again later to check if processing is complete.")
            return None
        
        check_count += 1
        elapsed_minutes = int(elapsed / 60)
        elapsed_seconds = int(elapsed % 60)
        
        # Try to get the result
        try:
            result = preprocess.result()
            chunks = getattr(result, "chunks", None)
            
            if chunks:
                print(f"✓ Chunks are ready after {elapsed_minutes}m {elapsed_seconds}s!")
                return chunks
            else:
                print(f"  Check #{check_count} - Not ready yet ({elapsed_minutes}m {elapsed_seconds}s). Retrying in {check_interval}s...")
        except Exception as e:
            print(f"  Check #{check_count} - Still processing ({elapsed_minutes}m {elapsed_seconds}s): {str(e)[:50]}")
        
        time.sleep(check_interval)


def load_and_preprocess_document():
    """Load and preprocess the document using Preprocess API."""
    print(f"Loading document: {DOCUMENT_PATH}")
    
    if not PREPROCESS_API_KEY:
        raise ValueError("PREPROCESS_API_KEY not found in environment variables")
    
    # Initialize Preprocess client
    preprocess = Preprocess(api_key=PREPROCESS_API_KEY)
    
    # Set the filepath
    preprocess.set_filepath(DOCUMENT_PATH)
    
    # Set preprocessing options
    preprocess.set_options(PREPROCESS_OPTIONS)
    
    # Process the document (upload and chunk)
    print("Uploading and starting chunking job...")
    preprocess.chunk()
    
    # Poll for chunks with robust retry logic
    chunks = poll_for_chunks(preprocess, max_wait_minutes=120, check_interval=30)
    
    if not chunks:
        print("No chunks available yet. Please retry later.")
        return []
    
    print(f"Successfully loaded {len(chunks)} chunks")
    
    # Convert chunks to LlamaIndex Document objects
    documents = []
    for i, chunk in enumerate(chunks):
        # Handle both dict and object attribute access
        chunk_text = getattr(chunk, "content", "")
        page = getattr(chunk, "page", None)
        chunk_type = getattr(chunk, "type", "text")
        
        metadata = {
            "chunk_id": i,
            "page": page,
            "type": chunk_type,
        }
        
        doc = Document(
            text=chunk_text,
            metadata=metadata
        )
        documents.append(doc)
    
    return documents


def chunk_and_store_documents(documents, pinecone_index):
    """Chunk documents and store them in Pinecone."""
    print("Setting up embedding model...")
    
    # Initialize embedding model
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-base-en-v1.5"
    )
    
    # Create vector store
    print("Setting up Pinecone vector store...")
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index
    )
    
    # Create storage context
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )
    
    # Create index from documents
    print("Creating vector index and storing chunks in Pinecone...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )
    
    print("Documents successfully chunked and stored in Pinecone!")
    return index


def main():
    """Main execution function."""
    print("=" * 60)
    print("Knowledge Base Document Chunking Pipeline")
    print("=" * 60)
    
    try:
        # Step 1: Initialize Pinecone
        pinecone_index = initialize_pinecone()
        
        # Step 2: Load and preprocess document
        documents = load_and_preprocess_document()
        
        # Step 3: Chunk and store in Pinecone
        index = chunk_and_store_documents(documents, pinecone_index)
        
        print("\n" + "=" * 60)
        print("Processing completed successfully!")
        print(f"Index name: {INDEX_NAME}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
