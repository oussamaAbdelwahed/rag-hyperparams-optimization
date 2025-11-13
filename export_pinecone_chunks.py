"""
Script to export all chunks from Pinecone index to a JSON file.
Loads all vectors and their metadata from the index.

Usage:
    python export_pinecone_chunks.py
"""

import os
import json
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from pinecone import Pinecone
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings


# Configuration
PINECONE_INDEX_NAME = "enhanced-preprocess-tool-chunking-ats-chrono"
OUTPUT_FILE = "pinecone_chunks_export.json"


def _initialize_pinecone():
    """Initialize Pinecone client."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in environment variables")
    
    pc = Pinecone(
        api_key=api_key,
        pool_threads=4,
        timeout=30,
    )
    return pc


def _check_index_populated(pc, index_name):
    """Check if a Pinecone index exists and has vectors."""
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        return False, 0
    
    pinecone_index = pc.Index(index_name)
    stats = pinecone_index.describe_index_stats()
    total_vector_count = stats.get('total_vector_count', 0)
    
    return total_vector_count > 0, total_vector_count


def main():
    """Main function to export all chunks from Pinecone."""
    start_time = time.time()
    
    print("="*70)
    print("Pinecone Chunk Export Tool")
    print("="*70)
    
    print(f"\n🔧 Configuration:")
    print(f"   Index: {PINECONE_INDEX_NAME}")
    print(f"   Output File: {OUTPUT_FILE}")
    
    # Initialize Pinecone
    print("\n🔧 Initializing Pinecone...")
    pc = _initialize_pinecone()
    
    # Check if index exists and get stats
    is_populated, vector_count = _check_index_populated(pc, PINECONE_INDEX_NAME)
    
    if not is_populated:
        print(f"❌ Error: Index '{PINECONE_INDEX_NAME}' not found or is empty")
        return
    
    print(f"✓ Index found with {vector_count} vectors")
    
    # Load index
    print(f"\n📂 Loading Pinecone index...")
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    
    # Setup embedding model (needed to load the index)
    print("🔧 Setting up embedding model...")
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-base-en-v1.5",
        device="cpu",
        trust_remote_code=True,
    )
    Settings.embed_model = embed_model
    print("✓ Embedding model configured")
    
    # Create vector store and load index
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context
    )
    print("✓ Index loaded successfully")
    
    # Query Pinecone directly to get all vectors
    print(f"\n📋 Extracting chunks from Pinecone...")
    
    # Alternative approach: Query Pinecone directly using list and fetch
    chunks_data = []
    
    # List all vector IDs in the index
    print("   Fetching vector IDs...")
    
    # Pinecone list_paginated to get all IDs
    all_vector_ids = []
    pagination_token = None
    
    while True:
        if pagination_token:
            list_response = pinecone_index.list_paginated(pagination_token=pagination_token)
        else:
            list_response = pinecone_index.list_paginated()
        
        # Extract IDs from the response
        if hasattr(list_response, 'vectors'):
            vector_ids = [v.id for v in list_response.vectors]
            all_vector_ids.extend(vector_ids)
        
        # Check if there are more pages
        if hasattr(list_response, 'pagination') and list_response.pagination:
            pagination_token = list_response.pagination.next
            if not pagination_token:
                break
        else:
            break
    
    print(f"✓ Found {len(all_vector_ids)} vector IDs")
    
    # Fetch vectors in batches (reduce batch size to avoid URI too large error)
    batch_size = 100  # Smaller batch size to avoid 414 error
    total_batches = (len(all_vector_ids) + batch_size - 1) // batch_size
    
    print(f"   Fetching vectors in {total_batches} batches...")
    
    for batch_idx in range(0, len(all_vector_ids), batch_size):
        batch_ids = all_vector_ids[batch_idx:batch_idx + batch_size]
        current_batch = batch_idx // batch_size + 1
        
        print(f"   Processing batch {current_batch}/{total_batches} ({len(batch_ids)} vectors)...", end='\r')
        
        # Fetch vectors
        fetch_response = pinecone_index.fetch(ids=batch_ids)
        
        # Access vectors from the response (Pinecone v3+ uses .vectors attribute)
        vectors_dict = fetch_response.vectors
        
        for vector_id, vector_data in vectors_dict.items():
            # Extract metadata (Pinecone v3+ uses .metadata attribute)
            metadata = vector_data.metadata if vector_data.metadata else {}
            
            # Extract text from metadata (LlamaIndex stores it there)
            text = metadata.get('text', '') if 'text' in metadata else metadata.get('_node_content', '')
            
            # Clean metadata (remove text fields that we already extracted)
            clean_metadata = {k: v for k, v in metadata.items() if k not in ['text', '_node_content']}
            
            chunk_info = {
                "chunk_id": vector_id,
                "chunk_index": len(chunks_data),
                "text": text,
                "metadata": clean_metadata,
            }
            
            chunks_data.append(chunk_info)
    
    print(f"\n✓ Extracted {len(chunks_data)} chunks")
    
    # Prepare output data
    output_data = {
        "metadata": {
            "pinecone_index": PINECONE_INDEX_NAME,
            "total_chunks": len(chunks_data),
            "embedding_model": "BAAI/bge-base-en-v1.5",
            "embedding_dimension": 768,
            "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "chunks": chunks_data,
    }
    
    # Save to JSON
    print(f"\n💾 Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Export saved to {OUTPUT_FILE}")
    
    # End timing
    end_time = time.time()
    duration = end_time - start_time
    
    # Print summary
    print("\n" + "="*70)
    print("📊 Export Summary")
    print("="*70)
    print(f"Index Name: {PINECONE_INDEX_NAME}")
    print(f"Total Chunks Exported: {len(chunks_data)}")
    print(f"Output File: {OUTPUT_FILE}")
    print(f"⏱️  Execution Time: {duration:.2f} seconds")
    print("="*70)
    print("✓ Export complete!")


if __name__ == "__main__":
    main()
