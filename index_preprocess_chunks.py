"""
Script to index pre-downloaded Preprocess API chunks to Pinecone.

This script reads JSON files containing chunks from the Preprocess API,
embeds them using the standard embedding model, and stores them in Pinecone.

Usage:
    python index_preprocess_chunks.py
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "enhanced-preprocess-tool-chunking-ats-chrono"
EMBEDDING_DIMENSION = 768
CHUNKS_DIRECTORY = "chrono-preprocess-chunks"

# Mapping of JSON files to document names
CHUNK_FILES = {
    "part1.json": "document_1",
    "part2.json": "document_2",
    "part3.json": "document_3",
    "part4.json": "document_4",
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


def load_chunks_from_json(json_file_path, document_name):
    """Load chunks from a JSON file.
    
    Args:
        json_file_path: Path to the JSON file containing chunks
        document_name: Name to identify the source document
    
    Returns:
        List of LlamaIndex Document objects
    """
    print(f"Loading chunks from: {json_file_path}")
    
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")
    
    # Read the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract chunks from the JSON structure
    # The chunks are in data.chunks as an array of strings
    if 'data' not in data or 'chunks' not in data['data']:
        raise ValueError(f"Invalid JSON structure in {json_file_path}. Expected 'data.chunks' field.")
    
    chunks = data['data']['chunks']
    
    if not isinstance(chunks, list):
        raise ValueError(f"Chunks must be a list in {json_file_path}")
    
    print(f"Found {len(chunks)} chunks in {json_file_path}")
    
    # Convert chunks to LlamaIndex Document objects
    documents = []
    for i, chunk_text in enumerate(chunks):
        if not isinstance(chunk_text, str):
            print(f"Warning: Chunk {i} is not a string, skipping...")
            continue
        
        if not chunk_text.strip():
            print(f"Warning: Chunk {i} is empty, skipping...")
            continue
        
        metadata = {
            "chunk_id": i,
            "document_name": document_name,
            "source_file": os.path.basename(json_file_path),
        }
        
        doc = Document(
            text=chunk_text,
            metadata=metadata
        )
        documents.append(doc)
    
    print(f"Created {len(documents)} valid documents from {json_file_path}")
    return documents


def index_documents(documents, pinecone_index):
    """Embed documents and store them in Pinecone.
    
    Args:
        documents: List of Document objects to index
        pinecone_index: Pinecone index instance
    
    Returns:
        VectorStoreIndex instance
    """
    print("Setting up embedding model...")
    
    # Initialize embedding model (same as in the original script)
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-base-en-v1.5",  # 768 dimensions - matches Pinecone index
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
    print(f"Creating vector index and storing {len(documents)} chunks in Pinecone...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )
    
    print("Documents successfully indexed in Pinecone!")
    return index


def main():
    """Main execution function."""
    print("=" * 60)
    print("Preprocess Chunks Indexing Pipeline")
    print("=" * 60)
    
    try:
        # Step 1: Initialize Pinecone
        pinecone_index = initialize_pinecone()
        
        # Step 2: Load all chunks from JSON files
        all_documents = []
        
        for json_file, doc_name in CHUNK_FILES.items():
            json_path = os.path.join(CHUNKS_DIRECTORY, json_file)
            
            if not os.path.exists(json_path):
                print(f"Warning: {json_path} not found, skipping...")
                continue
            
            documents = load_chunks_from_json(json_path, doc_name)
            all_documents.extend(documents)
        
        if not all_documents:
            print("No documents to index. Exiting.")
            return
        
        print(f"\n{'=' * 60}")
        print(f"Total documents to index: {len(all_documents)}")
        print(f"{'=' * 60}\n")
        
        # Step 3: Index all documents in Pinecone
        index = index_documents(all_documents, pinecone_index)
        
        print("\n" + "=" * 60)
        print("Indexing completed successfully!")
        print(f"Total chunks indexed: {len(all_documents)}")
        print(f"Index name: {INDEX_NAME}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
