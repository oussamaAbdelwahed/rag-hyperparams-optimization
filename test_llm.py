"""
Interactive RAG Query Tool - Query populated Pinecone indexes with OpenAI LLM
"""

import os
import sys
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.vector_stores.pinecone import PineconeVectorStore  # type: ignore
from pinecone import Pinecone  # type: ignore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore
from llama_index.llms.openai import OpenAI  # type: ignore
from llama_index.core.postprocessor import SimilarityPostprocessor


def _get_pinecone_index_name(chunk_size):
    """Get Pinecone index name based on chunk size."""
    base_name = os.getenv("PINECONE_INDEX_BASE_NAME", "rag")
    return f"{base_name}-chunk-{chunk_size}"


def _initialize_pinecone():
    """Initialize Pinecone client with timeout configuration."""
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
        return False
    
    pinecone_index = pc.Index(index_name)
    stats = pinecone_index.describe_index_stats()
    total_vector_count = stats.get('total_vector_count', 0)
    
    return total_vector_count > 0


def _load_index(chunk_size):
    """Load existing Pinecone index."""
    pc = _initialize_pinecone()
    index_name = _get_pinecone_index_name(chunk_size)
    
    if not _check_index_populated(pc, index_name):
        raise ValueError(f"Index '{index_name}' not found or is empty")
    
    print(f"✓ Loading index: {index_name}")
    pinecone_index = pc.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context
    )
    return index


def _setup_llm():
    """Configure OpenAI LLM."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    # Use GPT-4 (best available) or GPT-4o if available
    llm = OpenAI(
        model="gpt-4o",
        api_key=api_key,
        temperature=0.7,
    )
    print("✓ OpenAI LLM configured (gpt-4o)")
    return llm


def _setup_embedding():
    """Configure HuggingFace embedding model."""
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-base-en-v1.5",
        device="cpu",
        trust_remote_code=True,
    )
    print("✓ HuggingFace embedding model configured")
    return embed_model


def main():
    """Main interactive RAG query loop."""
    print("="*70)
    print("Interactive RAG Query Tool")
    print("Using OpenAI GPT-4o + Pinecone Vector DB")
    print("="*70)
    
    # Setup
    print("\n🔧 Initializing components...")
    
    # Configure embedding model
    embed_model = _setup_embedding()
    Settings.embed_model = embed_model
    
    # Configure LLM
    llm = _setup_llm()
    Settings.llm = llm
    
    # Check available indexes
    print("\n📋 Available Pinecone indexes:")
    pc = _initialize_pinecone()
    chunk_sizes = [256, 512]
    available_indexes = {}
    
    for chunk_size in chunk_sizes:
        index_name = _get_pinecone_index_name(chunk_size)
        is_populated = _check_index_populated(pc, index_name)
        if is_populated:
            available_indexes[chunk_size] = index_name
            status = "✓ Available"
        else:
            status = "✗ Not populated"
        print(f"   Chunk size {chunk_size}: {index_name} - {status}")
    
    if not available_indexes:
        print("❌ No populated indexes found!")
        return
    
    # Select index
    print("\n📍 Select chunk size:")
    for i, chunk_size in enumerate(sorted(available_indexes.keys()), 1):
        print(f"   {i}. Chunk size {chunk_size}")
    
    choice = input("\nEnter choice (default: 1): ").strip() or "1"
    try:
        choice_idx = int(choice) - 1
        selected_chunk_size = sorted(available_indexes.keys())[choice_idx]
    except (ValueError, IndexError):
        print("Invalid choice, using default (256)")
        selected_chunk_size = 256
    
    # Load index
    print(f"\n📂 Loading index with chunk size {selected_chunk_size}...")
    try:
        index = _load_index(selected_chunk_size)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return
    
    # Configure query engine with similarity filtering
    print("\n⚙️  Configuring query engine...")
    similarity_cutoff = 0.7  # High semantic relevance threshold
    similarity_processor = SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)
    
    query_engine = index.as_query_engine(
        similarity_top_k=5,  # Retrieve top 5 chunks
        node_postprocessors=[similarity_processor],
    )
    print(f"✓ Query engine ready (top_k=5, similarity_cutoff={similarity_cutoff})")
    
    # Interactive query loop
    print("\n" + "="*70)
    print("Ready for queries! (Type 'exit' or 'quit' to stop)")
    print("="*70 + "\n")
    
    while True:
        try:
            query = input("🔍 Enter your question: ").strip()
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not query:
                print("⚠️  Please enter a question\n")
                continue
            
            print("\n⏳ Processing...\n")
            
            # Query
            response = query_engine.query(query)
            
            # Display response
            print("-" * 70)
            print("📝 RESPONSE:")
            print("-" * 70)
            print(response)
            print()
            
            # Display source documents
            if hasattr(response, 'source_nodes') and response.source_nodes:
                print("-" * 70)
                print(f"📚 RETRIEVED SOURCES ({len(response.source_nodes)} chunks):")
                print("-" * 70)
                for idx, source_node in enumerate(response.source_nodes, 1):
                    node_text = source_node.node.get_content()
                    score = source_node.score if hasattr(source_node, 'score') else 'N/A'
                    print(f"\n[Source {idx}] Similarity Score: {score}")
                    print("-" * 70)
                    # Display first 500 characters
                    print(node_text[:500])
                    if len(node_text) > 500:
                        print(f"\n... [{len(node_text) - 500} more characters]")
            print("\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}\n")


if __name__ == "__main__":
    main()
