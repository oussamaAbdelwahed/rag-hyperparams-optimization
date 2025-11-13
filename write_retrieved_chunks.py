"""
Script to retrieve top-k chunks from Pinecone for evaluation dataset questions.
Writes retrieved chunks to JSON file for analysis.

Usage:
    python write_retrieved_chunks.py
"""

import os
import json
import time
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.core.evaluation import QueryResponseDataset
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# Configuration
PINECONE_INDEX_NAME = "enhanced-preprocess-tool-chunking-ats-chrono"
TOP_K = 80
EVAL_DATASET_FILE = "eval_dataset_chrono.json"
OUTPUT_FILE = "retrieved_chunks_top80.json"


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


def _load_index(index_name):
    """Load existing Pinecone index."""
    pc = _initialize_pinecone()
    
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


def _setup_embedding():
    """Configure HuggingFace embedding model."""
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-base-en-v1.5",
        device="cpu",
        trust_remote_code=True,
    )
    print("✓ HuggingFace embedding model configured (BAAI/bge-base-en-v1.5, 768 dim)")
    return embed_model


def main():
    """Main function to retrieve chunks for evaluation dataset."""
    start_time = time.time()
    
    print("="*70)
    print("Chunk Retrieval Tool - Extract Top-K Chunks from Pinecone")
    print("="*70)
    
    print(f"\n🔧 Configuration:")
    print(f"   Index: {PINECONE_INDEX_NAME}")
    print(f"   Top-k: {TOP_K}")
    print(f"   Eval Dataset: {EVAL_DATASET_FILE}")
    print(f"   Output File: {OUTPUT_FILE}")
    
    # Setup components
    print("\n🔧 Initializing components...")
    
    # Configure embedding model
    embed_model = _setup_embedding()
    Settings.embed_model = embed_model
    
    # Load index
    print(f"\n📂 Loading Pinecone index...")
    try:
        index = _load_index(PINECONE_INDEX_NAME)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return
    
    # Configure retriever
    print(f"\n⚙️  Configuring retriever...")
    retriever = index.as_retriever(similarity_top_k=TOP_K)
    print(f"✓ Retriever ready (top_k={TOP_K})")
    
    # Load evaluation dataset
    print(f"\n📋 Loading evaluation dataset...")
    if not os.path.exists(EVAL_DATASET_FILE):
        print(f"❌ Error: Evaluation dataset file not found: {EVAL_DATASET_FILE}")
        return
    
    eval_dataset = QueryResponseDataset.from_json(EVAL_DATASET_FILE)
    eval_qs = eval_dataset.questions
    ref_response_strs = [r for (_, r) in eval_dataset.qr_pairs]
    print(f"✓ Loaded {len(eval_qs)} evaluation questions")
    
    # Retrieve chunks for each question
    print("\n" + "="*70)
    print(f"Starting Chunk Retrieval ({len(eval_qs)} questions)")
    print("="*70 + "\n")
    
    results = []
    
    for idx, query in enumerate(eval_qs):
        print(f"[{idx + 1}/{len(eval_qs)}] Query: {query[:80]}...")
        
        try:
            # Retrieve chunks
            retrieved_nodes = retriever.retrieve(query)
            
            # Extract and format chunks
            chunks = []
            for node_idx, source_node in enumerate(retrieved_nodes):
                node_text = source_node.node.get_content()
                score = source_node.score if hasattr(source_node, 'score') else None
                
                chunk_data = {
                    "chunk_index": node_idx + 1,
                    "text": node_text,
                    "score": float(score) if score is not None else None,
                }
                
                # Add metadata if available
                if hasattr(source_node.node, 'metadata') and source_node.node.metadata:
                    chunk_data["metadata"] = source_node.node.metadata
                
                chunks.append(chunk_data)
            
            # Sort chunks by score (descending)
            chunks.sort(key=lambda x: x["score"] if x["score"] is not None else -1, reverse=True)
            
            # Update chunk_index after sorting
            for new_idx, chunk in enumerate(chunks, 1):
                chunk["chunk_index"] = new_idx
            
            print(f"   ✓ Retrieved {len(chunks)} chunks")
            if chunks:
                print(f"   📊 Score range: {chunks[0]['score']:.4f} (best) to {chunks[-1]['score']:.4f} (worst)")
            
            # Add to results
            result_entry = {
                "question_idx": idx,
                "question": query,
                "expected_answer": ref_response_strs[idx],
                "retrieved_chunks": chunks,
            }
            results.append(result_entry)
            
        except Exception as e:
            print(f"   ❌ Error retrieving chunks: {str(e)}")
            # Add empty result for this question
            result_entry = {
                "question_idx": idx,
                "question": query,
                "expected_answer": ref_response_strs[idx],
                "retrieved_chunks": [],
                "error": str(e),
            }
            results.append(result_entry)
    
    # Save results to JSON
    print("\n" + "="*70)
    print("Saving Results")
    print("="*70)
    
    output_data = {
        "metadata": {
            "pinecone_index": PINECONE_INDEX_NAME,
            "top_k": TOP_K,
            "embedding_model": "BAAI/bge-base-en-v1.5",
            "embedding_dimension": 768,
            "total_questions": len(eval_qs),
            "eval_dataset_file": EVAL_DATASET_FILE,
        },
        "results": results,
    }
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Results saved to {OUTPUT_FILE}")
    
    # End timing
    end_time = time.time()
    duration = end_time - start_time
    
    # Print summary
    print("\n" + "="*70)
    print("📊 Summary")
    print("="*70)
    print(f"Total Questions Processed: {len(results)}")
    print(f"Chunks per Question: {TOP_K}")
    print(f"Output File: {OUTPUT_FILE}")
    print(f"⏱️  Execution Time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print("="*70)
    print("✓ Chunk retrieval complete!")


if __name__ == "__main__":
    main()
