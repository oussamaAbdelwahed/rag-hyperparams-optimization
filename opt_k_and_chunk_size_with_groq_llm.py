"""
RAG Evaluation Tool - Evaluate Pinecone indexes with Groq LLM
Loops over eval dataset, queries one by one (1 question each 30 seconds),
and runs semantic similarity evaluation against expected answers.
"""

import os
import sys
import time
import asyncio
import json
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime

# Load environment variables
load_dotenv()

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.core.evaluation import QueryResponseDataset
from llama_index.vector_stores.pinecone import PineconeVectorStore  # type: ignore
from pinecone import Pinecone  # type: ignore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore
from llama_index.llms.groq import Groq  # type: ignore
from llama_index.core.evaluation import (
    SemanticSimilarityEvaluator,
    BatchEvalRunner,
)
from llama_index.core.postprocessor import SimilarityPostprocessor
import numpy as np


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


def _setup_groq_llm():
    """Configure Groq LLM with high-quality model."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    
    # Use Mixtral - high quality model with good reasoning
    llm = Groq(
        # model="llama-3.3-70b-versatile",
        model="llama-3.1-8b-instant",
        api_key=api_key,
        # temperature=0.7,
        # max_tokens=1024,
    )
    print("✓ Groq LLM configured (mixtral-8x7b-32768, max_tokens=1024)")
    return llm


def _setup_embedding():
    """Configure HuggingFace embedding model."""
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-base-en-v1.5",
        device="cpu",
        trust_remote_code=True,
    )
    print("✓ HuggingFace embedding model configured (BAAI/bge-base-en-v1.5)")
    return embed_model


def _get_eval_batch_runner():
    """Get evaluation batch runner using semantic similarity."""
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    evaluator_s = SemanticSimilarityEvaluator(embed_model=embed_model)
    eval_batch_runner = BatchEvalRunner(
        {"semantic_similarity": evaluator_s}, workers=1, show_progress=True
    )
    return eval_batch_runner


async def main():
    """Main async function to evaluate RAG with Groq LLM."""
    start_time = time.time()
    
    print("="*70)
    print("RAG Evaluation Tool - Groq LLM + Pinecone Vector DB")
    print("="*70)
    
    # Configuration
    PINECONE_INDEX_NAME = "ats-chrono-rag-hyperparams-optim-chunk-256"
    TOP_K = 5
    SIMILARITY_CUTOFF = 0.5
    QUERY_DELAY = 12  # 15 seconds between queries
    
    print(f"\n🔧 Configuration:")
    print(f"   Index: {PINECONE_INDEX_NAME}")
    print(f"   Top-k: {TOP_K}")
    print(f"   Similarity cutoff: {SIMILARITY_CUTOFF}")
    print(f"   Query delay: {QUERY_DELAY}s per question")
    
    # Setup components
    print("\n🔧 Initializing components...")
    
    # Configure embedding model
    embed_model = _setup_embedding()
    Settings.embed_model = embed_model
    
    # Configure Groq LLM
    llm = _setup_groq_llm()
    Settings.llm = llm
    
    # Load index
    print(f"\n📂 Loading Pinecone index...")
    try:
        index = _load_index(PINECONE_INDEX_NAME)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return
    
    # Configure query engine with similarity filtering
    print(f"\n⚙️  Configuring query engine...")
    similarity_processor = SimilarityPostprocessor(similarity_cutoff=SIMILARITY_CUTOFF)
    query_engine = index.as_query_engine(
        similarity_top_k=TOP_K,
        node_postprocessors=[similarity_processor],
    )
    print(f"✓ Query engine ready (top_k={TOP_K}, similarity_cutoff={SIMILARITY_CUTOFF})")
    
    # Load evaluation dataset
    print(f"\n📋 Loading evaluation dataset...")
    eval_dataset = QueryResponseDataset.from_json("eval_dataset_chrono.json")
    eval_qs = eval_dataset.questions
    ref_response_strs = [r for (_, r) in eval_dataset.qr_pairs]
    print(f"✓ Loaded {len(eval_qs)} evaluation questions")
    
    # Run evaluation loop
    print("\n" + "="*70)
    print(f"Starting Evaluation Loop ({len(eval_qs)} questions)")
    print("="*70 + "\n")
    
    pred_response_objs = []
    generated_answers = []  # Store LLM-generated answers
    
    for idx, query in enumerate(eval_qs, 1):
        try:
            print(f"[{idx}/{len(eval_qs)}] Query: {query[:60]}...")
            
            # Query with RAG
            response = query_engine.query(query)
            pred_response_objs.append(response)
            
            # Extract and store the LLM-generated answer
            answer_text = str(response)
            generated_answers.append(answer_text)
            
            # Display retrieved chunks
            if hasattr(response, 'source_nodes') and response.source_nodes:
                print(f"   📄 Retrieved {len(response.source_nodes)} chunks:")
                for node_idx, source_node in enumerate(response.source_nodes, 1):
                    node_text = source_node.node.get_content()
                    score = source_node.score if hasattr(source_node, 'score') else 'N/A'
                    print(f"      Chunk {node_idx} (score: {score}): {node_text[:100]}...")
            else:
                print(f"   ⚠️  No source nodes retrieved")
            
            print(f"   ✓ Response received\n")
            
            # Delay between queries (30 seconds)
            if idx < len(eval_qs):
                print(f"   ⏳ Waiting {QUERY_DELAY}s before next query...")
                for remaining in range(QUERY_DELAY, 0, -1):
                    print(f"      {remaining}s remaining...", end='\r')
                    time.sleep(1)
                print(f"   ✓ Ready for next query\n")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:100]}\n")
            raise
    
    # Run evaluation
    print("\n" + "="*70)
    print("Running Semantic Similarity Evaluation")
    print("="*70 + "\n")
    
    eval_batch_runner = _get_eval_batch_runner()
    eval_results = await eval_batch_runner.aevaluate_responses(
        eval_qs, responses=pred_response_objs, reference=ref_response_strs
    )
    
    # Debug: Print all eval_results keys and structure (matching ats_async format)
    print(f"📊 Evaluation Results Structure:")
    print(f"   Keys in eval_results: {list(eval_results.keys())}")
    for key, value in eval_results.items():
        print(f"   - {key}: type={type(value)}, length={len(value) if hasattr(value, '__len__') else 'N/A'}")
        if hasattr(value, '__iter__') and len(value) > 0:
            first_item = value[0] if isinstance(value, list) else next(iter(value))
            print(f"     First item type: {type(first_item)}")
            if hasattr(first_item, '__dict__'):
                print(f"     First item attributes: {list(vars(first_item).keys())}")
            if hasattr(first_item, 'score'):
                print(f"     Sample scores: {[r.score for r in value[:3]]}")
    
    # Get semantic similarity metric
    semantic_scores = np.array(
        [r.score for r in eval_results["semantic_similarity"]]
    )
    mean_score = semantic_scores.mean()
    
    # End timing
    end_time = time.time()
    duration = end_time - start_time
    
    # Display results (matching ats_async format exactly)
    print("\n" + "="*60)
    print("RESULTS - Groq RAG Evaluation")
    print("="*60)
    print(f"✓ Completed: index_name={PINECONE_INDEX_NAME}, top_k={TOP_K}, score={mean_score:.4f}")
    
    # Save evaluation results to JSON
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "pinecone_index": PINECONE_INDEX_NAME,
            "top_k": TOP_K,
            "similarity_cutoff": SIMILARITY_CUTOFF,
            "query_delay_seconds": QUERY_DELAY,
            "llm_model": "mixtral-8x7b-32768",
            "llm_provider": "groq",
            "embedding_model": "BAAI/bge-base-en-v1.5",
        },
        "results": {
            "mean_score": float(mean_score),
            "min_score": float(np.min(semantic_scores)),
            "max_score": float(np.max(semantic_scores)),
            "std_dev": float(np.std(semantic_scores)),
            "total_questions": len(semantic_scores),
        },
        "individual_scores": [
            {
                "question_idx": idx,
                "question": eval_qs[idx],
                "llm_answer": generated_answers[idx],  # Add LLM-generated answer
                "score": float(score),
            }
            for idx, score in enumerate(semantic_scores)
        ],
        "execution_info": {
            "duration_seconds": round(duration, 2),
            "duration_minutes": round(duration / 60, 2),
        }
    }
    
    output_file = "eval_results_groq.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results saved to {output_file}")
    
    print("\n" + "="*60)
    print("Groq Evaluation Complete!")
    print("="*60)
    print(f"⏱️  Total Execution Time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
