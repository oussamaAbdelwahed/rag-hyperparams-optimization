"""
RAG Evaluation Tool - Evaluate Docling-processed indexes with Groq LLM
Specialized for Software Specification Documents with strict accuracy requirements.

Features:
- System prompt optimized for critical software specs
- No hallucination tolerance - suggests consulting specs when uncertain
- Concise, compact answers
- Semantic similarity evaluation
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
from llama_index.core.prompts import PromptTemplate
import numpy as np


# System prompt for critical software specifications
SYSTEM_PROMPT = """You are a specialized RAG assistant for software functional specifications documentation.

CRITICAL GUIDELINES:
1. ACCURACY IS PARAMOUNT - This is for critical software development where incorrect information can cause serious issues
2. ONLY answer based on the provided context/chunks - DO NOT fabricate or guess information
3. If you are uncertain or the context doesn't clearly support an answer, explicitly state: "I cannot find sufficient information in the specifications to answer this confidently. Please consult the full specification document."
4. Be CONCISE and COMPACT - provide direct, to-the-point answers
5. When answering, cite specific sections/parts from the context when possible
6. If multiple interpretations are possible, acknowledge this and present them briefly
7. Use technical precision - this is for software developers who need exact specifications

RESPONSE FORMAT:
- Start with a direct answer (1-2 sentences when possible)
- Add relevant details only if necessary
- If uncertain, state it clearly and suggest consulting the specs
- Avoid verbose explanations unless the question specifically requires them

Remember: It's better to say "I don't know, check the specs" than to provide low-confidence or fabricated information."""


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
    """Configure Groq LLM with system prompt for software specifications."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    
    # Use high-quality model with strict system prompt
    llm = Groq(
         model="llama-3.1-8b-instant",
        # model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.1,  # Low temperature for more deterministic, factual responses
        system_prompt=SYSTEM_PROMPT,
    )
    print("✓ Groq LLM configured with specialized software specs system prompt")
    print(f"   Model: model=\"llama-3.3-70b-versatile\"")
    print(f"   Temperature: 0.1 (low for factual accuracy)")
    print(f"   System Prompt: Activated (strict accuracy mode)")
    return llm


def _setup_embedding():
    """Configure HuggingFace embedding model."""
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-base-en-v1.5",  # 768 dimensions to match Docling index
        device="cpu",
        trust_remote_code=True,
    )
    print("✓ HuggingFace embedding model configured (BAAI/bge-base-en-v1.5, 768 dim)")
    return embed_model


def _get_eval_batch_runner():
    """Get evaluation batch runner using semantic similarity."""
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")  # 768 dim
    evaluator_s = SemanticSimilarityEvaluator(embed_model=embed_model)
    eval_batch_runner = BatchEvalRunner(
        {"semantic_similarity": evaluator_s}, workers=1, show_progress=True
    )
    return eval_batch_runner


async def main():
    """Main async function to evaluate RAG with Groq LLM."""
    start_time = time.time()
    
    print("="*70)
    print("RAG Evaluation Tool - Docling + Groq LLM (Software Specs Mode)")
    print("="*70)
    
    # Configuration
    PINECONE_INDEX_NAME = "docling-tool-chunking-ats-chrono"
    TOP_K = 80
    SIMILARITY_CUTOFF = 0.5
    QUERY_DELAY = 10  # 10 seconds between queries
    
    print(f"\n🔧 Configuration:")
    print(f"   Index: {PINECONE_INDEX_NAME}")
    print(f"   Top-k: {TOP_K}")
    print(f"   Similarity cutoff: {SIMILARITY_CUTOFF}")
    print(f"   Query delay: {QUERY_DELAY}s per question")
    print(f"   Mode: Critical Software Specifications (strict accuracy)")
    
    # Setup components
    print("\n🔧 Initializing components...")
    
    # Configure embedding model
    embed_model = _setup_embedding()
    Settings.embed_model = embed_model
    
    # Configure Groq LLM with system prompt
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
    retrieved_chunks_list = []  # Store retrieved chunks for each question
    
    for idx, query in enumerate(eval_qs, 1):
        try:
            print(f"[{idx}/{len(eval_qs)}] Query: {query[:60]}...")
            
            # Query with RAG
            response = query_engine.query(query)
            pred_response_objs.append(response)
            
            # Extract and store the LLM-generated answer
            answer_text = str(response)
            generated_answers.append(answer_text)
            
            # Extract and store retrieved chunks
            chunks_for_question = []
            if hasattr(response, 'source_nodes') and response.source_nodes:
                print(f"   📄 Retrieved {len(response.source_nodes)} chunks:")
                for node_idx, source_node in enumerate(response.source_nodes, 1):
                    node_text = source_node.node.get_content()
                    score = source_node.score if hasattr(source_node, 'score') else 'N/A'
                    chunks_for_question.append({
                        "chunk_index": node_idx,
                        "text": node_text,
                        "score": float(score) if isinstance(score, (int, float)) else score,
                    })
                    print(f"      Chunk {node_idx} (score: {score:.3f}): {node_text[:80]}...")
            else:
                print(f"   ⚠️  No source nodes retrieved")
            
            retrieved_chunks_list.append(chunks_for_question)
            
            # Show answer preview
            answer_preview = answer_text[:150] + "..." if len(answer_text) > 150 else answer_text
            print(f"   💬 Answer: {answer_preview}")
            print(f"   ✓ Response received\n")
            
            # Delay between queries
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
    
    # Get semantic similarity metric
    semantic_scores = np.array(
        [r.score for r in eval_results["semantic_similarity"]]
    )
    mean_score = semantic_scores.mean()
    
    # End timing
    end_time = time.time()
    duration = end_time - start_time
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS - Docling + Groq RAG Evaluation (Software Specs Mode)")
    print("="*60)
    print(f"✓ Completed: index={PINECONE_INDEX_NAME}, top_k={TOP_K}, score={mean_score:.4f}")
    
    # Save evaluation results to JSON
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "pinecone_index": PINECONE_INDEX_NAME,
            "top_k": TOP_K,
            "similarity_cutoff": SIMILARITY_CUTOFF,
            "query_delay_seconds": QUERY_DELAY,
            "llm_model": "model=\"llama-3.3-70b-versatile\"",
            "llm_provider": "groq",
            "llm_temperature": 0.1,
            "embedding_model": "BAAI/bge-base-en-v1.5",
            "embedding_dimension": 768,
            "system_prompt_mode": "critical_software_specifications",
            "chunking_method": "docling_semantic",
        },
        "system_prompt": SYSTEM_PROMPT,
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
                "expected_answer": ref_response_strs[idx],
                "llm_answer": generated_answers[idx],
                "score": float(score),
                "retrieved_chunks": retrieved_chunks_list[idx],
            }
            for idx, score in enumerate(semantic_scores)
        ],
        "execution_info": {
            "duration_seconds": round(duration, 2),
            "duration_minutes": round(duration / 60, 2),
        }
    }
    
    output_file = "eval_results_docling_groq.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results saved to {output_file}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("📊 Evaluation Summary")
    print("="*60)
    print(f"Mean Semantic Similarity: {mean_score:.4f}")
    print(f"Min Score: {np.min(semantic_scores):.4f}")
    print(f"Max Score: {np.max(semantic_scores):.4f}")
    print(f"Std Dev: {np.std(semantic_scores):.4f}")
    print(f"Total Questions: {len(semantic_scores)}")
    
    print("\n" + "="*60)
    print("Docling + Groq Evaluation Complete!")
    print("="*60)
    print(f"⏱️  Total Execution Time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
