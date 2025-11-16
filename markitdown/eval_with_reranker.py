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
import openai
from typing import List, Dict, Any

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
from sentence_transformers import CrossEncoder  # For reranking


# System prompt for critical software specifications
SYSTEM_PROMPT = """You are a specialized RAG assistant for software functional specifications documentation.

CRITICAL GUIDELINES:
1. ACCURACY IS NEEDED - Accuracy and confidence is required since incorrect information can cause issues
2. ONLY answer based on the provided context/chunks - DO NOT fabricate or guess information
3. If you are highly uncertain or the context doesn't clearly support an answer, explicitly state: "I cannot find sufficient information in the specifications to answer this confidently. Please consult the full specification document."
4. Be CONCISE and COMPACT - provide direct, to-the-point answers
5. When answering, cite specific sections/parts from the context when possible
6. If multiple interpretations are possible, acknowledge this and present them briefly
7. Use technical precision - this is for software developers and PO (product owners) who need exact specifications/need to validate their understanding

RESPONSE FORMAT:
- Start with a direct answer (1-2 sentences when possible)
- Add relevant details only if necessary
- If uncertain, state it clearly and suggest consulting the specs
- Avoid verbose explanations unless the question specifically requires them

Remember: It's better to say "I don't know, check the specs" than to provide low-confidence or fabricated information."""


# Global state for API key rotation
GROQ_API_KEYS = [
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
    os.getenv("GROQ_API_KEY_4"),
    os.getenv("GROQ_API_KEY_5"),
    os.getenv("GROQ_API_KEY_6"),
    os.getenv("GROQ_API_KEY_7"),
    os.getenv("GROQ_API_KEY_8"),
]
GROQ_API_KEYS = [key for key in GROQ_API_KEYS if key]  # Filter out None values
CURRENT_API_KEY_INDEX = 0


EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
# PINECONE_INDEX_NAME = "markitdown-parse-and-md-node-parser-split"
PINECONE_INDEX_NAME = "markitdown-docling-hybrid-chunker" 

# Reranking configuration
INITIAL_TOP_K = 25  # Fetch top 25 chunks from Pinecone
RERANK_TOP_N = 4    # After reranking, keep only top 4 for LLM
RERANKER_MODEL = "BAAI/bge-reranker-large"  # Same family as embedding model, better quality

# Original parameters for similarity filtering
SIMILARITY_CUTOFF = 0.55
QUERY_DELAY = 10  # 10 seconds between queries

# Path to root directory datasets (markitdown folder is one level down from root)
ROOT_DIR = Path(__file__).parent.parent
EVAL_DATASET_PATH = ROOT_DIR / "eval_dataset_chrono.json"
OUTPUT_DIR = Path(__file__).parent  # Save outputs in markitdown folder
OUTPUT_FILE = OUTPUT_DIR / "eval_results_with_reranking_groq.json"
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
    global CURRENT_API_KEY_INDEX
    
    if not GROQ_API_KEYS:
        raise ValueError("No GROQ API keys found in environment variables")
    
    api_key = GROQ_API_KEYS[CURRENT_API_KEY_INDEX]
    
    # Use high-quality model with strict system prompt
    llm = Groq(
        #model="llama-3.1-8b-instant",
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.1,  # Low temperature for more deterministic, factual responses
        system_prompt=SYSTEM_PROMPT,
    )
    print("✓ Groq LLM configured with specialized software specs system prompt")
    print(f"   Model: llama-3.1-8b-instant")
    print(f"   Temperature: 0.1 (low for factual accuracy)")
    print(f"   System Prompt: Activated (strict accuracy mode)")
    print(f"   Using API Key #{CURRENT_API_KEY_INDEX + 1} of {len(GROQ_API_KEYS)}")
    return llm


def _switch_to_next_groq_api_key():
    """Switch to the next Groq API key and reinitialize the LLM."""
    global CURRENT_API_KEY_INDEX
    
    CURRENT_API_KEY_INDEX = (CURRENT_API_KEY_INDEX + 1) % len(GROQ_API_KEYS)
    
    print(f"\n🔄 Switching to API Key #{CURRENT_API_KEY_INDEX + 1}")
    
    llm = _setup_groq_llm()
    Settings.llm = llm
    
    return llm


def _is_rate_limit_error(error):
    """Check if the error is a rate limit error from Groq."""
    # Check for openai.RateLimitError exception type
    error_type = type(error).__name__
    if error_type == "RateLimitError":
        return True
    
    # Also check error message content
    error_str = str(error).lower()
    return any(phrase in error_str for phrase in [
        "rate limit",
        "rate_limit",
        "too many requests",
        "quota exceeded",
        "429",
        "limit exceeded",
        "rate_limit_exceeded",
        "tokens per day"
    ])


def _setup_embedding():
    """Configure HuggingFace embedding model."""
    embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL,  # 1024 dimensions to match Pinecone index
        device="cpu",
        trust_remote_code=True,
    )
    print("✓ HuggingFace embedding model configured (BAAI/bge-large-en-v1.5, 1024 dim)")
    return embed_model


def _setup_reranker():
    """Configure the reranker model for reranking retrieved chunks."""
    reranker = CrossEncoder(RERANKER_MODEL, max_length=512, device="cpu")
    print(f"✓ Reranker model configured ({RERANKER_MODEL})")
    return reranker


def _rerank_chunks(reranker, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Rerank chunks using the cross-encoder reranker.
    
    Args:
        reranker: CrossEncoder model
        query: The user query
        chunks: List of chunk dictionaries with 'text' and 'score' keys
    
    Returns:
        List of reranked chunks with updated 'rerank_score' and 'original_rank' keys
    """
    if not chunks:
        return []
    
    # Prepare pairs for reranking
    pairs = [(query, chunk["text"]) for chunk in chunks]
    
    # Get reranking scores
    rerank_scores = reranker.predict(pairs)
    
    # Add rerank scores and original rank to chunks
    for idx, chunk in enumerate(chunks):
        chunk["rerank_score"] = float(rerank_scores[idx])
        chunk["original_rank"] = idx + 1
        chunk["original_score"] = chunk.get("score", "N/A")
    
    # Sort by rerank score (descending)
    reranked_chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    
    # Add new rank after reranking
    for idx, chunk in enumerate(reranked_chunks):
        chunk["new_rank"] = idx + 1
    
    return reranked_chunks


def _get_eval_batch_runner():
    """Get evaluation batch runner using semantic similarity."""
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL, device="cpu", trust_remote_code=True)
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
    

    
    print(f"\n🔧 Configuration:")
    print(f"   Index: {PINECONE_INDEX_NAME}")
    print(f"   Initial retrieval (from Pinecone): {INITIAL_TOP_K}")
    print(f"   Reranker model: {RERANKER_MODEL}")
    print(f"   Top-N after reranking (fed to LLM): {RERANK_TOP_N}")
    print(f"   Similarity cutoff: {SIMILARITY_CUTOFF}")
    print(f"   Query delay: {QUERY_DELAY}s per question")
    print(f"   Mode: Critical Software Specifications (strict accuracy)")
    
    # Setup components
    print("\n🔧 Initializing components...")
    
    # Configure embedding model
    embed_model = _setup_embedding()
    Settings.embed_model = embed_model
    
    # Configure reranker
    reranker = _setup_reranker()
    
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
        similarity_top_k=INITIAL_TOP_K,  # Fetch more chunks initially for reranking
        node_postprocessors=[similarity_processor],
    )
    print(f"✓ Query engine ready (initial_top_k={INITIAL_TOP_K}, similarity_cutoff={SIMILARITY_CUTOFF})")
    print(f"✓ Reranking pipeline: {INITIAL_TOP_K} chunks → rerank → top {RERANK_TOP_N} to LLM")
    
    # Load evaluation dataset
    print(f"\n📋 Loading evaluation dataset...")
    eval_dataset = QueryResponseDataset.from_json(str(EVAL_DATASET_PATH))
    eval_qs = eval_dataset.questions
    ref_response_strs = [r for (_, r) in eval_dataset.qr_pairs]
    print(f"✓ Loaded {len(eval_qs)} evaluation questions")
    
    # Run evaluation loop
    print("\n" + "="*70)
    print(f"Starting Evaluation Loop ({len(eval_qs)} questions)")
    print("="*70 + "\n")
    
    pred_response_objs = []
    generated_answers = []  # Store LLM-generated answers
    original_chunks_list = []  # Store original retrieved chunks (top 25)
    reranked_chunks_list = []  # Store reranked chunks (all 25 with scores)
    final_chunks_to_llm_list = []  # Store top 4 chunks fed to LLM
    api_keys_used = set()  # Track which API keys were used
    
    for idx, query in enumerate(eval_qs, 1):
        max_retries = len(GROQ_API_KEYS)
        retry_count = 0
        response = None
        answer_text = ""
        original_chunks = []
        reranked_chunks = []
        top_n_chunks = []
        
        while retry_count < max_retries:
            try:
                print(f"[{idx}/{len(eval_qs)}] Query: {query[:60]}...")
                
                # Step 1: Retrieve initial chunks from Pinecone (top 25)
                print(f"   🔍 Step 1: Retrieving top {INITIAL_TOP_K} chunks from Pinecone...")
                retriever = index.as_retriever(similarity_top_k=INITIAL_TOP_K)
                initial_nodes = retriever.retrieve(query)
                
                # Store original chunks
                original_chunks = []
                for node_idx, source_node in enumerate(initial_nodes, 1):
                    node_text = source_node.node.get_content()
                    score = source_node.score if hasattr(source_node, 'score') else 0.0
                    original_chunks.append({
                        "chunk_index": node_idx,
                        "text": node_text,
                        "score": float(score) if isinstance(score, (int, float)) else score,
                    })
                
                print(f"      ✓ Retrieved {len(original_chunks)} chunks")
                
                # Step 2: Rerank chunks using cross-encoder
                print(f"   🎯 Step 2: Reranking {len(original_chunks)} chunks...")
                reranked_chunks = _rerank_chunks(reranker, query, original_chunks.copy())
                print(f"      ✓ Reranking complete")
                
                # Show top 5 reranked chunks
                print(f"      Top 5 after reranking:")
                for i, chunk in enumerate(reranked_chunks[:5], 1):
                    print(f"         {i}. Rank {chunk['original_rank']}→{chunk['new_rank']} "
                          f"(rerank: {chunk['rerank_score']:.3f}, orig: {chunk['original_score']:.3f})")
                
                # Step 3: Select top N chunks for LLM
                top_n_chunks = reranked_chunks[:RERANK_TOP_N]
                print(f"   ✂️  Step 3: Selecting top {RERANK_TOP_N} chunks for LLM context")
                
                # Step 4: Synthesize response with LLM using top N chunks
                print(f"   🤖 Step 4: Generating answer with LLM...")
                
                # Create a custom context string from top N chunks
                context_parts = []
                for i, chunk in enumerate(top_n_chunks, 1):
                    context_parts.append(f"[Context {i}]:\n{chunk['text']}\n")
                
                context_str = "\n".join(context_parts)
                
                # Create the prompt for the LLM
                prompt = f"""{SYSTEM_PROMPT}

Context Information:
{context_str}

Query: {query}

Answer:"""
                
                # Query LLM directly
                llm_response = Settings.llm.complete(prompt)
                answer_text = str(llm_response)
                
                # Create a response object for evaluation
                # We'll use a simple dict-like object that the evaluator can work with
                class SimpleResponse:
                    def __init__(self, response_text, source_nodes):
                        self.response = response_text
                        self.source_nodes = source_nodes
                    
                    def __str__(self):
                        return self.response
                
                # Convert reranked chunks back to NodeWithScore objects
                from llama_index.core.schema import NodeWithScore, TextNode
                source_nodes = []
                for chunk in top_n_chunks:
                    text_node = TextNode(text=chunk["text"])
                    node_with_score = NodeWithScore(
                        node=text_node,
                        score=chunk["rerank_score"]
                    )
                    source_nodes.append(node_with_score)
                
                response = SimpleResponse(
                    response_text=answer_text,
                    source_nodes=source_nodes
                )
                
                # Track which API key was used for this successful query
                api_keys_used.add(CURRENT_API_KEY_INDEX + 1)
                
                # If we got here, the query was successful - break out of retry loop
                break
                
            except openai.RateLimitError as e:
                retry_count += 1
                print(f"   ⚠️  Rate limit hit on API Key #{CURRENT_API_KEY_INDEX + 1}")
                
                if retry_count < max_retries:
                    print(f"   🔄 Switching to next API key (attempt {retry_count + 1}/{max_retries})...")
                    llm = _switch_to_next_groq_api_key()
                    
                    # Recreate query engine with new LLM
                    similarity_processor = SimilarityPostprocessor(similarity_cutoff=SIMILARITY_CUTOFF)
                    query_engine = index.as_query_engine(
                        similarity_top_k=INITIAL_TOP_K,
                        node_postprocessors=[similarity_processor],
                    )
                    
                    print(f"   ⏳ Waiting 5 seconds before retry...")
                    time.sleep(5)
                else:
                    print(f"   ❌ All API keys exhausted. Cannot continue.")
                    raise Exception(f"All {max_retries} Groq API keys have hit rate limits") from e
                    
            except Exception as e:
                if _is_rate_limit_error(e):
                    retry_count += 1
                    print(f"   ⚠️  Rate limit hit on API Key #{CURRENT_API_KEY_INDEX + 1}")
                    
                    if retry_count < max_retries:
                        print(f"   🔄 Switching to next API key (attempt {retry_count + 1}/{max_retries})...")
                        llm = _switch_to_next_groq_api_key()
                        
                        # Recreate query engine with new LLM
                        similarity_processor = SimilarityPostprocessor(similarity_cutoff=SIMILARITY_CUTOFF)
                        query_engine = index.as_query_engine(
                            similarity_top_k=INITIAL_TOP_K,
                            node_postprocessors=[similarity_processor],
                        )
                        
                        print(f"   ⏳ Waiting 5 seconds before retry...")
                        time.sleep(5)
                    else:
                        print(f"   ❌ All API keys exhausted. Cannot continue.")
                        raise Exception(f"All {max_retries} Groq API keys have hit rate limits") from e
                else:
                    # Non-rate-limit error, re-raise it
                    print(f"   ❌ Error: {str(e)[:100]}")
                    raise
        
        if response is None:
            raise Exception(f"Failed to get response for query {idx} after all retries")
        
        # Store results
        pred_response_objs.append(response)
        generated_answers.append(answer_text)
        original_chunks_list.append(original_chunks)
        reranked_chunks_list.append(reranked_chunks)
        final_chunks_to_llm_list.append(top_n_chunks)
        
        # Show answer preview
        answer_preview = answer_text[:150] + "..." if len(answer_text) > 150 else answer_text
        print(f"   💬 Answer: {answer_preview}")
        print(f"   ✓ Response generated successfully\n")
        
        # Delay between queries
        if idx < len(eval_qs):
            print(f"   ⏳ Waiting {QUERY_DELAY}s before next query...")
            for remaining in range(QUERY_DELAY, 0, -1):
                print(f"      {remaining}s remaining...", end='\r')
                time.sleep(1)
            print(f"   ✓ Ready for next query\n")
    
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
    print("\n" + "="*70)
    print("RESULTS - RAG with Reranking Evaluation (Software Specs Mode)")
    print("="*70)
    print(f"✓ Completed: index={PINECONE_INDEX_NAME}, initial_k={INITIAL_TOP_K}, rerank_n={RERANK_TOP_N}, score={mean_score:.4f}")
    
    # Save evaluation results to JSON
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "pinecone_index": PINECONE_INDEX_NAME,
            "initial_top_k": INITIAL_TOP_K,
            "reranker_model": RERANKER_MODEL,
            "rerank_top_n": RERANK_TOP_N,
            "similarity_cutoff": SIMILARITY_CUTOFF,
            "query_delay_seconds": QUERY_DELAY,
            "llm_model": "llama-3.1-8b-instant",
            "llm_provider": "groq",
            "llm_temperature": 0.1,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimension": 1024,
            "system_prompt_mode": "critical_software_specifications",
            "chunking_method": "markitdown_with_reranking",
            "pipeline_stages": "retrieve → rerank → synthesize",
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
                "original_chunks": original_chunks_list[idx],  # All 25 chunks from Pinecone
                "reranked_chunks": reranked_chunks_list[idx],  # All 25 with rerank scores
                "final_chunks_to_llm": final_chunks_to_llm_list[idx],  # Top 4 used for generation
            }
            for idx, score in enumerate(semantic_scores)
        ],
        "execution_info": {
            "duration_seconds": round(duration, 2),
            "duration_minutes": round(duration / 60, 2),
        },
        "api_keys_info": {
            "total_api_keys_available": len(GROQ_API_KEYS),
            "api_keys_used": sorted(list(api_keys_used)),
        }
    }
    
    output_file = OUTPUT_DIR / "eval_results_with_reranker.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results saved to {output_file}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("📊 Evaluation Summary")
    print("="*70)
    print(f"Mean Semantic Similarity: {mean_score:.4f}")
    print(f"Min Score: {np.min(semantic_scores):.4f}")
    print(f"Max Score: {np.max(semantic_scores):.4f}")
    print(f"Std Dev: {np.std(semantic_scores):.4f}")
    print(f"Total Questions: {len(semantic_scores)}")
    
    print("\n" + "="*70)
    print("🎯 Reranking Pipeline Summary")
    print("="*70)
    print(f"Initial retrieval: {INITIAL_TOP_K} chunks from Pinecone")
    print(f"Reranker model: {RERANKER_MODEL}")
    print(f"Top-N after reranking: {RERANK_TOP_N} chunks to LLM")
    print(f"Pipeline: Retrieve → Rerank → Synthesize")
    
    print("\n" + "="*70)
    print("🔑 API Keys Used During Evaluation")
    print("="*70)
    for key_num in sorted(api_keys_used):
        print(f"  • GROQ_API_KEY_{key_num}")
    print(f"Total API keys utilized: {len(api_keys_used)}")
    
    print("\n" + "="*70)
    print("RAG with Reranking Evaluation Complete!")
    print("="*70)
    print(f"⏱️  Total Execution Time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
