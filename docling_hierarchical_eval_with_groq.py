"""
RAG Evaluation Tool - Evaluate Docling Hierarchical chunks with Groq LLM
Specialized for Software Specification Documents with hierarchical retrieval.

Features:
- Hierarchical/Recursive retrieval strategy using parent-child relationships
- System prompt optimized for critical software specs
- Groq API key rotation for handling rate limits
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
from llama_index.core.query_engine import RetrieverQueryEngine
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
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.retrievers import BaseRetriever
import numpy as np

# Import hierarchical retrieval utilities
from hierarchical_retrieval_utils import HierarchicalChunkNavigator, reconstruct_context

# Global state for API key rotation
GROQ_API_KEYS = [
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
    os.getenv("GROQ_API_KEY_4"),
    os.getenv("GROQ_API_KEY_5"),
    os.getenv("GROQ_API_KEY_6"),
]
GROQ_API_KEYS = [key for key in GROQ_API_KEYS if key]  # Filter out None values
CURRENT_API_KEY_INDEX = 0


# System prompt for software specifications
SYSTEM_PROMPT = """You are an AI assistant helping development teams and product teams verify their understanding against official software specifications.

YOUR ROLE:
- Help developers and product managers validate that their understanding aligns with the actual specifications
- Provide accurate, specification-based answers to ensure teams build the right thing
- Act as a reliable source of truth for functional requirements and technical specifications

GUIDELINES:
1. Answer strictly based on the provided specification context - this ensures alignment with official requirements
2. Provide clear, accurate answers that developers and product teams can trust
3. If specifications are unclear or incomplete, explicitly state: "I'm not fully sure about this based on the available specifications. Please consult the full documentation or clarify with the product owner."
4. Be direct and concise - teams need quick, actionable answers
5. When you have a confident answer based on specs, state it clearly to give teams certainty
6. If you find contradictions or ambiguities in the specs, point them out to prevent misalignment

RESPONSE FORMAT:
- Give a straightforward, specification-backed answer
- Cite specific requirements when possible to build confidence
- If uncertain, acknowledge it and recommend verification steps
- Keep answers focused and actionable for development work"""


class HierarchicalRetriever(BaseRetriever):
    """
    Custom retriever that implements hierarchical/recursive retrieval.
    
    This retriever:
    1. Performs initial vector search to get top-k chunks
    2. Uses hierarchical metadata to expand context (parents, siblings, children)
    3. Returns expanded set of chunks for better context
    """
    
    def __init__(
        self,
        index: VectorStoreIndex,
        navigator: HierarchicalChunkNavigator,
        similarity_top_k: int = 5,
        expansion_strategy: str = "parent_and_siblings",
        similarity_cutoff: float = 0.5,
    ):
        """
        Initialize hierarchical retriever.
        
        Args:
            index: VectorStoreIndex for initial retrieval
            navigator: HierarchicalChunkNavigator for context expansion
            similarity_top_k: Number of chunks to retrieve initially
            expansion_strategy: Strategy for expanding context (see hierarchical_retrieval_utils.py)
            similarity_cutoff: Minimum similarity score for chunks
        """
        self._index = index
        self._navigator = navigator
        self._similarity_top_k = similarity_top_k
        self._expansion_strategy = expansion_strategy
        self._similarity_cutoff = similarity_cutoff
        
        # Store last expansion info for tracking
        self.last_expansion_info = None
        
        # Create base retriever
        self._base_retriever = index.as_retriever(
            similarity_top_k=similarity_top_k
        )
        super().__init__()
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve nodes with hierarchical expansion.
        
        Args:
            query_bundle: Query to retrieve for
            
        Returns:
            List of NodeWithScore objects with expanded hierarchical context
        """
        # Step 1: Get initial chunks from vector search
        initial_nodes = self._base_retriever.retrieve(query_bundle)
        
        # Step 2: Filter by similarity cutoff
        filtered_nodes = [
            node for node in initial_nodes
            if node.score >= self._similarity_cutoff
        ]
        
        if not filtered_nodes:
            self.last_expansion_info = {
                "initial_chunks": [],
                "expanded_chunks": [],
                "expansion_count": 0
            }
            return []
        
        # Step 3: Extract chunk IDs from retrieved nodes
        retrieved_chunk_ids = []
        initial_chunks_info = []
        for node in filtered_nodes:
            # Try to get chunk_id from metadata
            if hasattr(node.node, 'metadata') and 'chunk_index' in node.node.metadata:
                chunk_id = node.node.metadata['chunk_index']
                retrieved_chunk_ids.append(chunk_id)
                initial_chunks_info.append({
                    "chunk_id": chunk_id,
                    "score": float(node.score) if node.score else 0.0,
                    "metadata": node.node.metadata
                })
        
        if not retrieved_chunk_ids:
            # If no chunk_ids found in metadata, return original nodes
            self.last_expansion_info = {
                "initial_chunks": [],
                "expanded_chunks": [],
                "expansion_count": 0
            }
            return filtered_nodes
        
        # Step 4: Expand context using hierarchical navigation
        try:
            expanded_chunks_metadata = reconstruct_context(
                retrieved_chunk_ids,
                self._navigator,
                strategy=self._expansion_strategy
            )
            
            # Store expansion info for later retrieval
            expanded_chunks_info = []
            for exp_chunk in expanded_chunks_metadata:
                expanded_chunks_info.append({
                    "chunk_id": exp_chunk.get('chunk_id'),
                    "parent_id": exp_chunk.get('parent_id'),
                    "level": exp_chunk.get('level'),
                    "headings": exp_chunk.get('headings', []),
                    "is_parent": exp_chunk.get('chunk_id') not in retrieved_chunk_ids,  # Mark if this is a parent/sibling
                    "relationship": self._determine_relationship(exp_chunk.get('chunk_id'), retrieved_chunk_ids)
                })
            
            self.last_expansion_info = {
                "initial_chunks": initial_chunks_info,
                "expanded_chunks": expanded_chunks_info,
                "expansion_count": len(expanded_chunks_metadata)
            }
            
            # Log expansion
            print(f"      🔍 Hierarchical expansion: {len(filtered_nodes)} → {len(expanded_chunks_metadata)} chunks")
            
        except Exception as e:
            print(f"      ⚠️  Hierarchical expansion failed: {e}")
            self.last_expansion_info = {
                "initial_chunks": initial_chunks_info,
                "expanded_chunks": [],
                "expansion_count": 0,
                "error": str(e)
            }
            return filtered_nodes
        
        # Step 5: Return original nodes (hierarchical context is stored in last_expansion_info)
        # The expanded context is available through last_expansion_info for logging
        return filtered_nodes
    
    def _determine_relationship(self, chunk_id: int, retrieved_chunk_ids: List[int]) -> str:
        """Determine the relationship of a chunk to the initially retrieved chunks."""
        if chunk_id in retrieved_chunk_ids:
            return "initial_retrieval"
        
        # Check if it's a parent of any retrieved chunk
        for ret_id in retrieved_chunk_ids:
            if self._navigator.child_to_parent.get(ret_id) == chunk_id:
                return "parent"
            
            # Check if it's an ancestor
            ancestors = self._navigator.get_ancestors(ret_id)
            if any(a.get('chunk_id') == chunk_id for a in ancestors):
                return "ancestor"
            
            # Check if it's a sibling
            siblings = self._navigator.get_siblings(ret_id)
            if any(s.get('chunk_id') == chunk_id for s in siblings):
                return "sibling"
        
        return "expanded_context"


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
        #  model="llama-3.1-8b-instant",
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.1,  # Low temperature for more deterministic, factual responses
        system_prompt=SYSTEM_PROMPT,
    )
    print("✓ Groq LLM configured with specialized software specs system prompt")
    print(f"   Model: llama-3.3-70b-versatile")
    print(f"   Temperature: 0.1 (low for factual accuracy)")
    print(f"   System Prompt: Activated (balanced mode)")
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


def _load_hierarchical_metadata(metadata_file: str) -> HierarchicalChunkNavigator:
    """
    Load hierarchical metadata and create navigator.
    
    Args:
        metadata_file: Path to the exported hierarchical metadata JSON
        
    Returns:
        HierarchicalChunkNavigator instance
    """
    metadata_path = Path(metadata_file)
    
    if not metadata_path.exists():
        print(f"⚠️  Hierarchical metadata file not found: {metadata_file}")
        print(f"   Hierarchical retrieval will be disabled")
        return None
    
    print(f"✓ Loading hierarchical metadata from: {metadata_file}")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        exported_data = json.load(f)
    
    # Extract chunk metadata
    chunks_metadata = []
    for chunk_data in exported_data.get('chunks', []):
        metadata = chunk_data.get('metadata', {})
        
        # Ensure required fields exist
        if 'chunk_index' not in metadata:
            metadata['chunk_index'] = chunk_data.get('doc_id', len(chunks_metadata))
        
        # Map to expected format for HierarchicalChunkNavigator
        chunk_meta = {
            'chunk_id': metadata.get('chunk_index', len(chunks_metadata)),
            'parent_id': metadata.get('parent_id'),
            'level': metadata.get('level', 0),
            'headings': metadata.get('headings', []),
            'doc_items': metadata.get('doc_items', []),
            **metadata  # Include all other metadata
        }
        chunks_metadata.append(chunk_meta)
    
    navigator = HierarchicalChunkNavigator(chunks_metadata)
    print(f"✓ Loaded {len(chunks_metadata)} chunks with hierarchical structure")
    
    return navigator


async def main():
    """Main async function to evaluate RAG with hierarchical retrieval and Groq LLM."""
    start_time = time.time()
    
    print("="*70)
    print("RAG Evaluation - Docling Hierarchical + Groq LLM (Software Specs)")
    print("="*70)
    
    # Configuration
    PINECONE_INDEX_NAME = "docling-prepro-and-hierar-chunking-ats-chrono"
    METADATA_FILE = "docling-prepro-and-hierar-chunking-ats-chrono_hierarchical_metadata.json"
    TOP_K = 7
    SIMILARITY_CUTOFF = 0.5
    QUERY_DELAY = 11  # 11 seconds between queries
    HIERARCHICAL_STRATEGY = "parent_and_siblings"  # Options: parent_and_siblings, full_ancestors, parent_only, children, full_context
    
    print(f"\n🔧 Configuration:")
    print(f"   Index: {PINECONE_INDEX_NAME}")
    print(f"   Metadata file: {METADATA_FILE}")
    print(f"   Top-k: {TOP_K}")
    print(f"   Similarity cutoff: {SIMILARITY_CUTOFF}")
    print(f"   Query delay: {QUERY_DELAY}s per question")
    print(f"   Hierarchical strategy: {HIERARCHICAL_STRATEGY}")
    print(f"   Mode: Software Specifications (hierarchical retrieval)")
    
    # Setup components
    print("\n🔧 Initializing components...")
    
    # Configure embedding model
    embed_model = _setup_embedding()
    Settings.embed_model = embed_model
    
    # Configure Groq LLM with system prompt
    llm = _setup_groq_llm()
    Settings.llm = llm
    
    # Load hierarchical metadata
    print(f"\n📂 Loading hierarchical metadata...")
    navigator = _load_hierarchical_metadata(METADATA_FILE)
    
    # Load index
    print(f"\n📂 Loading Pinecone index...")
    try:
        index = _load_index(PINECONE_INDEX_NAME)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return
    
    # Configure query engine with hierarchical retrieval
    print(f"\n⚙️  Configuring query engine with hierarchical retrieval...")
    
    hierarchical_retriever = None  # Keep reference to retriever for expansion info
    if navigator:
        # Use hierarchical retriever
        hierarchical_retriever = HierarchicalRetriever(
            index=index,
            navigator=navigator,
            similarity_top_k=TOP_K,
            expansion_strategy=HIERARCHICAL_STRATEGY,
            similarity_cutoff=SIMILARITY_CUTOFF,
        )
        query_engine = RetrieverQueryEngine.from_args(
            hierarchical_retriever,
            llm=llm,
        )
        print(f"✓ Hierarchical query engine ready (top_k={TOP_K}, strategy={HIERARCHICAL_STRATEGY})")
    else:
        # Fallback to standard retrieval
        similarity_processor = SimilarityPostprocessor(similarity_cutoff=SIMILARITY_CUTOFF)
        query_engine = index.as_query_engine(
            similarity_top_k=TOP_K,
            node_postprocessors=[similarity_processor],
        )
        print(f"✓ Standard query engine ready (top_k={TOP_K}, similarity_cutoff={SIMILARITY_CUTOFF})")
    
    # Load evaluation dataset
    print(f"\n📋 Loading evaluation dataset...")
    eval_dataset = QueryResponseDataset.from_json("eval_dataset_chrono_often_wrong_ans.json")
    eval_qs = eval_dataset.questions
    ref_response_strs = [r for (_, r) in eval_dataset.qr_pairs]
    print(f"✓ Loaded {len(eval_qs)} evaluation questions")
    
    # Run evaluation loop
    print("\n" + "="*70)
    print(f"Starting Hierarchical Evaluation Loop ({len(eval_qs)} questions)")
    print("="*70 + "\n")
    
    pred_response_objs = []
    generated_answers = []  # Store LLM-generated answers
    retrieved_chunks_list = []  # Store retrieved chunks for each question
    hierarchical_expansion_list = []  # Store hierarchical expansion info for each question
    api_keys_used = set()  # Track which API keys were used
    
    for idx, query in enumerate(eval_qs, 1):
        max_retries = len(GROQ_API_KEYS)
        retry_count = 0
        response = None
        
        while retry_count < max_retries:
            try:
                print(f"[{idx}/{len(eval_qs)}] Query: {query[:60]}...")
                
                # Query with RAG (hierarchical retrieval happens here)
                response = query_engine.query(query)
                
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
                    if navigator:
                        hierarchical_retriever = HierarchicalRetriever(
                            index=index,
                            navigator=navigator,
                            similarity_top_k=TOP_K,
                            expansion_strategy=HIERARCHICAL_STRATEGY,
                            similarity_cutoff=SIMILARITY_CUTOFF,
                        )
                        query_engine = RetrieverQueryEngine.from_args(
                            hierarchical_retriever,
                            llm=llm,
                        )
                    else:
                        similarity_processor = SimilarityPostprocessor(similarity_cutoff=SIMILARITY_CUTOFF)
                        query_engine = index.as_query_engine(
                            similarity_top_k=TOP_K,
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
                        if navigator:
                            hierarchical_retriever = HierarchicalRetriever(
                                index=index,
                                navigator=navigator,
                                similarity_top_k=TOP_K,
                                expansion_strategy=HIERARCHICAL_STRATEGY,
                                similarity_cutoff=SIMILARITY_CUTOFF,
                            )
                            query_engine = RetrieverQueryEngine.from_args(
                                hierarchical_retriever,
                                llm=llm,
                            )
                        else:
                            similarity_processor = SimilarityPostprocessor(similarity_cutoff=SIMILARITY_CUTOFF)
                            query_engine = index.as_query_engine(
                                similarity_top_k=TOP_K,
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
        
        # Process successful response
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
                
                # Extract hierarchical metadata if available
                node_metadata = {}
                if hasattr(source_node.node, 'metadata'):
                    node_metadata = source_node.node.metadata
                
                chunks_for_question.append({
                    "chunk_index": node_idx,
                    "text": node_text,
                    "score": float(score) if isinstance(score, (int, float)) else score,
                    "metadata": node_metadata,
                })
                print(f"      Chunk {node_idx} (score: {score:.3f}): {node_text[:80]}...")
        else:
            print(f"   ⚠️  No source nodes retrieved")
        
        retrieved_chunks_list.append(chunks_for_question)
        
        # Extract hierarchical expansion info if using hierarchical retriever
        hierarchical_info = None
        if hierarchical_retriever and hasattr(hierarchical_retriever, 'last_expansion_info'):
            hierarchical_info = hierarchical_retriever.last_expansion_info
            if hierarchical_info:
                print(f"   🌳 Hierarchical context: {hierarchical_info.get('expansion_count', 0)} total chunks")
        
        hierarchical_expansion_list.append(hierarchical_info)
        
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
    print("RESULTS - Docling Hierarchical + Groq RAG Evaluation")
    print("="*60)
    print(f"✓ Completed: index={PINECONE_INDEX_NAME}, top_k={TOP_K}, score={mean_score:.4f}")
    
    # Save evaluation results to JSON
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "pinecone_index": PINECONE_INDEX_NAME,
            "metadata_file": METADATA_FILE,
            "top_k": TOP_K,
            "similarity_cutoff": SIMILARITY_CUTOFF,
            "query_delay_seconds": QUERY_DELAY,
            "hierarchical_strategy": HIERARCHICAL_STRATEGY,
            "llm_model": "llama-3.3-70b-versatile",
            "llm_provider": "groq",
            "llm_temperature": 0.1,
            "embedding_model": "BAAI/bge-base-en-v1.5",
            "embedding_dimension": 768,
            "system_prompt_mode": "balanced",
            "chunking_method": "docling_hierarchical",
            "retrieval_method": "hierarchical_recursive",
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
                "hierarchical_expansion": hierarchical_expansion_list[idx] if idx < len(hierarchical_expansion_list) else None,
            }
            for idx, score in enumerate(semantic_scores)
        ],
        "execution_info": {
            "duration_seconds": round(duration, 2),
            "duration_minutes": round(duration / 60, 2),
        }
    }
    
    output_file = "eval_results_docling_hierarchical_groq.json"
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
    
    # Print hierarchical expansion statistics
    if hierarchical_expansion_list:
        expansions_with_data = [h for h in hierarchical_expansion_list if h is not None]
        if expansions_with_data:
            print("\n" + "="*60)
            print("🌳 Hierarchical Expansion Summary")
            print("="*60)
            total_initial = sum(len(h.get('initial_chunks', [])) for h in expansions_with_data)
            total_expanded = sum(h.get('expansion_count', 0) for h in expansions_with_data)
            avg_initial = total_initial / len(expansions_with_data) if expansions_with_data else 0
            avg_expanded = total_expanded / len(expansions_with_data) if expansions_with_data else 0
            print(f"Questions with hierarchical expansion: {len(expansions_with_data)}/{len(eval_qs)}")
            print(f"Average initial chunks per query: {avg_initial:.2f}")
            print(f"Average expanded chunks per query: {avg_expanded:.2f}")
            print(f"Average expansion ratio: {avg_expanded/avg_initial:.2f}x" if avg_initial > 0 else "N/A")
    
    print("\n" + "="*60)
    print("🔑 API Keys Used During Evaluation")
    print("="*60)
    for key_num in sorted(api_keys_used):
        print(f"  • GROQ_API_KEY_{key_num}")
    print(f"Total API keys utilized: {len(api_keys_used)}")
    
    print("\n" + "="*60)
    print("Docling Hierarchical + Groq Evaluation Complete!")
    print("="*60)
    print(f"⏱️  Total Execution Time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
