"""
RAG Semantic Chunking & Evaluation Tool
Loads source document with semantic chunking, creates Pinecone index,
and evaluates RAG performance with Groq LLM against eval dataset.
"""

import os
import sys
import time
import asyncio
import json
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
from typing import List

# Load environment variables
load_dotenv()

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    Document,
)
from llama_index.core.evaluation import QueryResponseDataset
from llama_index.vector_stores.pinecone import PineconeVectorStore  # type: ignore
from pinecone import Pinecone, ServerlessSpec  # type: ignore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore
from llama_index.llms.groq import Groq  # type: ignore
from llama_index.core.evaluation import (
    SemanticSimilarityEvaluator,
    BatchEvalRunner,
)
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.readers import SimpleDirectoryReader
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


def _load_source_document(doc_path: str) -> List[Document]:
    """Load source document from file path."""
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"Document not found: {doc_path}")
    
    print(f"📄 Loading document: {doc_path}")
    reader = SimpleDirectoryReader(input_files=[doc_path])
    documents = reader.load_data()
    print(f"✓ Loaded {len(documents)} document(s)")
    return documents


def _apply_semantic_chunking(documents: List[Document], embed_model, max_chunk_size: int = 2000) -> List:
    """Apply semantic chunking to documents using SemanticSplitterNodeParser with size threshold."""
    print(f"\n🔪 Applying semantic chunking to {len(documents)} document(s)...")
    
    # Initialize semantic splitter node parser
    splitter = SemanticSplitterNodeParser(
        buffer_size=1,  # number of sentences to overlap between chunks
        breakpoint_percentile_threshold=95,  # percentile for semantic similarity threshold
        embed_model=embed_model,
    )
    
    # Parse documents into semantic chunks
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"✓ Created {len(nodes)} semantic chunks (before size filtering)")
    
    # Apply size threshold to prevent oversized nodes
    print(f"   Applying size threshold: max_chunk_size={max_chunk_size} chars...")
    filtered_nodes = []
    oversized_count = 0
    
    for node in nodes:
        content = node.get_content()
        content_size = len(content)
        
        if content_size > max_chunk_size:
            oversized_count += 1
            # Split oversized nodes into smaller chunks
            sentences = content.split('. ')
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 2 <= max_chunk_size:
                    current_chunk += sentence + ". " if sentence else ""
                else:
                    if current_chunk.strip():
                        # Create new node for this chunk
                        new_node = node.copy()
                        new_node.set_content(current_chunk.strip())
                        filtered_nodes.append(new_node)
                    current_chunk = sentence + ". " if sentence else ""
            
            # Add remaining chunk
            if current_chunk.strip():
                new_node = node.copy()
                new_node.set_content(current_chunk.strip())
                filtered_nodes.append(new_node)
        else:
            filtered_nodes.append(node)
    
    print(f"✓ After size filtering: {len(filtered_nodes)} chunks (split {oversized_count} oversized nodes)")
    
    # Calculate statistics
    sizes = [len(node.get_content()) for node in filtered_nodes]
    avg_size = sum(sizes) / len(sizes) if sizes else 0
    max_size = max(sizes) if sizes else 0
    min_size = min(sizes) if sizes else 0
    
    print(f"\n📊 Chunk Size Statistics:")
    print(f"   Min size: {min_size} chars")
    print(f"   Avg size: {avg_size:.0f} chars")
    print(f"   Max size: {max_size} chars")
    
    # Print sample chunks for verification
    print(f"\n📊 Sample Chunks (first 3):")
    for idx, node in enumerate(filtered_nodes[:3]):
        content = node.get_content()[:150]
        content_size = len(node.get_content())
        print(f"   Chunk {idx+1} ({content_size} chars): {content}...")
    
    return filtered_nodes


def _check_index_populated(pc, index_name):
    """Check if a Pinecone index exists and has vectors."""
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        return False
    
    pinecone_index = pc.Index(index_name)
    stats = pinecone_index.describe_index_stats()
    total_vector_count = stats.get('total_vector_count', 0)
    
    return total_vector_count > 0


def _create_or_get_pinecone_index(pc, index_name: str, dimension: int = 768):
    """Create Pinecone index if it doesn't exist, or get existing index."""
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    
    if index_name in existing_indexes:
        print(f"✓ Using existing Pinecone index: {index_name}")
        return pc.Index(index_name)
    
    print(f"📌 Creating new Pinecone index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"✓ Pinecone index created: {index_name}")
    return pc.Index(index_name)


def _index_semantic_chunks(nodes, index_name: str, embed_model, llm):
    """Create and populate Pinecone index with semantic chunks using direct embeddings."""
    print(f"\n🗂️  Indexing {len(nodes)} semantic chunks into Pinecone...")
    
    pc = _initialize_pinecone()
    pinecone_index = _create_or_get_pinecone_index(pc, index_name)
    
    # Generate embeddings and prepare vectors directly
    print(f"   Generating embeddings for {len(nodes)} chunks...")
    vectors_to_upsert = []
    
    for idx, node in enumerate(nodes):
        try:
            # Get the text content
            text_content = node.get_content()
            text_size = len(text_content)
            
            # Generate embedding
            embedding = embed_model.get_text_embedding(text_content)
            
            # Create minimal metadata with only essential info
            # Keep metadata very small to avoid exceeding 40KB limit
            minimal_metadata = {
                "text": text_content[:250],  # Store only first 250 chars in metadata
                "size": text_size,  # Store original size
            }
            
            # Create vector tuple (id, embedding, metadata)
            vector_id = f"chunk_{idx}"
            vectors_to_upsert.append((vector_id, embedding, minimal_metadata))
            
            if (idx + 1) % 10 == 0:
                print(f"   Generated embeddings: {idx + 1}/{len(nodes)}")
        
        except Exception as e:
            print(f"   ⚠️  Error processing node {idx}: {str(e)[:100]}")
            continue
    
    # Upsert vectors in batches to Pinecone
    print(f"\n   Upserting {len(vectors_to_upsert)} vectors to Pinecone...")
    batch_size = 25  # Reduced batch size to be more conservative
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i+batch_size]
        try:
            pinecone_index.upsert(vectors=batch)
            print(f"   Upserted batch {i//batch_size + 1}/{(len(vectors_to_upsert)-1)//batch_size + 1}")
        except Exception as e:
            print(f"   ⚠️  Error upserting batch: {str(e)[:150]}")
            raise
    
    # Create a VectorStoreIndex pointing to the Pinecone index
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create index object for querying (using existing nodes but they won't be re-upserted)
    index = VectorStoreIndex(
        nodes=[],  # Empty nodes since we already upserted
        storage_context=storage_context,
        embed_model=embed_model,
    )
    
    print(f"✓ Index created and populated with {len(vectors_to_upsert)} chunks")
    return index


def _setup_groq_llm():
    """Configure Groq LLM with high-quality model."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    
    llm = Groq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
    )
    print("✓ Groq LLM configured (llama-3.3-70b-versatile)")
    return llm


def _setup_embedding(model_name: str = "BAAI/bge-base-en-v1.5"):
    """Configure HuggingFace embedding model."""
    embed_model = HuggingFaceEmbedding(
        model_name=model_name,
        device="cpu",
        trust_remote_code=True,
    )
    print(f"✓ HuggingFace embedding model configured ({model_name})")
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
    """Main async function to load, chunk, index, and evaluate RAG with Groq LLM."""
    start_time = time.time()
    
    print("="*80)
    print("Semantic Chunking & RAG Evaluation - Groq LLM + Pinecone")
    print("="*80)
    
    # Configuration
    SOURCE_DOC_PATH = "Spec détaillées - Middleware XL EDS (ATS __ XL EDS __ CHRONOPOST).docx"
    PINECONE_INDEX_NAME = "ats-chrono-semantic-chunking-index"
    TOP_K = 5
    SIMILARITY_CUTOFF = 0.5
    QUERY_DELAY = 15  # 15 seconds between queries
    
    print(f"\n🔧 Configuration:")
    print(f"   Source Document: {SOURCE_DOC_PATH}")
    print(f"   Pinecone Index: {PINECONE_INDEX_NAME}")
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
    
    # Step 1: Load source document
    print(f"\n📂 Step 1: Loading source document...")
    try:
        documents = _load_source_document(SOURCE_DOC_PATH)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"   Please ensure '{SOURCE_DOC_PATH}' exists in the project directory")
        return
    
    # Step 2: Apply semantic chunking
    print(f"\n🔪 Step 2: Applying semantic chunking...")
    nodes = _apply_semantic_chunking(documents, embed_model, max_chunk_size=2000)
    
    # Step 3: Create Pinecone index and populate with semantic chunks
    print(f"\n📌 Step 3: Creating Pinecone index and indexing chunks...")
    index = _index_semantic_chunks(nodes, PINECONE_INDEX_NAME, embed_model, llm)
    
    # Save nodes to storage context for later retrieval
    print(f"   Saving {len(nodes)} nodes to storage context...")
    for node in nodes:
        index.storage_context.docstore.add_documents([node])
    
    # Step 4: Configure query engine with similarity filtering
    print(f"\n⚙️  Step 4: Configuring query engine...")
    similarity_processor = SimilarityPostprocessor(similarity_cutoff=SIMILARITY_CUTOFF)
    query_engine = index.as_query_engine(
        similarity_top_k=TOP_K,
        node_postprocessors=[similarity_processor],
    )
    print(f"✓ Query engine ready (top_k={TOP_K}, similarity_cutoff={SIMILARITY_CUTOFF})")
    
    # Step 5: Load evaluation dataset
    print(f"\n📋 Step 5: Loading evaluation dataset...")
    eval_dataset = QueryResponseDataset.from_json("eval_dataset_chrono.json")
    eval_qs = eval_dataset.questions
    ref_response_strs = [r for (_, r) in eval_dataset.qr_pairs]
    print(f"✓ Loaded {len(eval_qs)} evaluation questions")
    
    # Step 6: Run evaluation loop
    print("\n" + "="*80)
    print(f"Step 6: Starting Evaluation Loop ({len(eval_qs)} questions)")
    print("="*80 + "\n")
    
    pred_response_objs = []
    
    for idx, query in enumerate(eval_qs, 1):
        try:
            print(f"[{idx}/{len(eval_qs)}] Query: {query[:60]}...")
            
            # Query with RAG
            response = query_engine.query(query)
            pred_response_objs.append(response)
            
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
    
    # Step 7: Run semantic similarity evaluation
    print("\n" + "="*80)
    print("Step 7: Running Semantic Similarity Evaluation")
    print("="*80 + "\n")
    
    eval_batch_runner = _get_eval_batch_runner()
    eval_results = await eval_batch_runner.aevaluate_responses(
        eval_qs, responses=pred_response_objs, reference=ref_response_strs
    )
    
    # Debug: Print evaluation results structure
    print(f"📊 Evaluation Results Structure:")
    print(f"   Keys in eval_results: {list(eval_results.keys())}")
    for key, value in eval_results.items():
        print(f"   - {key}: type={type(value)}, length={len(value) if hasattr(value, '__len__') else 'N/A'}")
        if hasattr(value, '__iter__') and len(value) > 0:
            first_item = value[0] if isinstance(value, list) else next(iter(value))
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
    
    # Display results
    print("\n" + "="*80)
    print("RESULTS - Semantic Chunking & RAG Evaluation")
    print("="*80)
    print(f"✓ Completed: semantic_chunks={len(nodes)}, top_k={TOP_K}, score={mean_score:.4f}")
    
    # Save evaluation results to JSON
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "source_document": SOURCE_DOC_PATH,
            "semantic_chunks_count": len(nodes),
            "pinecone_index": PINECONE_INDEX_NAME,
            "top_k": TOP_K,
            "similarity_cutoff": SIMILARITY_CUTOFF,
            "query_delay_seconds": QUERY_DELAY,
            "llm_model": "llama-3.3-70b-versatile",
            "llm_provider": "groq",
            "embedding_model": "BAAI/bge-base-en-v1.5",
            "chunking_strategy": "semantic",
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
                "score": float(score),
            }
            for idx, score in enumerate(semantic_scores)
        ],
        "execution_info": {
            "duration_seconds": round(duration, 2),
            "duration_minutes": round(duration / 60, 2),
        }
    }
    
    output_file = "eval_results_semantic_chunking.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results saved to {output_file}")
    
    print("\n" + "="*80)
    print("Semantic Chunking & Evaluation Complete!")
    print("="*80)
    print(f"⏱️  Total Execution Time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
