"""
Async Hyperparameter Optimization for RAG
"""

import asyncio
import nest_asyncio
import os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import time

nest_asyncio.apply()

# Load environment variables
load_dotenv()

from llama_index.readers.file import PDFReader  # type: ignore
from llama_index.core import Document
from llama_index.core.evaluation import QueryResponseDataset
from llama_index.core import (
    VectorStoreIndex,
    load_index_from_storage,
    StorageContext,
)
from llama_index.vector_stores.pinecone import PineconeVectorStore  # type: ignore
from pinecone import Pinecone, ServerlessSpec  # type: ignore
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.experimental.param_tuner import AsyncParamTuner  # type: ignore
from llama_index.experimental.param_tuner.base import RunResult  # type: ignore
from llama_index.core.evaluation.eval_utils import aget_responses
from llama_index.core.evaluation import (
    SemanticSimilarityEvaluator,
    BatchEvalRunner,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore
from llama_index.llms.huggingface import HuggingFaceLLM  # type: ignore
from llama_index.core import Settings  # type: ignore


# Helper Functions

def _get_pinecone_index_name(chunk_size):
    """Get Pinecone index name based on chunk size."""
    base_name = os.getenv("PINECONE_INDEX_BASE_NAME", "rag")
    return f"{base_name}-chunk-{chunk_size}"


def _initialize_pinecone():
    """Initialize Pinecone client."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in environment variables")
    
    pc = Pinecone(api_key=api_key)
    return pc


def _check_index_populated(pc, index_name):
    """Check if a Pinecone index exists and has vectors."""
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        return False
    
    # Get index stats to check if it has vectors
    pinecone_index = pc.Index(index_name)
    stats = pinecone_index.describe_index_stats()
    total_vector_count = stats.get('total_vector_count', 0)
    
    return total_vector_count > 0


def _build_index(chunk_size, docs):
    """Build or load index with Pinecone vector store."""
    # Initialize Pinecone
    pc = _initialize_pinecone()
    index_name = _get_pinecone_index_name(chunk_size)
    
    # Check if index exists and is populated
    index_populated = _check_index_populated(pc, index_name)
    
    if index_populated:
        print(f"✓ Using existing populated Pinecone index: {index_name}")
        # Just connect to existing index without re-indexing
        pinecone_index = pc.Index(index_name)
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create index from existing vector store (no documents needed)
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context
        )
    else:
        # Index doesn't exist or is empty - need to create/populate it
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
        
        if index_name not in existing_indexes:
            print(f"Creating Pinecone index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=384,  # BAAI/bge-base-en-v1.5 produces 384-dimensional embeddings
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            print(f"✓ Created index: {index_name}")
        else:
            print(f"Index {index_name} exists but is empty. Populating...")
        
        # Initialize Pinecone vector store
        pinecone_index = pc.Index(index_name)
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        
        # Create storage context with Pinecone
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Parse docs and build index
        print(f"Chunking documents with chunk_size={chunk_size}...")
        node_parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size)
        base_nodes = node_parser.get_nodes_from_documents(docs)
        print(f"✓ Created {len(base_nodes)} chunks")
        
        # Build index with Pinecone (this will embed and store vectors)
        print(f"Embedding and indexing chunks...")
        index = VectorStoreIndex(base_nodes, storage_context=storage_context)
        print(f"✓ Indexed {len(base_nodes)} chunks into Pinecone")
    
    return index


def _get_eval_batch_runner():
    """Get evaluation batch runner."""
    # Use free HuggingFace embedding model instead of OpenAI
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    evaluator_s = SemanticSimilarityEvaluator(embed_model=embed_model)
    eval_batch_runner = BatchEvalRunner(
        {"semantic_similarity": evaluator_s}, workers=2, show_progress=True
    )
    return eval_batch_runner


async def aobjective_function(params_dict):
    """Async objective function for hyperparameter optimization."""
    chunk_size = params_dict["chunk_size"]
    docs = params_dict["docs"]
    top_k = params_dict["top_k"]
    eval_qs = params_dict["eval_qs"]
    ref_response_strs = params_dict["ref_response_strs"]

    # build index
    index = _build_index(chunk_size, docs)

    # query engine
    query_engine = index.as_query_engine(similarity_top_k=top_k)

    # get predicted responses
    pred_response_objs = await aget_responses(
        eval_qs, query_engine, show_progress=True
    )

    # run evaluator
    eval_batch_runner = _get_eval_batch_runner()
    eval_results = await eval_batch_runner.aevaluate_responses(
        eval_qs, responses=pred_response_objs, reference=ref_response_strs  # type: ignore
    )

    # get semantic similarity metric
    mean_score = np.array(
        [r.score for r in eval_results["semantic_similarity"]]
    ).mean()

    return RunResult(score=mean_score, params=params_dict)


async def main():
    """Main async function to run hyperparameter optimization."""
    # Start overall timing
    start_time_total = time.time()
    
    print("="*60)
    print("Starting ASYNC RAG Hyperparameter Optimization")
    print("Using FREE HuggingFace Embeddings (No API key needed!)")
    print("="*60)
    
    # Configure to use HuggingFace embedding model globally
    print("\nConfiguring HuggingFace embedding model...")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    print("✓ Embedding model configured")
    
    # Configure HuggingFace LLM - using a small, efficient model
    print("\nConfiguring HuggingFace LLM (this may take a moment to download)...")
    Settings.llm = HuggingFaceLLM(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        tokenizer_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        context_window=2048,
        max_new_tokens=128,
        generate_kwargs={"temperature": 0.7, "do_sample": True, "top_p": 0.95},
        device_map="auto",
    )
    print("✓ HuggingFace LLM configured (TinyLlama-1.1B)")
    
    # Define parameters first to check indexes
    param_dict = {"chunk_size": [256, 512], "top_k": [2, 5]}
    # For quick testing, uncomment below:
    # param_dict = {"chunk_size": [256], "top_k": [1]}
    
    # Check which indexes are already populated
    print("\nChecking Pinecone indexes...")
    pc = _initialize_pinecone()
    indexes_status = {}
    for chunk_size in param_dict["chunk_size"]:
        index_name = _get_pinecone_index_name(chunk_size)
        is_populated = _check_index_populated(pc, index_name)
        indexes_status[chunk_size] = is_populated
        status = "✓ Populated" if is_populated else "✗ Empty/Missing"
        print(f"  {index_name}: {status}")
    
    # Load documents only if needed (at least one index is not populated)
    need_docs = not all(indexes_status.values())
    
    if need_docs:
        print("\n📄 Loading documents (needed for indexing)...")
        loader = PDFReader()
        docs0 = loader.load_data(file=Path("./data/llama2.pdf"))
        doc_text = "\n\n".join([d.get_content() for d in docs0])
        docs = [Document(text=doc_text)]
        print(f"✓ Loaded {len(docs0)} pages from PDF")
    else:
        print("\n✓ All indexes are populated - skipping document loading")
        docs = []  # Empty list since we don't need docs
    
    # Load evaluation dataset
    print("\nLoading evaluation dataset...")
    eval_dataset = QueryResponseDataset.from_json(
        "data/llama2_eval_qr_dataset.json"
    )
    eval_qs = eval_dataset.questions
    ref_response_strs = [r for (_, r) in eval_dataset.qr_pairs]
    print(f"✓ Loaded {len(eval_qs)} evaluation questions")
    
    fixed_param_dict = {
        "docs": docs,
        "eval_qs": eval_qs[:5],  # Using first 5 for speed
        "ref_response_strs": ref_response_strs[:5],
    }
    
    print(f"\nParameter combinations to test: {len(param_dict['chunk_size']) * len(param_dict['top_k'])}")
    print(f"Chunk sizes: {param_dict['chunk_size']}")
    print(f"Top-k values: {param_dict['top_k']}")
    
    # Run AsyncParamTuner
    print("\n" + "="*60)
    print("Running AsyncParamTuner (Async Grid Search)")
    print("="*60)
    
    # Start tuning timing
    start_time_tuning = time.time()
    
    aparam_tuner = AsyncParamTuner(
        aparam_fn=aobjective_function,
        param_dict=param_dict,
        fixed_param_dict=fixed_param_dict,
        num_workers=2,
        show_progress=True,
    )

    results = await aparam_tuner.atune()
    
    # End tuning timing
    tuning_duration = time.time() - start_time_tuning

    # Display results
    print("\n" + "="*60)
    print("RESULTS - AsyncParamTuner")
    print("="*60)
    print(f"⏱️  Tuning Duration: {tuning_duration:.2f} seconds ({tuning_duration/60:.2f} minutes)")
    best_result = results.best_run_result
    best_top_k = results.best_run_result.params["top_k"]
    best_chunk_size = results.best_run_result.params["chunk_size"]
    print(f"Best Score: {best_result.score:.4f}")
    print(f"Best Top-k: {best_top_k}")
    print(f"Best Chunk size: {best_chunk_size}")
    
    # Show all results
    print("\nAll Results:")
    for idx, run_result in enumerate(results.run_results):
        print(f"  Run {idx}: score={run_result.score:.4f}, "
              f"chunk_size={run_result.params['chunk_size']}, "
              f"top_k={run_result.params['top_k']}")
    
    # End overall timing
    total_duration = time.time() - start_time_total
    
    print("\n" + "="*60)
    print("Async Optimization Complete!")
    print("="*60)
    print(f"⏱️  Total Execution Time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())


