"""
Hyperparameter Optimization for RAG
Based on: https://developers.llamaindex.ai/python/examples/param_optimizer/param_optimizer/
"""

import nest_asyncio

nest_asyncio.apply()

from pathlib import Path
from llama_index.readers.file import PDFReader  # type: ignore
from llama_index.readers.file import UnstructuredReader  # type: ignore
from llama_index.readers.file import PyMuPDFReader  # type: ignore

from llama_index.core import Document


from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import IndexNode
from llama_index.core.evaluation import QueryResponseDataset

from llama_index.core import (
    VectorStoreIndex,
    load_index_from_storage,
    StorageContext,
)
from llama_index.experimental.param_tuner import ParamTuner # type: ignore
from llama_index.experimental.param_tuner.base import RunResult  # type: ignore
from llama_index.core.evaluation.eval_utils import (
    get_responses,
    aget_responses,
)
from llama_index.core.evaluation import (
    SemanticSimilarityEvaluator,
    BatchEvalRunner,
)
from llama_index.llms.huggingface import HuggingFaceLLM  # type: ignore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore
from llama_index.experimental.param_tuner import RayTuneParamTuner  # type: ignore
from llama_index.core import Settings  # type: ignore
from llama_index.core.llms import MockLLM  # type: ignore


import os
import numpy as np
from pathlib import Path
import time


# Helper Functions

def _build_index(chunk_size, docs):
    index_out_path = f"./storage_{chunk_size}"
    if not os.path.exists(index_out_path):
        Path(index_out_path).mkdir(parents=True, exist_ok=True)
        # parse docs
        node_parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size)
        base_nodes = node_parser.get_nodes_from_documents(docs)

        # build index
        index = VectorStoreIndex(base_nodes)
        # save index to disk
        index.storage_context.persist(index_out_path)
    else:
        # rebuild storage context
        storage_context = StorageContext.from_defaults(
            persist_dir=index_out_path
        )
        # load index
        index = load_index_from_storage(
            storage_context,
        )
    return index


def _get_eval_batch_runner():
    # Use free HuggingFace embedding model instead of OpenAI
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    evaluator_s = SemanticSimilarityEvaluator(embed_model=embed_model)
    # Increased workers from 2 to 4 for faster parallel evaluation
    eval_batch_runner = BatchEvalRunner(
        {"semantic_similarity": evaluator_s}, workers=4, show_progress=False  # Disabled progress for speed
    )

    return eval_batch_runner


def objective_function(params_dict):
    chunk_size = params_dict["chunk_size"]
    docs = params_dict["docs"]
    top_k = params_dict["top_k"]
    eval_qs = params_dict["eval_qs"]
    ref_response_strs = params_dict["ref_response_strs"]

    # build index
    index = _build_index(chunk_size, docs)

    # query engine
    query_engine = index.as_query_engine(similarity_top_k=top_k)

    # get predicted responses - disabled progress bar for speed
    pred_response_objs = get_responses(
        eval_qs, query_engine, show_progress=False
    )

    # run evaluator
    # NOTE: can uncomment other evaluators
    eval_batch_runner = _get_eval_batch_runner()
    eval_results = eval_batch_runner.evaluate_responses(
        queries=eval_qs, responses=pred_response_objs, reference=ref_response_strs
    )

    # get semantic similarity metric: basic evaluation: could use other score like F1@k , MRR , ...
    # TODO: try another eval metrics
    # This metric 'semantic_similarity' seems to be an e2e evaluation score
    # A recommended apporach https://developers.llamaindex.ai/python/framework/optimizing/evaluation/evaluation/ is to start by an e2e 
    # eval and then , when identifying specific weaknesses/failures, to add more targeted/component specific evals / Component-Wise
    mean_score = np.array(
        [r.score for r in eval_results["semantic_similarity"]]
    ).mean()
    
    # Minimal logging for speed - only show final score
    print(f"[chunk_size={chunk_size}, top_k={top_k}] Mean score: {mean_score:.4f}")

    return RunResult(score=mean_score, params=params_dict)


def main():
    """Main function to run hyperparameter optimization."""
    # Start overall timing
    start_time_total = time.time()
    
    print("="*60)
    print("Starting RAG Hyperparameter Optimization")
    print("Using FREE HuggingFace Models (No API key needed!)")
    print("="*60)
    
    # Configure to use HuggingFace embedding model globally (lightweight)
    print("\nConfiguring HuggingFace embedding model...")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # Configure HuggingFace LLM - using a small, efficient model
    print("\nConfiguring HuggingFace LLM (this may take a moment to download)...")
    # Using a smaller model for faster inference: TinyLlama-1.1B
    # For better quality (but slower), you could use "HuggingFaceH4/zephyr-7b-beta"
    Settings.llm = HuggingFaceLLM(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        tokenizer_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        context_window=2048,# must be bigger n i encountered one with 2090, put it at max as possible
        max_new_tokens=128,  # Reduced from 256 to 128 for faster generation
        generate_kwargs={"temperature": 0.7, "do_sample": True, "top_p": 0.95},
        device_map="auto",
    )
    print("✓ Embedding model configured")
    print("✓ HuggingFace LLM configured (TinyLlama-1.1B)")
    
    # Load documents
    print("\nLoading documents...")
    loader = PDFReader()
    docs0 = loader.load_data(file=Path("./data/llama2.pdf"))
    doc_text = "\n\n".join([d.get_content() for d in docs0])
    docs = [Document(text=doc_text)]
    print(f"✓ Loaded {len(docs0)} pages from PDF")
    
    # Load evaluation dataset
    print("\nLoading evaluation dataset...")
    eval_dataset = QueryResponseDataset.from_json(
        "data/llama2_eval_qr_dataset.json"
    )
    eval_qs = eval_dataset.questions
    ref_response_strs = [r for (_, r) in eval_dataset.qr_pairs]
    print(f"✓ Loaded {len(eval_qs)} evaluation questions")
    
    # Define parameters - optimized for demo (1-2 min runtime)
    # Using: 1 chunk_size value × 2 top_k values = 2 combinations
    # Testing on only 3 question-answer pairs
    param_dict = {"chunk_size": [512], "top_k": [2, 5]}
    
    fixed_param_dict = {
        "docs": docs,
        "eval_qs": eval_qs[:3],  # Using first 3 for fast demo
        "ref_response_strs": ref_response_strs[:3],  # Fixed: must match eval_qs length
    }
    
    print(f"\nParameter combinations to test: {len(param_dict['chunk_size']) * len(param_dict['top_k'])}")
    print(f"Chunk sizes: {param_dict['chunk_size']}")
    print(f"Top-k values: {param_dict['top_k']}")
    print(f"Evaluation questions: {len(fixed_param_dict['eval_qs'])}")
    
    # Run ParamTuner
    print("\n" + "="*60)
    print("Running ParamTuner (Standard Grid Search)")
    print("="*60)
    
    # Start tuning timing
    start_time_tuning = time.time()
    
    param_tuner = ParamTuner(
        param_fn=objective_function,
        param_dict=param_dict,
        fixed_param_dict=fixed_param_dict,
        show_progress=False,  # Disabled for speed - using custom logging instead
    )
    
    results = param_tuner.tune()
    
    # End tuning timing
    tuning_duration = time.time() - start_time_tuning
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS - Standard ParamTuner")
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
    
    # Optional: Run RayTune ParamTuner
    print("\n" + "="*60)
    print("Running RayTuneParamTuner (Advanced)")
    print("="*60)
    try:
        param_tuner_ray = RayTuneParamTuner(
            param_fn=objective_function,
            param_dict=param_dict,
            fixed_param_dict=fixed_param_dict,
            run_config_dict={"storage_path": "/tmp/custom/ray_tune", "name": "my_exp"},
        )
        
        results_ray = param_tuner_ray.tune()
        
        print("\n" + "="*60)
        print("RESULTS - RayTuneParamTuner")
        print("="*60)
        best_result_ray = results_ray.best_run_result
        best_top_k_ray = results_ray.best_run_result.params["top_k"]
        best_chunk_size_ray = results_ray.best_run_result.params["chunk_size"]
        print(f"Best Score: {best_result_ray.score:.4f}")
        print(f"Best Top-k: {best_top_k_ray}")
        print(f"Best Chunk size: {best_chunk_size_ray}")
    except Exception as e:
        print(f"Note: RayTune optimization skipped: {e}")
    
    # End overall timing
    total_duration = time.time() - start_time_total
    
    print("\n" + "="*60)
    print("Optimization Complete!")
    print("="*60)
    print(f"⏱️  Total Execution Time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    print("="*60)


if __name__ == "__main__":
    main()