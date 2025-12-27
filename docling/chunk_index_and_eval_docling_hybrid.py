"""
Docling Hybrid Chunker with Metadata Enrichment - Index and Evaluate
================================================================================

This script:
1. Loads markdown document using Docling
2. Chunks with HybridChunker (tokenization-aware)
3. Enriches chunks with QuestionsAnsweredExtractor
4. Creates LlamaIndex nodes with contextual metadata
5. Indexes to Pinecone
6. Runs RAG evaluation with Groq LLM

Features:
- Docling HybridChunker for semantic + token-aware chunking
- Question generation for enhanced retrieval
- Contextual metadata enrichment
- Reranking pipeline (retrieve → rerank → synthesize)
- Strict accuracy mode for software specifications

Usage:
    python chunk_index_and_eval_docling_hybrid.py --mode index
    python chunk_index_and_eval_docling_hybrid.py --mode eval
    python chunk_index_and_eval_docling_hybrid.py --mode all
"""

import os
import sys
import time
import asyncio
import json
import argparse
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
from typing import Any, List, Dict
import openai
import numpy as np
import re
import pprint

load_dotenv()

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    Document,
)
from llama_index.core.schema import TextNode, MetadataMode
from llama_index.core.extractors import QuestionsAnsweredExtractor
from llama_index.core.prompts import PromptTemplate
from llama_index.core.evaluation import (
    QueryResponseDataset,
    SemanticSimilarityEvaluator,
    BatchEvalRunner,
)
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import CrossEncoder

# Docling imports
try:
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
    from docling_core.transforms.chunker.hierarchical_chunker import ChunkingSerializerProvider
    from docling_core.transforms.serializer.markdown import MarkdownDocSerializer
    from docling.document_converter import DocumentConverter
    from transformers import AutoTokenizer
    HAS_DOCLING = True
except ImportError as e:
    print(f"❌ Docling import error: {e}")
    print("Install with: pip install 'docling-core[chunking]' docling transformers")
    HAS_DOCLING = False
    sys.exit(1)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Index Configuration
PINECONE_INDEX_NAME = "docling-hybrid-enriched-ats-chrono"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIMENSION = 768
MAX_TOKENS_PER_CHUNK = 512
MERGE_PEERS = True

# Document paths
ROOT_DIR = Path(__file__).parent.parent
SOURCE_DOCUMENT = ROOT_DIR / "markitdown" / "chrono-specs-markitdown.md"
EVAL_DATASET_PATH = ROOT_DIR / "eval_dataset_chrono.json"
OUTPUT_DIR = Path(__file__).parent
OUTPUT_CHUNKS_FILE = OUTPUT_DIR / "chunks_with_metadata.json"
OUTPUT_NODES_FILE = OUTPUT_DIR / "nodes_with_metadata.json"  # Persisted enriched nodes
OUTPUT_EVAL_FILE = OUTPUT_DIR / "eval_results_docling_hybrid.json"

# Question extraction
NUM_QUESTIONS = 3

# LLM Configuration
# LLM_MODEL = "xiaomi/mimo-v2-flash:free"  # Free model on OpenRouter
LLM_MODEL = "x-ai/grok-4.1-fast"

# Evaluation Configuration
INITIAL_TOP_K = 25
RERANK_TOP_N = 5
RERANKER_MODEL = "BAAI/bge-reranker-large"
SIMILARITY_CUTOFF = 0.55
QUERY_DELAY = 2

# OpenRouter API Key
OPENROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
print(f"Using OPEN_ROUTER_API_KEY: {OPENROUTER_API_KEY}")
if not OPENROUTER_API_KEY:
    raise ValueError("OPEN_ROUTER_API_KEY not found in environment variables")

# System Prompt
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


# ============================================================================
# PERSISTENCE FUNCTIONS
# ============================================================================

def save_nodes_to_json(nodes: List[Document], output_path: Path):
    """Save enriched nodes with all metadata to JSON for persistence."""
    nodes_data = []
    for node in nodes:
        node_dict = {
            "text": node.text,
            "metadata": node.metadata,
            "excluded_embed_metadata_keys": node.excluded_embed_metadata_keys,
            "excluded_llm_metadata_keys": node.excluded_llm_metadata_keys,
            "metadata_separator": node.metadata_separator,
            "metadata_template": node.metadata_template,
            "text_template": node.text_template,
        }
        nodes_data.append(node_dict)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nodes_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved {len(nodes_data)} enriched nodes to {output_path}")


def load_nodes_from_json(input_path: Path) -> List[Document]:
    """Load enriched nodes from JSON persistence file."""
    with open(input_path, "r", encoding="utf-8") as f:
        nodes_data = json.load(f)
    
    nodes = []
    i=0
    for node_dict in nodes_data:
        doc = Document(
            text=node_dict["text"],
            metadata=node_dict["metadata"],
            excluded_embed_metadata_keys=[],
            excluded_llm_metadata_keys=["questions_this_excerpt_can_answer"],
            metadata_separator=node_dict.get("metadata_separator", "\n"),
            metadata_template=node_dict.get("metadata_template", "{key}: {value}"),
            text_template="<metadata>{metadata_str}</metadata>\n-----\nContent:\n{content}",
        )
        doc.metadata["questions_this_excerpt_can_answer"] = node_dict.get("questions_this_excerpt_can_answer", "")
        doc.metadata["questions_this_excerpt_can_answer"] = "\n".join(doc.metadata["questions_this_excerpt_can_answer"]) if isinstance(doc.metadata["questions_this_excerpt_can_answer"], list) else doc.metadata["questions_this_excerpt_can_answer"]
        doc.metadata["headings"] = "\n".join(doc.metadata["headings"]) if isinstance(doc.metadata.get("headings", ""), list) else doc.metadata.get("headings", "")
        if i==61:
            print("full chunk tetx representatin n(with meta) to be passed to embedder", doc.get_content(metadata_mode=MetadataMode.EMBED))
        nodes.append(doc)
        i+=1
    
    print(f"✓ Loaded {len(nodes)} enriched nodes from {input_path}")
    return nodes


def load_chunks_from_json_for_indexing(json_path: Path) -> List[Document]:
    """
    Load chunks from chunks_with_metadata.json and prepare for Pinecone indexing.
    
    This function:
    1. Loads chunks from JSON
    2. Converts array fields (questions, headings) to strings for Pinecone
    3. Ensures questions are NOT excluded from embedding (for better retrieval)
    4. Creates Document objects ready for indexing
    """
    print(f"\n{'='*70}")
    print("LOADING CHUNKS FROM JSON FOR INDEXING")
    print(f"{'='*70}")
    print(f"Source: {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    
    print(f"✓ Loaded {len(chunks_data)} chunks from JSON")
    
    nodes = []
    i = 0
    for chunk in chunks_data:
        # Prepare metadata with string conversions
        metadata = {
            "chunk_index": chunk["metadata"].get("chunk_index", chunk.get("chunk_id", 0))
        }
        
        # Convert headings array to string if present
        if "headings" in chunk["metadata"] and chunk["metadata"]["headings"]:
            headings = chunk["metadata"]["headings"]
            if isinstance(headings, list):
                metadata["headings"] = " > ".join(str(h) for h in headings)
            else:
                metadata["headings"] = str(headings)
        
        # Convert captions array to string if present
        if "captions" in chunk["metadata"] and chunk["metadata"]["captions"]:
            captions = chunk["metadata"]["captions"]
            if isinstance(captions, list):
                metadata["captions"] = " | ".join(str(c) for c in captions)
            else:
                metadata["captions"] = str(captions)
        
        # Convert questions array to string if present
        # IMPORTANT: Questions should be included in embedding for better retrieval!
        if "questions_this_excerpt_can_answer" in chunk and chunk["questions_this_excerpt_can_answer"]:
            questions = chunk["questions_this_excerpt_can_answer"]
            if isinstance(questions, list):
                # Join with newlines for better readability
                metadata["questions_this_excerpt_can_answer"] = "\n".join(str(q) for q in questions)
            else:
                metadata["questions_this_excerpt_can_answer"] = str(questions)
        
        # Create Document with questions NOT excluded from embedding
        doc = Document(
            text=chunk["text"],
            metadata=metadata,
            metadata_separator="\n",
            metadata_template="{key}: {value}",
            text_template="<metadata>{metadata_str}</metadata>\n-----\nContent:\n{content}",
            excluded_embed_metadata_keys=["chunk_index"],  # Only exclude chunk_index
            excluded_llm_metadata_keys=["chunk_index"],     # Only exclude chunk_index
        )
        nodes.append(doc)
        if i==61:
            print("full chunk tetx representatin n(with meta) to be passed to embedder", doc.get_content(metadata_mode=MetadataMode.EMBED))
    
    print(f"✓ Converted {len(nodes)} chunks to Documents")
    print(f"  - Questions ARE included in embeddings for better retrieval")
    print(f"  - Headings converted to strings (separator: ' > ')")
    print(f"  - Captions converted to strings (separator: ' | ')")
    print(f"  - Only 'chunk_index' excluded from embeddings")
    
    return nodes


# ============================================================================
# MARKDOWN SERIALIZER
# ============================================================================

class MDSerializerProvider(ChunkingSerializerProvider):
    """Markdown serializer for Docling chunking."""
    def get_serializer(self, doc):
        return MarkdownDocSerializer(doc=doc)


# ============================================================================
# HELPER FUNCTIONS - LLMS
# ============================================================================

def _initialize_question_extraction_llm():
    """Initialize OpenRouter LLM for question extraction."""
    api_key = os.getenv("OPEN_ROUTER_API_KEY")
    if not api_key:
        print("⚠️  OPEN_ROUTER_API_KEY not found. Question extraction will be skipped.")
        return None
    
    llm = OpenAILike(
        model=LLM_MODEL,
        api_base="https://openrouter.ai/api/v1",
        api_key=api_key,
        max_retries=3,
        timeout=60.0,
    )
    print(f"✓ OpenRouter LLM initialized for question extraction ({LLM_MODEL})")
    return llm


def _initialize_qa_extractor(llm, num_questions: int = 3) -> QuestionsAnsweredExtractor:
    """Initialize QuestionsAnsweredExtractor with the given LLM."""
    return QuestionsAnsweredExtractor(
        questions=num_questions,
        llm=llm,
        embedding_only=False,
        show_progress=True,
    )


def _extract_questions_sync(nodes: List[Document], extractor: QuestionsAnsweredExtractor):
    """Extract questions synchronously using manual LLM calls.
    
    Note: Uses manual LLM calls instead of extractor's aextract() method
    due to issues with Document nodes in certain contexts.
    """
    prompt_template = extractor.prompt_template
    prompt = PromptTemplate(template=prompt_template)
    
    # Extract questions for each node using manual LLM calls
    for i, node in enumerate(nodes):
        try:
            context_str = node.get_content(metadata_mode=MetadataMode.LLM)
            
            # Call LLM with the question generation prompt
            questions_response = extractor.llm.predict(
                prompt, num_questions=extractor.questions, context_str=context_str
            )
            
            # Add to node metadata
            node.metadata['questions_this_excerpt_can_answer'] = questions_response.strip()
            
        except Exception as e:
            print(f"    ❌ Node {i+1}: Error extracting questions - {e}")
            node.metadata['questions_this_excerpt_can_answer'] = ""
    
    return nodes


def _setup_openrouter_llm():
    """Configure OpenRouter LLM with system prompt for software specifications."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPEN_ROUTER_API_KEY not found in environment variables")
    
    llm = OpenAILike(
        model=LLM_MODEL,
        api_base="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        temperature=0.1,
        max_retries=3,
        timeout=60.0,
        system_prompt=SYSTEM_PROMPT,
    )
    print(f"✓ OpenRouter LLM configured (gemini-2.0-flash-exp:free)")
    return llm


def _is_rate_limit_error(error):
    """Check if the error is a rate limit error."""
    error_type = type(error).__name__
    if error_type == "RateLimitError":
        return True
    error_str = str(error).lower()
    return any(phrase in error_str for phrase in [
        "rate limit", "rate_limit", "too many requests", "quota exceeded",
        "429", "limit exceeded", "rate_limit_exceeded", "tokens per day"
    ])


# ============================================================================
# CHUNKING FUNCTIONS
# ============================================================================

def chunk_markdown_with_docling_hybrid(file_path: Path, llm=None) -> tuple[List[dict], List[Document]]:
    """Chunk markdown document using Docling's HybridChunker with metadata enrichment."""
    
    print(f"\n{'='*70}")
    print("CHUNKING WITH DOCLING HYBRIDCHUNKER")
    print(f"{'='*70}")
    print(f"Source: {file_path}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Max tokens per chunk: {MAX_TOKENS_PER_CHUNK}")
    print(f"Merge peers: {MERGE_PEERS}")
    
    # Step 1: Convert markdown to DoclingDocument
    print(f"\n📄 Converting markdown to DoclingDocument...")
    converter = DocumentConverter()
    conv_result = converter.convert(
        source=file_path,
        headers={"type": "markdown", "encoding": "utf-8"},
    )
    dl_doc = conv_result.document
    print(f"✓ Document converted")
    
    # Step 2: Initialize tokenizer
    print(f"\n🔧 Loading tokenizer: {EMBEDDING_MODEL}...")
    hf_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    tokenizer = HuggingFaceTokenizer(
        tokenizer=hf_tokenizer,
        max_tokens=MAX_TOKENS_PER_CHUNK,
    )
    print(f"✓ Tokenizer loaded")
    
    # Step 3: Initialize HybridChunker
    print(f"\n✂️  Initializing HybridChunker...")
    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=MERGE_PEERS,
        serializer_provider=MDSerializerProvider(),
    )
    print(f"✓ HybridChunker initialized")
    
    # Step 4: Chunk the document
    print(f"\n⚙️  Chunking document...")
    chunk_iter = chunker.chunk(dl_doc=dl_doc)
    chunks = list(chunk_iter)
    print(f"✓ Created {len(chunks)} chunks")
    
    # Step 5: Convert chunks to Documents with metadata
    print(f"\n📦 Converting chunks to Documents...")
    nodes = []
    output_chunks = []
    
    for i, chunk in enumerate(chunks):
        enriched_text = chunker.contextualize(chunk=chunk)
        dumped_meta = chunk.meta.model_dump()
        
        # Relevant metadata for retrieval (filter out None values for Pinecone)
        relevant_metadata = {
            "chunk_index": i,
        }
        
        # Only add headings if not None/empty
        if dumped_meta["headings"]:
            relevant_metadata["headings"] = dumped_meta["headings"]
        
        # Only add captions if not None/empty
        if dumped_meta["captions"]:
            relevant_metadata["captions"] = dumped_meta["captions"]
        
        doc = Document(
            text=chunk.text,
            metadata=relevant_metadata,
            metadata_seperator="\n",
            metadata_template="{key}: {value}",
            text_template="<metadata>{metadata_str}</metadata>\n-----\nContent:\n{content}",
            excluded_embed_metadata_keys=["chunk_index"],
            excluded_llm_metadata_keys=["chunk_index"],
        )
        nodes.append(doc)
        
        chunk_dict = {
            "chunk_id": i,
            "text": chunk.text,
            "enriched_text": enriched_text,
            "char_count": len(enriched_text),
            "token_count": len(hf_tokenizer.encode(enriched_text)),
            "metadata": relevant_metadata,
        }
        output_chunks.append(chunk_dict)
    
    print(f"✓ Converted {len(nodes)} chunks to Documents")
    
    # Step 6: Extract questions if LLM available
    if llm and nodes:
        print(f"\n🤖 Extracting questions using QuestionsAnsweredExtractor...")
        try:
            qa_extractor = _initialize_qa_extractor(llm, NUM_QUESTIONS)
            extracted_nodes = _extract_questions_sync(nodes, qa_extractor)
            
            # Update output_chunks with questions
            for i, node in enumerate(extracted_nodes):
                if i < len(output_chunks):
                    questions_str = node.metadata.get('questions_this_excerpt_can_answer', '')
                    questions = [q.strip() for q in questions_str.split('\n') if q.strip()] if questions_str else []
                    output_chunks[i]['questions_this_excerpt_can_answer'] = questions
                    if questions:
                        print(f"  ✅ Chunk {i+1}: Extracted {len(questions)} questions")
            
            nodes = extracted_nodes
            print(f"✓ Question extraction complete")
            
        except Exception as e:
            print(f"❌ Error during question extraction: {e}")
            import traceback
            traceback.print_exc()
            for chunk_dict in output_chunks:
                chunk_dict['questions_this_excerpt_can_answer'] = []
    else:
        for chunk_dict in output_chunks:
            chunk_dict['questions_this_excerpt_can_answer'] = []
    
    return output_chunks, nodes


# ============================================================================
# PINECONE FUNCTIONS
# ============================================================================

def _initialize_pinecone():
    """Initialize Pinecone client."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in environment variables")
    
    pc = Pinecone(api_key=api_key, pool_threads=4, timeout=30)
    return pc


def _create_or_get_pinecone_index(pc, index_name: str):
    """Create Pinecone index if it doesn't exist."""
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f"\n📊 Creating Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"✓ Index created")
    else:
        print(f"✓ Index already exists: {index_name}")
    
    return pc.Index(index_name)


def _check_index_populated(pc, index_name):
    """Check if a Pinecone index exists and has vectors."""
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        return False
    
    pinecone_index = pc.Index(index_name)
    stats = pinecone_index.describe_index_stats()
    total_vector_count = stats.get('total_vector_count', 0)
    
    return total_vector_count > 0


def index_documents_to_pinecone(nodes: List[Document], index_name: str):
    """Index documents to Pinecone."""
    print(f"\n{'='*70}")
    print("INDEXING TO PINECONE")
    print(f"{'='*70}")
    
    # Initialize Pinecone
    pc = _initialize_pinecone()
    pinecone_index = _create_or_get_pinecone_index(pc, index_name)
    
    # Setup embedding model
    print(f"\n🔧 Initializing embedding model: {EMBEDDING_MODEL}...")
    embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL,
        device="cpu",
        trust_remote_code=True,
    )
    Settings.embed_model = embed_model
    print(f"✓ Embedding model initialized")
    
    # Convert Documents to TextNodes for indexing
    print(f"\n📦 Converting Documents to TextNodes...")
    text_nodes = []
    for idx, doc in enumerate(nodes):
        # Print metadata before embedding
        print(f"\n--- Chunk {idx} Metadata ---")
        print(json.dumps(doc.metadata, indent=2))
        
        text_node = TextNode(
            text=doc.text,
            metadata=doc.metadata,
            excluded_embed_metadata_keys=doc.excluded_embed_metadata_keys,
            excluded_llm_metadata_keys=doc.excluded_llm_metadata_keys,
        )
        text_nodes.append(text_node)
    print(f"\n✓ Converted {len(text_nodes)} nodes")
    
    # Create vector store and index
    print(f"\n🚀 Indexing to Pinecone...")
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex(
        nodes=text_nodes,
        storage_context=storage_context,
        show_progress=True,
    )
    
    # Verify indexing
    stats = pinecone_index.describe_index_stats()
    total_vectors = stats.get('total_vector_count', 0)
    print(f"\n✓ Indexing complete!")
    print(f"  Total vectors in index: {total_vectors}")
    
    return index


# ============================================================================
# RERANKING FUNCTIONS
# ============================================================================

def _setup_reranker():
    """Configure the reranker model."""
    reranker = CrossEncoder(RERANKER_MODEL, max_length=512, device="cpu")
    print(f"✓ Reranker model configured ({RERANKER_MODEL})")
    return reranker


def _rerank_chunks(reranker, query: str, chunks: List[Dict[str, Any]], enriched_texts: List[str] = None) -> List[Dict[str, Any]]:
    """Rerank chunks using cross-encoder reranker with metadata-enriched content.
    
    Args:
        reranker: CrossEncoder model for reranking
        query: Query string
        chunks: List of chunk dictionaries with basic info
        enriched_texts: List of metadata-enriched texts (same format as used for embedding)
                       If None, uses plain chunk["text"]
    """
    if not chunks:
        return []
    
    # Use enriched texts (with metadata) if provided, otherwise use plain text
    chunk_texts = enriched_texts if enriched_texts else [chunk["text"] for chunk in chunks]
    
    pairs = [(query, chunk_text) for chunk_text in chunk_texts]
    rerank_scores = reranker.predict(pairs)
    
    for idx, chunk in enumerate(chunks):
        chunk["rerank_score"] = float(rerank_scores[idx])
        chunk["original_rank"] = idx + 1
        chunk["original_score"] = chunk.get("score", "N/A")
    
    reranked_chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    
    for idx, chunk in enumerate(reranked_chunks):
        chunk["new_rank"] = idx + 1
    
    return reranked_chunks


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

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


def _get_eval_batch_runner():
    """Get evaluation batch runner using semantic similarity."""
    embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL,
        device="cpu",
        trust_remote_code=True
    )
    evaluator_s = SemanticSimilarityEvaluator(embed_model=embed_model)
    eval_batch_runner = BatchEvalRunner(
        {"semantic_similarity": evaluator_s},
        workers=1,
        show_progress=True
    )
    return eval_batch_runner


async def evaluate_rag():
    """Run RAG evaluation."""
    start_time = time.time()
    
    print(f"\n{'='*70}")
    print("RAG EVALUATION - DOCLING HYBRID CHUNKER WITH RERANKING")
    print(f"{'='*70}")
    
    print(f"\n🔧 Configuration:")
    print(f"   Index: {PINECONE_INDEX_NAME}")
    print(f"   Initial retrieval: {INITIAL_TOP_K}")
    print(f"   Reranker: {RERANKER_MODEL}")
    print(f"   Top-N after reranking: {RERANK_TOP_N}")
    print(f"   Similarity cutoff: {SIMILARITY_CUTOFF}")
    print(f"   Query delay: {QUERY_DELAY}s")
    
    # Setup components
    print(f"\n🔧 Initializing components...")
    
    # Embedding model
    embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL,
        device="cpu",
        trust_remote_code=True
    )
    Settings.embed_model = embed_model
    print(f"✓ Embedding model: {EMBEDDING_MODEL}")
    
    # Reranker
    reranker = _setup_reranker()
    
    # OpenRouter LLM
    llm = _setup_openrouter_llm()
    Settings.llm = llm
    
    # Load index
    print(f"\n📂 Loading Pinecone index...")
    try:
        index = _load_index(PINECONE_INDEX_NAME)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return
    
    # Load evaluation dataset
    print(f"\n📋 Loading evaluation dataset...")
    eval_dataset = QueryResponseDataset.from_json(str(EVAL_DATASET_PATH))
    eval_qs = eval_dataset.questions
    ref_response_strs = [r for (_, r) in eval_dataset.qr_pairs]
    print(f"✓ Loaded {len(eval_qs)} evaluation questions")
    
    # Run evaluation loop
    print(f"\n{'='*70}")
    print(f"Starting Evaluation Loop ({len(eval_qs)} questions)")
    print(f"{'='*70}\n")
    
    pred_response_objs = []
    generated_answers = []
    original_chunks_list = []
    reranked_chunks_list = []
    final_chunks_to_llm_list = []
    
    for idx, query in enumerate(eval_qs, 1):
        max_retries = 3
        retry_count = 0
        response = None
        answer_text = ""
        original_chunks = []
        reranked_chunks = []
        top_n_chunks = []
        
        while retry_count < max_retries:
            try:
                print(f"[{idx}/{len(eval_qs)}] Query: {query[:60]}...")
                
                # Step 1: Retrieve initial chunks
                print(f"   🔍 Step 1: Retrieving top {INITIAL_TOP_K} chunks...")
                retriever = index.as_retriever(similarity_top_k=INITIAL_TOP_K)
                initial_nodes = retriever.retrieve(query)
                
                original_chunks = []
                enriched_chunk_texts = []  # Store metadata-enriched versions for reranking
                for node_idx, source_node in enumerate(initial_nodes, 1):
                    # Get plain text for storage
                    node_text = source_node.node.get_content()
        
                    # Get enriched text (with metadata) as it was embedded in Pinecone
                    enriched_text = source_node.node.get_content(metadata_mode=MetadataMode.EMBED)
                    print("*****node Text with metadata (enriched) **********")
                    print(enriched_text)
                    # # Extract and print metadata section for debugging (first chunk only)
                    # if node_idx == 1:
                    metadata_match = re.search(r'<metadata>.*?</metadata>', enriched_text, re.DOTALL)
                    if metadata_match:
                        metadata_section = metadata_match.group(0)
                        print(f"\n      📋 METADATA SECTION (used for reranking):")
                        print(f"      {metadata_section}\n")
                    
                    score = source_node.score if hasattr(source_node, 'score') else 0.0
                    original_chunks.append({
                        "chunk_index": node_idx,
                        "text": node_text,
                        "score": float(score) if isinstance(score, (int, float)) else score,
                    })
                    enriched_chunk_texts.append(enriched_text)
                
                print(f"      ✓ Retrieved {len(original_chunks)} chunks")
                print(f"      ℹ️  Using metadata-enriched content (same as embedding) for reranking")
                
                # Step 2: Rerank chunks using metadata-enriched content
                print(f"   🎯 Step 2: Reranking {len(original_chunks)} chunks with enriched metadata...")
                reranked_chunks = _rerank_chunks(reranker, query, original_chunks.copy(), enriched_chunk_texts)
                print(f"      ✓ Reranking complete")
                
                # Step 3: Select top N and get their enriched texts
                top_n_chunks = reranked_chunks[:RERANK_TOP_N]
                
                # Map reranked chunks back to their enriched texts
                # Build a mapping of original chunk indices to enriched texts
                enriched_texts_mapping = {}
                for orig_idx, enriched_text in enumerate(enriched_chunk_texts):
                    enriched_texts_mapping[orig_idx] = enriched_text
                
                # Get enriched versions of top N chunks
                top_n_chunks_with_enriched = []
                for chunk in top_n_chunks:
                    chunk_copy = chunk.copy()
                    # Find the enriched text for this chunk based on its original index
                    orig_index = chunk.get("chunk_index", 0) - 1  # Convert to 0-based
                    if orig_index in enriched_texts_mapping:
                        chunk_copy["text_with_metadata"] = enriched_texts_mapping[orig_index]
                    top_n_chunks_with_enriched.append(chunk_copy)
                
                print(f"   ✂️  Step 3: Selected top {RERANK_TOP_N} chunks for LLM")
                
                # Step 4: Generate answer
                print(f"   🤖 Step 4: Generating answer with LLM...")
                
                context_parts = []
                for i, chunk in enumerate(top_n_chunks, 1):
                    context_parts.append(f"[Context {i}]:\n{chunk['text']}\n")
                
                context_str = "\n".join(context_parts)
                
                prompt = f"""{SYSTEM_PROMPT}

Context Information:
{context_str}

Query: {query}

Answer:"""
                
                llm_response = Settings.llm.complete(prompt)
                answer_text = str(llm_response)
                
                # Create response object
                class SimpleResponse:
                    def __init__(self, response_text, source_nodes):
                        self.response = response_text
                        self.source_nodes = source_nodes
                    
                    def __str__(self):
                        return self.response
                
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
                
                break
                
            except openai.RateLimitError as e:
                retry_count += 1
                print(f"   ⚠️  Rate limit hit (attempt {retry_count}/{max_retries})")
                
                if retry_count < max_retries:
                    wait_time = 30 * retry_count
                    print(f"   ⏳ Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Rate limit exceeded after {max_retries} retries") from e
                    
            except Exception as e:
                if _is_rate_limit_error(e):
                    retry_count += 1
                    if retry_count < max_retries:
                        wait_time = 30 * retry_count
                        print(f"   ⏳ Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    else:
                        raise Exception(f"Rate limit exceeded after {max_retries} retries") from e
                else:
                    print(f"   ❌ Error: {str(e)[:100]}")
                    raise
        
        if response is None:
            raise Exception(f"Failed to get response for query {idx}")
        
        pred_response_objs.append(response)
        generated_answers.append(answer_text)
        original_chunks_list.append(original_chunks)
        reranked_chunks_list.append(reranked_chunks)
        final_chunks_to_llm_list.append(top_n_chunks_with_enriched)  # Use enriched chunks with metadata
        
        answer_preview = answer_text[:150] + "..." if len(answer_text) > 150 else answer_text
        print(f"   💬 Answer: {answer_preview}")
        print(f"   ✓ Response generated\n")
        
        if idx < len(eval_qs):
            print(f"   ⏳ Waiting {QUERY_DELAY}s...")
            time.sleep(QUERY_DELAY)
    
    # Run evaluation
    print(f"\n{'='*70}")
    print("Running Semantic Similarity Evaluation")
    print(f"{'='*70}\n")
    
    eval_batch_runner = _get_eval_batch_runner()
    eval_results = await eval_batch_runner.aevaluate_responses(
        eval_qs, responses=pred_response_objs, reference=ref_response_strs
    )
    
    semantic_scores = np.array([r.score for r in eval_results["semantic_similarity"]])
    mean_score = semantic_scores.mean()
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Save results
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "pinecone_index": PINECONE_INDEX_NAME,
            "source_document": str(SOURCE_DOCUMENT),
            "chunking_method": "docling_hybrid_chunker",
            "initial_top_k": INITIAL_TOP_K,
            "reranker_model": RERANKER_MODEL,
            "rerank_top_n": RERANK_TOP_N,
            "similarity_cutoff": SIMILARITY_CUTOFF,
            "max_tokens_per_chunk": MAX_TOKENS_PER_CHUNK,
            "merge_peers": MERGE_PEERS,
            "llm_model": LLM_MODEL,
            "llm_provider": "openrouter",
            "llm_temperature": 0.1,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "system_prompt_mode": "critical_software_specifications",
            "metadata_enrichment": "questions_answered_extractor",
            "excluded_embed_metadata_keys": ["chunk_index"],
            "excluded_llm_metadata_keys": ["chunk_index"],
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
                "original_chunks": original_chunks_list[idx],
                "reranked_chunks": reranked_chunks_list[idx],
                "final_chunks_to_llm": final_chunks_to_llm_list[idx],
            }
            for idx, score in enumerate(semantic_scores)
        ],
        "execution_info": {
            "duration_seconds": round(duration, 2),
            "duration_minutes": round(duration / 60, 2),
        }
    }
    
    with open(OUTPUT_EVAL_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print("📊 EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"Mean Semantic Similarity: {mean_score:.4f}")
    print(f"Min Score: {np.min(semantic_scores):.4f}")
    print(f"Max Score: {np.max(semantic_scores):.4f}")
    print(f"Std Dev: {np.std(semantic_scores):.4f}")
    print(f"Total Questions: {len(semantic_scores)}")
    print(f"\n✓ Results saved to {OUTPUT_EVAL_FILE}")
    print(f"⏱️  Total Execution Time: {duration:.2f}s ({duration/60:.2f}m)")
    print(f"{'='*70}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

async def main():
    """Main function to orchestrate chunking, indexing, and evaluation."""
    
    parser = argparse.ArgumentParser(
        description="Docling Hybrid Chunker - Index and Evaluate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index only (will chunk document and extract questions)
  python chunk_index_and_eval_docling_hybrid.py --mode index
  
  # Index from chunks JSON (skip chunking, just load and index)
  python chunk_index_and_eval_docling_hybrid.py --mode index-from-json
  
  # Evaluate only (requires existing index)
  python chunk_index_and_eval_docling_hybrid.py --mode eval
  
  # Run both indexing and evaluation
  python chunk_index_and_eval_docling_hybrid.py --mode all
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["index", "eval", "all", "index-from-json"],
        default="all",
        help="Operation mode: index, eval, all, or index-from-json (default: all)"
    )
    
    parser.add_argument(
        "--chunks-json",
        type=str,
        default=str(OUTPUT_CHUNKS_FILE),
        help=f"Path to chunks JSON file (default: {OUTPUT_CHUNKS_FILE})"
    )
    
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip evaluation phase (only index)"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("DOCLING HYBRID CHUNKER WITH METADATA ENRICHMENT")
    print(f"{'='*70}")
    print(f"Mode: {args.mode.upper()}")
    if args.mode != "index-from-json":
        print(f"Source Document: {SOURCE_DOCUMENT}")
    else:
        print(f"Chunks JSON: {args.chunks_json}")
    print(f"Pinecone Index: {PINECONE_INDEX_NAME}")
    print(f"{'='*70}")
    
    # Handle index-from-json mode
    if args.mode == "index-from-json":
        chunks_json_path = Path(args.chunks_json)
        if not chunks_json_path.exists():
            print(f"❌ Error: Chunks JSON not found: {chunks_json_path}")
            return
        
        print("\n" + "="*70)
        print("PHASE: LOAD CHUNKS FROM JSON AND INDEX")
        print("="*70)
        
        # Load chunks from JSON
        nodes = load_chunks_from_json_for_indexing(chunks_json_path)
        
        # Index to Pinecone
        try:
            index_documents_to_pinecone(nodes, PINECONE_INDEX_NAME)
            print(f"\n✓ COMPLETE: Chunks loaded from JSON and indexed to Pinecone")
        except Exception as e:
            print(f"\n❌ Indexing failed: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        print(f"\n{'='*70}")
        print("OPERATION COMPLETE!")
        print(f"{'='*70}")
        return
    
    # Check source document exists for other modes
    if not SOURCE_DOCUMENT.exists():
        print(f"❌ Error: Source document not found: {SOURCE_DOCUMENT}")
        return
    
    # Index mode
    if args.mode in ["index", "all"]:
        print("\n" + "="*70)
        print("PHASE 1: CHUNKING AND INDEXING")
        print("="*70)
        
        # Check if persisted nodes exist
        if OUTPUT_NODES_FILE.exists():
            print(f"\n💾 Found persisted nodes: {OUTPUT_NODES_FILE}")
            print(f"   File size: {OUTPUT_NODES_FILE.stat().st_size} bytes")
            response = input("\n   Load from persisted nodes? (y/n): ").strip().lower()
            
            if response == 'y':
                print("\n📦 Loading enriched nodes from persistence...")
                nodes = load_nodes_from_json(OUTPUT_NODES_FILE)
                print(f"✓ Loaded {len(nodes)} nodes (skipping chunking and question extraction)")
            else:
                print("\n⚠️  Will re-chunk and re-extract questions...")
                # Initialize LLM for question extraction
                qa_llm = _initialize_question_extraction_llm()
                
                # Chunk document
                output_chunks, nodes = chunk_markdown_with_docling_hybrid(
                    SOURCE_DOCUMENT,
                    llm=qa_llm
                )
                
                # Save chunks to JSON
                print(f"\n💾 Saving chunks to {OUTPUT_CHUNKS_FILE}...")
                with open(OUTPUT_CHUNKS_FILE, "w", encoding="utf-8") as f:
                    json.dump(output_chunks, f, indent=2, ensure_ascii=False)
                print(f"✓ Chunks saved ({OUTPUT_CHUNKS_FILE.stat().st_size} bytes)")
                
                # Persist enriched nodes
                print(f"\n💾 Persisting enriched nodes to {OUTPUT_NODES_FILE}...")
                save_nodes_to_json(nodes, OUTPUT_NODES_FILE)
        else:
            print("\n⚙️  No persisted nodes found, will chunk and extract questions...")
            # Initialize LLM for question extraction
            qa_llm = _initialize_question_extraction_llm()
            
            # Chunk document
            output_chunks, nodes = chunk_markdown_with_docling_hybrid(
                SOURCE_DOCUMENT,
                llm=qa_llm
            )
            
            # Save chunks to JSON
            print(f"\n💾 Saving chunks to {OUTPUT_CHUNKS_FILE}...")
            with open(OUTPUT_CHUNKS_FILE, "w", encoding="utf-8") as f:
                json.dump(output_chunks, f, indent=2, ensure_ascii=False)
            print(f"✓ Chunks saved ({OUTPUT_CHUNKS_FILE.stat().st_size} bytes)")
            
            # Persist enriched nodes
            print(f"\n💾 Persisting enriched nodes to {OUTPUT_NODES_FILE}...")
            save_nodes_to_json(nodes, OUTPUT_NODES_FILE)
        
        # Index to Pinecone
        try:
            index_documents_to_pinecone(nodes, PINECONE_INDEX_NAME)
            print(f"\n✓ PHASE 1 COMPLETE: Chunking and Indexing")
        except Exception as e:
            print(f"\n❌ Indexing failed: {e}")
            print(f"\n💡 TIP: Enriched nodes are saved to {OUTPUT_NODES_FILE}")
            print(f"   You can re-run with --mode index to retry indexing without re-chunking.")
            raise
    
    # Eval mode
    if args.mode in ["eval", "all"] and not args.no_eval:
        if args.mode == "all":
            print("\n" + "="*70)
            print("PHASE 2: EVALUATION")
            print("="*70)
        
        await evaluate_rag()
        
        if args.mode == "all":
            print(f"\n✓ PHASE 2 COMPLETE: Evaluation")
    elif args.no_eval and args.mode == "all":
        print("\n⚠️  Evaluation skipped (--no-eval flag)")
    
    print(f"\n{'='*70}")
    print("ALL OPERATIONS COMPLETE!")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
