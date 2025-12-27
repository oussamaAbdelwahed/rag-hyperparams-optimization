"""
Test script to chunk a markdown document using Docling's HybridChunker.

This script:
1. Loads the test-docling-chunking.md file
2. Uses Docling's HybridChunker to chunk the document
3. Extracts questions using QuestionsAnsweredExtractor
4. Writes the output chunks to chunks.json

Usage:
    python test-markdown-alteration.py
"""
from dotenv import load_dotenv

import os
import json
from pathlib import Path
from llama_index.core import Document
from llama_index.core.extractors import QuestionsAnsweredExtractor
from llama_index.llms.openai_like import OpenAILike #to use OpenRouter instead of Groq

from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import MetadataMode

# assume documents are defined -> extract nodes
from llama_index.core.ingestion import IngestionPipeline
try:
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
    from transformers import AutoTokenizer
    from docling_core.types.doc.document import DoclingDocument
    from docling_core.transforms.chunker.hierarchical_chunker import (
        ChunkingDocSerializer,
        ChunkingSerializerProvider,
    )
    from docling_core.transforms.serializer.markdown import MarkdownParams, MarkdownMetaSerializer, MarkdownTableSerializer, MarkdownTextSerializer, MarkdownAnnotationSerializer, MarkdownDocSerializer, MarkdownFallbackSerializer, MarkdownInlineSerializer, MarkdownKeyValueSerializer, MarkdownPictureSerializer, MarkdownListSerializer, MarkdownFormSerializer, OrigListItemMarkerMode


    HAS_HYBRID_CHUNKER = True
    print("✅ HybridChunker and HuggingFaceTokenizer imported successfully.")
except ImportError as e:
    print(f"⚠️  Could not import HybridChunker: {e}")
    print("Install with: pip install 'docling-core[chunking]'")
    HAS_HYBRID_CHUNKER = False
    exit(1)

from docling.document_converter import DocumentConverter
from typing import Any, List


load_dotenv()



class MDSerializerProvider(ChunkingSerializerProvider):
    def get_serializer(self, doc):
        # return ChunkingDocSerializer(
        #     doc=doc,
        #     text_serializer=MarkdownTextSerializer(),
        #     annotation_serializer=MarkdownAnnotationSerializer(),
        #     inline_serializer=MarkdownInlineSerializer(),
        #     form_serializer=MarkdownFormSerializer(),
        #     key_value_serializer=MarkdownKeyValueSerializer(),
        #     picture_serializer=MarkdownPictureSerializer(),
        #     list_serializer=MarkdownListSerializer(),
        #     table_serializer=MarkdownTableSerializer(),
        #     fallback_serializer=MarkdownFallbackSerializer(),
        #     meta_serializer=MarkdownMetaSerializer(),
        #     params=MarkdownParams(
        #         caption_delim="<<<<",
        #         ensure_valid_list_item_marker=True,
        #         include_formatting=True,
        #     )
        # )
        return MarkdownDocSerializer(
            doc=doc,
        ) 


# Configuration
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"  # Tokenizer aligned with embedding model
MAX_TOKENS = 512  # Maximum tokens per chunk
MERGE_PEERS = True  # Merge undersized peer chunks when possible
MARKDOWN_INPUT_FILE = "test-docling-chunking.md"
OUTPUT_FILE = "chunks.json"
NUM_QUESTIONS = 3  # Number of questions to extract per chunk


def _initialize_llm():
    """Initialize OpenRouter LLM for question extraction."""
    api_key = os.getenv("OPEN_ROUTER_API_KEY")
    if not api_key:
        print("⚠️  OPEN_ROUTER_API_KEY not found in environment variables. Question extraction will be skipped.")
        return None
    
    llm = OpenAILike(
        model="x-ai/grok-4.1-fast",
        api_base="https://openrouter.ai/api/v1",
        api_key=api_key,
        max_retries=3,
        timeout=60.0,
    )
    return llm


def _initialize_qa_extractor(llm, num_questions: int = 3) -> QuestionsAnsweredExtractor:
    """Initialize QuestionsAnsweredExtractor with the given LLM.
    
    Args:
        llm: LLM instance to use for question generation
        num_questions: Number of questions to generate per node
        
    Returns:
        QuestionsAnsweredExtractor instance
    """
    return QuestionsAnsweredExtractor(
        questions=num_questions,
        llm=llm,
        embedding_only=False,  # Set to False to include in retrievable metadata
        show_progress=True,  # Show progress during extraction
    )


def _extract_questions_sync(nodes: List[Document], extractor: QuestionsAnsweredExtractor):
    """Extract questions synchronously using QuestionsAnsweredExtractor.
    
    Note: This uses manual LLM calls instead of the extractor's aextract() method
    because aextract() has issues with Document nodes in certain contexts.
    
    Args:
        nodes: List of Document nodes
        extractor: QuestionsAnsweredExtractor instance
        
    Returns:
        Nodes with extracted questions in metadata
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
            node.excluded_embed_metadata_keys = ["captions"]
            node.excluded_llm_metadata_keys = ["captions","questions_this_excerpt_can_answer"]
    
    return nodes


def chunk_markdown_with_docling_hybrid(file_path: Path, llm=None) -> tuple[List[dict], List[Document]]:
    """Chunk markdown document using Docling's HybridChunker with proper tokenization.
    
    Args:
        file_path: Path to the markdown file
        llm: Optional LLM instance for question extraction
    
    Returns:
        List of chunk dictionaries with text and metadata, and list of enriched Document nodes
    """
    print("\nChunking document with Docling HybridChunker...")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Max tokens per chunk: {MAX_TOKENS}")

    if llm:
        print(f"✅ LLM initialized for question extraction")
    else:
        print(f"⚠️  LLM not available - questions will not be extracted")

    
    # Step 1: Convert the markdown file to DoclingDocument
    # Note: This reads the markdown as-is and creates a proper DoclingDocument structure
    print(f"Converting markdown file to DoclingDocument...")
    converter = DocumentConverter()
    conv_result = converter.convert(
        source=file_path,
        headers={
            "type": "markdown",
            "encoding": "utf-8",
        },
    )
    dl_doc = conv_result.document

    
    # Step 2: Initialize tokenizer aligned with embedding model
    # This is CRITICAL for RAG: the chunker and embedding model must use the same tokenizer
    print(f"Loading tokenizer for embedding model: {EMBEDDING_MODEL}...")
    hf_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    
    tokenizer = HuggingFaceTokenizer(
        tokenizer=hf_tokenizer,
        max_tokens=MAX_TOKENS,  # Maximum tokens per chunk
    )
    
    # Step 3: Initialize HybridChunker with the tokenizer
    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=MERGE_PEERS,  # Merge undersized peer chunks when possible
        serializer_provider=MDSerializerProvider(),
    )
    
    # Step 4: Chunk the DoclingDocument
    print("Chunking with HybridChunker...")
    chunk_iter = chunker.chunk(dl_doc=dl_doc)
    chunks = list(chunk_iter)
    
    print(f"Created {len(chunks)} chunks from HybridChunker")
    
    # Step 5: Convert chunks to Documents (nodes)
    nodes = []
    output_chunks = []
    
    for i, chunk in enumerate(chunks):
        # Use contextualize() to get the enriched text with metadata
        enriched_text = chunker.contextualize(chunk=chunk)

        dumpedMeta: dict[str, Any]  = chunk.meta.model_dump()
        relevantMetadata = {
            "headings": dumpedMeta["headings"],
            "captions": dumpedMeta["captions"],
            "chunk_index": i,
            "questions_this_excerpt_can_answer": "",  # Placeholder for extracted questions
        }

        doc = Document(
            text=chunk.text,
            metadata=relevantMetadata,
            excluded_embedding_metadata_keys=["captions"],  # Exclude captions from embedding metadata
            excluded_llm_metadata_keys=["captions", "questions_this_excerpt_can_answer"],
            metadata_seperator="\n",
            metadata_template="{key}:{value}",
            text_template="<metadata>{metadata_str}</metadata>\n-----\nContent:\n{content}",
        )
        nodes.append(doc)

        chunk_dict = {
            "chunk_id": i,
            "text": chunk.text,
            "enriched_text": enriched_text,
            "char_count": len(enriched_text),
            "token_count": len(hf_tokenizer.encode(enriched_text)),
            "metadata": relevantMetadata,
        }
        output_chunks.append(chunk_dict)
    
    # Step 6: Extract questions using QuestionsAnsweredExtractor
    if llm and nodes:
        print(f"\nExtracting questions using QuestionsAnsweredExtractor...")
        try:
            qa_extractor = _initialize_qa_extractor(llm, NUM_QUESTIONS)
            
            # Use synchronous extraction
            extracted_nodes = _extract_questions_sync(nodes, qa_extractor)
            
            # Update output_chunks with extracted questions from metadata
            for i, node in enumerate(extracted_nodes):
                if i < len(output_chunks):
                    # QuestionsAnsweredExtractor stores questions as a string in 'questions_this_excerpt_can_answer'
                    questions_str = node.metadata.get('questions_this_excerpt_can_answer', '')
                    # Split the questions string into a list (questions are typically separated by newlines)
                    questions = [q.strip() for q in questions_str.split('\n') if q.strip()] if questions_str else []
                    output_chunks[i]['questions_this_excerpt_can_answer'] = questions
                    if questions:
                        print(f"  ✅ Chunk {i+1}: Extracted {len(questions)} questions")
                    else:
                        print(f"  ⚠️  Chunk {i+1}: No questions extracted (raw: {questions_str[:100] if questions_str else 'empty'})")
            
            nodes = extracted_nodes
            
        except Exception as e:
            import traceback
            print(f"  ❌ Error during question extraction: {e}")
            print(f"  Full traceback:")
            traceback.print_exc()
            # Set empty questions if extraction fails
            for chunk_dict in output_chunks:
                chunk_dict['questions_this_excerpt_can_answer'] = []
    else:
        # Set empty questions if no LLM available
        for chunk_dict in output_chunks:
            chunk_dict['questions_this_excerpt_can_answer'] = []

    return output_chunks, nodes




def main():
    """Main function to chunk markdown and save to JSON."""
    # Initialize LLM for question extraction
    llm = _initialize_llm()
    
    # Get the script directory
    script_dir = Path(__file__).parent
    
    # Resolve input file path
    input_path = script_dir / MARKDOWN_INPUT_FILE
    if not input_path.exists():
        print(f"❌ Error: Markdown file not found: {input_path}")
        return
    
    print(f"Loading markdown document from: {input_path}")
    print(f"File size: {input_path.stat().st_size} bytes")
    
    # Chunk the document
    chunks, nodes = chunk_markdown_with_docling_hybrid(input_path, llm=llm)
    for i, node in enumerate(nodes):
        res = node.get_content(metadata_mode=MetadataMode.LLM)
        print(f"Node Metadata enriched text {res}")
    
    # Write chunks to JSON file
    output_path = script_dir / OUTPUT_FILE
    print(f"\nWriting {len(chunks)} chunks to {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Successfully wrote chunks to {output_path}")
    print(f"  Output file size: {output_path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
