"""
Test script to chunk a markdown document using Docling's HybridChunker.

This script:
1. Loads the test-docling-chunking.md file
2. Uses Docling's HybridChunker to chunk the document
3. Writes the output chunks to chunks.json

Usage:
    python test-markdown-alteration.py
"""

import json
import pprint
from pathlib import Path
from llama_index.core import Document
from llama_index.core.schema import MetadataMode

# Try to import HybridChunker and tokenizer from docling
try:
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
    from transformers import AutoTokenizer
    # from docling_core.transforms.serializer.markdown import MarkdownDocSerializer
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


# class MDTableSerializerProvider(ChunkingSerializerProvider):
#     def get_serializer(self, doc):
#         return ChunkingDocSerializer(
#             doc=doc,
#             table_serializer=MarkdownTableSerializer(),  # configuring a different table serializer
#             text_serializer=MarkdownTextSerializer(),
#             annotation_serializer=MarkdownAnnotationSerializer(),
#         )

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


def chunk_markdown_with_docling_hybrid(file_path: Path):
    """Chunk markdown document using Docling's HybridChunker with proper tokenization.
    
    Args:
        file_path: Path to the markdown file
    
    Returns:
        List of chunk dictionaries with text and metadata
    """
    print("\nChunking document with Docling HybridChunker...")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Max tokens per chunk: {MAX_TOKENS}")


    
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
        # raises_on_error = True,
        # max_num_pages = 500,
        # max_file_size= 10 * 1024 * 1024, 
        # page_range = (1, 2), 
    )
    dl_doc = conv_result.document

    # exported_markdown = dl_doc.export_to_markdown()
    
    print(f"DoclingDocument created: {len(dl_doc.texts)} text elements")
    
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
    
    # Step 5: Convert chunks to dictionaries with enriched text
    output_chunks = []
    nodes = []
    for i, chunk in enumerate(chunks):
        # Use contextualize() to get the enriched text with metadata
        # This includes the metadata in the chunk itself (default behavior): llama Dociument allow to define how to serialize metadata and which metadata to pass to the embedding model and which to the llm :https://www.youtube.com/watch?v=yzPQaNhuVGU
        enriched_text = chunker.contextualize(chunk=chunk)

        dumpedMeta  = chunk.meta.model_dump()
        relevantMetadata = {
            "headings": dumpedMeta["headings"],
            "captions": dumpedMeta["captions"],
            "questions_this_excerpt_can_answer": "1: q1,2: q2" # TODO: to be generated by an LLM and written automatically using the dedicated LLAMAINDEX QuestionsAnsweredExtractor(llm=llm_transformations, questions=3)
            # "origin": dumpedMeta["origin"], # kind of useless (just infos about the source file)
        }

        doc = Document(
            text=chunk.text,
            metadata=relevantMetadata,
            #excluded_embed_metadata_keys=[],
            #excluded_llm_metadata_keys=["headings", "captions"],

            # Formatting the metadata
            metadata_seperator="\n",
            metadata_template="{key}:{value}",
            text_template="<metadata>{metadata_str}</metadata>\n-----\nContent:\n{content}", # The Anthropic way to define the metadata
        )
        nodes.append(doc)

        print("*********************** chunck metadata (from docling doc) ***************** \n", chunk.meta.model_dump())
           
        print("*********************** chunck metadata (from llamaindex created doc) ***************** \n", doc.metadata)
        
        chunk_dict = {
            "chunk_id": i,
            "text": chunk.text,
            "enriched_text": enriched_text,
            "char_count": len(enriched_text),
            # Add token count estimation (approximate)
            "token_count": len(hf_tokenizer.encode(enriched_text)),
            "docling_metadata": chunk.meta.model_dump(),
            "llamaindex_metadata": doc.metadata,
            "metadata_enriched_chunk": doc.get_content(metadata_mode=MetadataMode.EMBED)
        }
        output_chunks.append(chunk_dict)
    
    print(f"Converted to {len(output_chunks)} output chunks")
    
    # Print some statistics
    if output_chunks:
        char_counts = [c["char_count"] for c in output_chunks]
        token_counts = [c["token_count"] for c in output_chunks]
        
        print(f"\nChunk Statistics:")
        print(f"  Total chunks: {len(output_chunks)}")
        print(f"  Character count - Min: {min(char_counts)}, Max: {max(char_counts)}, Avg: {sum(char_counts) // len(char_counts)}")
        print(f"  Token count - Min: {min(token_counts)}, Max: {max(token_counts)}, Avg: {sum(token_counts) // len(token_counts)}")
    
    return output_chunks


def main():
    """Main function to chunk markdown and save to JSON."""
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
    chunks = chunk_markdown_with_docling_hybrid(input_path)
    for chunk in chunks:
        pprint.pprint(chunk)
    
    # Write chunks to JSON file
    output_path = script_dir / OUTPUT_FILE
    print(f"\nWriting {len(chunks)} chunks to {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Successfully wrote chunks to {output_path}")
    print(f"  Output file size: {output_path.stat().st_size} bytes")
    
    # Show preview of first chunk
    if chunks:
        print(f"\n=== Preview of First Chunk ===")
        print(f"Chunk ID: {chunks[0]['chunk_id']}")
        print(f"Character count: {chunks[0]['char_count']}")
        print(f"Token count: {chunks[0]['token_count']}")
        print(f"Text preview (first 200 chars):")
        print(chunks[0]['text'][:200] + "..." if len(chunks[0]['text']) > 200 else chunks[0]['text'])


if __name__ == "__main__":
    main()
