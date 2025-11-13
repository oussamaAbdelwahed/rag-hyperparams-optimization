"""
Script to chunk knowledge base documents using Preprocess API
and store them in Pinecone vector database.

Usage:
    python chunk_kb_docs_with_preprocess_tool.py --preprocess-api-key-name PREPROCESS_API_KEY_1 --doc-name chrono-docs/document1.docx
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from pypreprocess import Preprocess

# Load environment variables
load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "enhanced-preprocess-tool-chunking-ats-chrono"
EMBEDDING_DIMENSION= 768
# Preprocessing parameters
PREPROCESS_OPTIONS = {
    "table_output_format": "markdown",
    "repeat_table_header": True, 
    "merge": True,
    "keep_header": True, 
    "smart_header": False,
    "keep_footer": False,
    "image_text": False, # TODO: see if this could help (set it to true and text) (based on the nature of images we have in the documents)
    "repeat_title": True,
    "language": "fr", 
}


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Chunk knowledge base documents using Preprocess API and store in Pinecone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python chunk_kb_docs_with_preprocess_tool.py --preprocess-api-key-name PREPROCESS_API_KEY_1 --doc-name chrono-docs/doc1.docx
  python chunk_kb_docs_with_preprocess_tool.py --preprocess-api-key-name PREPROCESS_API_KEY_2 --doc-name chrono-docs/doc2.pdf
        """
    )
    
    parser.add_argument(
        "--preprocess-api-key-name",
        required=True,
        help="Name of the environment variable containing the Preprocess API key (e.g., PREPROCESS_API_KEY_1)"
    )
    
    parser.add_argument(
        "--doc-name",
        required=True,
        help="Path to the document to process (e.g., chrono-docs/document.docx)"
    )
    
    return parser.parse_args()


def initialize_pinecone():
    """Initialize Pinecone client and create index if it doesn't exist."""
    print("Initializing Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists
    existing_indexes = pc.list_indexes()
    index_names = [idx.name for idx in existing_indexes]
    
    if INDEX_NAME not in index_names:
        print(f"Creating new index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"Index '{INDEX_NAME}' created successfully.")
    else:
        print(f"Index '{INDEX_NAME}' already exists.")
    
    return pc.Index(INDEX_NAME)


def load_and_preprocess_document(preprocess_api_key, document_path):
    """Load and preprocess the document using Preprocess API.
    
    Args:
        preprocess_api_key: API key for Preprocess service
        document_path: Path to the document to process
    
    Returns:
        List of LlamaIndex Document objects
    """
    print(f"Loading document: {document_path}")
    
    if not preprocess_api_key:
        raise ValueError("Preprocess API key not found")
    
    if not os.path.exists(document_path):
        raise FileNotFoundError(f"Document not found: {document_path}")
    
    # Get document name for metadata
    doc_name = Path(document_path).name
    
    # Initialize Preprocess client
    preprocess = Preprocess(api_key=preprocess_api_key)
    
    # Set the filepath
    preprocess.set_filepath(document_path)
    
    # Set preprocessing options
    preprocess.set_options(PREPROCESS_OPTIONS)
    
    # Process the document (upload and chunk)
    print("Uploading and starting chunking job...")
    preprocess.chunk()
    
    # Wait for the chunking to complete (blocks until done)
    print("Waiting for chunking to complete...")
    result = preprocess.wait()
    
    # Get chunks from the result
    chunks = result.data['chunks']
    print(f"✓ Successfully loaded {len(chunks)} chunks")
    
    # Convert chunks to LlamaIndex Document objects
    documents = []
    for i, chunk in enumerate(chunks):
        # Chunks are strings according to the SDK documentation
        chunk_text = chunk
        
        metadata = {
            "chunk_id": i,
            "document_name": doc_name,
            "document_path": document_path,
        }
        
        doc = Document(
            text=chunk_text,
            metadata=metadata
        )
        documents.append(doc)
    
    return documents


def chunk_and_store_documents(documents, pinecone_index):
    """Chunk documents and store them in Pinecone."""
    print("Setting up embedding model...")
    
    # Initialize embedding model
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-base-en-v1.5"
    )
    
    # Create vector store
    print("Setting up Pinecone vector store...")
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index
    )
    
    # Create storage context
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )
    
    # Create index from documents
    print("Creating vector index and storing chunks in Pinecone...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )
    
    print("Documents successfully chunked and stored in Pinecone!")
    return index


def main():
    """Main execution function."""
    print("=" * 60)
    print("Knowledge Base Document Chunking Pipeline")
    print("=" * 60)
    
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Get the Preprocess API key from environment
        preprocess_api_key = os.getenv(args.preprocess_api_key_name)
        if not preprocess_api_key:
            raise ValueError(
                f"Environment variable '{args.preprocess_api_key_name}' not found. "
                f"Please ensure it is set in your .env file or environment."
            )
        
        print(f"\nConfiguration:")
        print(f"  API Key Env Var: {args.preprocess_api_key_name}")
        print(f"  Document Path: {args.doc_name}")
        print(f"  Pinecone Index: {INDEX_NAME}")
        print()
        
        # Step 1: Initialize Pinecone
        pinecone_index = initialize_pinecone()
        
        # Step 2: Load and preprocess document
        documents = load_and_preprocess_document(preprocess_api_key, args.doc_name)
        
        if not documents:
            print("No documents to process. Exiting.")
            return
        
        # Step 3: Chunk and store in Pinecone
        index = chunk_and_store_documents(documents, pinecone_index)
        
        print("\n" + "=" * 60)
        print("Processing completed successfully!")
        print(f"Document: {args.doc_name}")
        print(f"Total chunks: {len(documents)}")
        print(f"Index name: {INDEX_NAME}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
