"""
Test script to process a document using Preprocess API with current parameters.
Tests chunking without storing to Pinecone.

Usage:
    python test.py
"""

import os
import json
from dotenv import load_dotenv
from pypreprocess import Preprocess

# Load environment variables
load_dotenv()

# Configuration
TEST_DOCUMENT = "test-include-sub-headers.docx"
OUTPUT_FILE = "test_chunks_output.json"

# Preprocessing parameters (from chunk_kb_docs_with_preprocess_tool.py)
PREPROCESS_OPTIONS = {
    "table_output_format": "markdown",
    "repeat_table_header": True, 
    "merge": True,
    "keep_header": True, 
    "smart_header": False,
    "keep_footer": False,
    "image_text": False,
    "repeat_title": True,
    "language": "fr", 
}


def main():
    """Main test function."""
    print("=" * 70)
    print("Preprocess API Test - Document Chunking")
    print("=" * 70)
    
    # Get API key
    preprocess_api_key = os.getenv("PREPROCESS_API_KEY_1")
    if not preprocess_api_key:
        print("❌ Error: PREPROCESS_API_KEY_1 not found in environment variables")
        return
    
    print(f"\n📄 Test Document: {TEST_DOCUMENT}")
    print(f"📋 Output File: {OUTPUT_FILE}")
    
    # Check if document exists
    if not os.path.exists(TEST_DOCUMENT):
        print(f"❌ Error: Document not found: {TEST_DOCUMENT}")
        return
    
    print("\n⚙️  Preprocess Options:")
    for key, value in PREPROCESS_OPTIONS.items():
        print(f"   {key}: {value}")
    
    # Initialize Preprocess client
    print("\n🔧 Initializing Preprocess client...")
    preprocess = Preprocess(api_key=preprocess_api_key)
    
    # Set the filepath
    preprocess.set_filepath(TEST_DOCUMENT)
    
    # Set preprocessing options
    preprocess.set_options(PREPROCESS_OPTIONS)
    
    # Process the document (upload and chunk)
    print("\n📤 Uploading and starting chunking job...")
    preprocess.chunk()
    
    # Wait for the chunking to complete (blocks until done)
    print("⏳ Waiting for chunking to complete...")
    result = preprocess.wait()
    
    # Get chunks from the result
    chunks = result.data['chunks']
    print(f"\n✓ Successfully loaded {len(chunks)} chunks")
    
    # Display chunk previews
    print("\n" + "=" * 70)
    print("Chunk Previews")
    print("=" * 70)
    
    for i, chunk in enumerate(chunks[:5], 1):  # Show first 5 chunks
        preview = chunk[:150] + "..." if len(chunk) > 150 else chunk
        print(f"\nChunk {i}:")
        print(f"Length: {len(chunk)} characters")
        print(f"Preview: {preview}")
    
    if len(chunks) > 5:
        print(f"\n... and {len(chunks) - 5} more chunks")
    
    # Save chunks to JSON file
    print(f"\n💾 Saving chunks to {OUTPUT_FILE}...")
    output_data = {
        "document": TEST_DOCUMENT,
        "total_chunks": len(chunks),
        "preprocess_options": PREPROCESS_OPTIONS,
        "chunks": [
            {
                "chunk_index": i,
                "text": chunk,
                "length": len(chunk)
            }
            for i, chunk in enumerate(chunks)
        ]
    }
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Chunks saved to {OUTPUT_FILE}")
    
    # Summary statistics
    chunk_lengths = [len(chunk) for chunk in chunks]
    avg_length = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
    
    print("\n" + "=" * 70)
    print("📊 Summary Statistics")
    print("=" * 70)
    print(f"Document: {TEST_DOCUMENT}")
    print(f"Total Chunks: {len(chunks)}")
    print(f"Average Chunk Length: {avg_length:.0f} characters")
    print(f"Min Chunk Length: {min(chunk_lengths) if chunk_lengths else 0} characters")
    print(f"Max Chunk Length: {max(chunk_lengths) if chunk_lengths else 0} characters")
    print("=" * 70)
    print("✓ Test complete!")


if __name__ == "__main__":
    main()
