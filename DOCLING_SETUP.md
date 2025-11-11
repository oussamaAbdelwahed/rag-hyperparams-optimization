# Docling Integration Setup

## Installation

To use the Docling-based document preprocessing (`chunk_and_embed_with_docling_tool.py`), you need to install additional packages.

### Step 1: Activate your virtual environment

```bash
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows
```

### Step 2: Install the updated requirements

```bash
pip install -r requirements.txt
```

This will install the new Docling packages:

- `llama-index-readers-docling` - For reading documents with Docling
- `llama-index-node-parser-docling` - For semantic chunking

### Step 3: Verify installation

```bash
python -c "from llama_index.readers.docling import DoclingReader; from llama_index.node_parser.docling import DoclingNodeParser; print('✓ Docling packages installed successfully')"
```

## Usage

Once installed, run the script:

```bash
python chunk_and_embed_with_docling_tool.py
```

## Features

The Docling integration provides:

- ✅ **Table extraction** with markdown formatting
- ✅ **Smart header/footer** detection and handling
- ✅ **Section hierarchy** preservation
- ✅ **Paragraph boundary** detection
- ✅ **Semantic chunking** that respects document structure
- ✅ **Rich metadata** including page numbers, bounding boxes, headings

## Configuration

The script uses the same configuration as `chunk_kb_docs_with_preprocess_tool.py`:

- **Document**: `Spec détaillées - Middleware XL EDS (ATS __ XL EDS __ CHRONOPOST).docx`
- **Index name**: `docling-tool-chunking-ats-chrono`
- **Embedding model**: `BAAI/bge-base-en-v1.5` (768 dimensions)
- **Vector store**: Pinecone (AWS us-east-1)

## Docling vs Preprocess API

| Feature    | Preprocess API  | Docling                     |
| ---------- | --------------- | --------------------------- |
| Processing | Remote API      | Local processing            |
| Speed      | Depends on API  | Fast, especially with GPU   |
| Tables     | Markdown format | Markdown + structure        |
| Metadata   | Basic           | Rich (bbox, page, headings) |
| Chunking   | API-based       | Semantic, structure-aware   |
| Cost       | API usage       | Free, open-source           |

## Troubleshooting

If you encounter import errors:

1. Make sure your venv is activated
2. Reinstall requirements: `pip install -r requirements.txt`
3. Check Python version (requires Python 3.9+)

For GPU acceleration (optional but recommended):

- Docling can use GPU for faster processing
- Install with CUDA support if available
