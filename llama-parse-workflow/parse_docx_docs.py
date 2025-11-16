"""
Parse DOCX documents using LlamaParse Cloud API.

This script uses LlamaParse to convert DOCX documents to high-quality markdown,
preserving structure, tables, and formatting.

Usage:
    python parse_docx_docs.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from llama_cloud_services import LlamaParse

# Load environment variables
load_dotenv()

# Configuration
LLAMA_PARSE_API_KEY = os.getenv("LLAMA_PARSE_CLOUD_API_KEY")
SOURCE_DOCUMENT = "../Spec détaillées - Middleware XL EDS (ATS __ XL EDS __ CHRONOPOST).docx"
OUTPUT_DIR = "parsed-output"


# LlamaParse Parameters
# IMPORTANT: auto_mode=True may flatten heading hierarchy. 
# If you need proper heading levels (##, ###), try setting AUTO_MODE=False
# and add result_type="markdown" parameter to the parser.
AUTO_MODE = False  # Changed from True to preserve heading hierarchy
RESULT_TYPE = "markdown"  # Explicitly set result type
MAX_PAGES = 1000
OUTPUT_TABLES_AS_HTML = True
ADAPTIVE_LONG_TABLE = True
OUTLINED_TABLE_EXTRACTION = True
HIGH_RES_OCR = False
DISABLE_OCR = False
DISABLE_IMAGE_EXTRACTION = True
ANNOTATE_LINKS = True
SKIP_DIAGONAL_TEXT = True
PRESERVE_VERY_SMALL_TEXT = True
PAGE_SEPARATOR = "\n\n---\n\n"
HIDE_HEADERS = True
HIDE_FOOTERS = True
REMOVE_HIDDEN_TEXT = True

# Parsing instruction to preserve document hierarchy
PARSING_INSTRUCTION = """
Parse this document preserving the hierarchical structure of headings.
Use proper markdown heading levels (# for main chapters, ## for sections/titles, ### for subsections/subtitles, etc.).
Maintain the document's logical hierarchy based on the original formatting and indentation.
"""


def parse_document():
    """Parse the source DOCX document and save markdown output."""
    print("=" * 60)
    print("LlamaParse Document Parsing")
    print("=" * 60)
    
    # Validate API key
    if not LLAMA_PARSE_API_KEY:
        raise ValueError("LLAMA_PARSE_CLOUD_API_KEY not found in .env file")
    
    # Validate source document
    source_path = Path(__file__).parent / SOURCE_DOCUMENT
    if not source_path.exists():
        raise FileNotFoundError(f"Source document not found: {source_path}")
    
    print(f"Source document: {source_path}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
 


    # Initialize parser
    print("Initializing LlamaParse...")
    parser = LlamaParse(
        api_key=LLAMA_PARSE_API_KEY,
       # auto_mode=AUTO_MODE,
        parse_mode="parse_page_with_llm",  # Use document mode to preserve structure
        max_pages=MAX_PAGES,
        output_tables_as_HTML=OUTPUT_TABLES_AS_HTML,
        adaptive_long_table=ADAPTIVE_LONG_TABLE,
        outlined_table_extraction=OUTLINED_TABLE_EXTRACTION,
        high_res_ocr=HIGH_RES_OCR,
        disable_ocr=DISABLE_OCR,
        disable_image_extraction=DISABLE_IMAGE_EXTRACTION,
        annotate_links=ANNOTATE_LINKS,
        skip_diagonal_text=SKIP_DIAGONAL_TEXT,
        preserve_very_small_text=PRESERVE_VERY_SMALL_TEXT,
        page_separator=PAGE_SEPARATOR,
        hide_headers=HIDE_HEADERS,
        hide_footers=HIDE_FOOTERS,
        parsing_instruction=PARSING_INSTRUCTION,  # Add parsing instruction to preserve hierarchy
        system_prompt_append=PARSING_INSTRUCTION,
    )
    
    # Parse the document
    print("Parsing document (this may take a few minutes)...")
    result = parser.parse(str(source_path))
    print("✓ Document parsed successfully")
    
    # Get markdown documents
    print("Extracting markdown documents...")
    # LlamaParse returns a list of JobResult, access the first result
    if isinstance(result, list):
        markdown_documents = result[0].get_markdown_documents(split_by_page=True)
    else:
        markdown_documents = result.get_markdown_documents(split_by_page=True)
    print(f"✓ Extracted {len(markdown_documents)} page(s)")
    
    # Create output directory
    output_path = Path(__file__).parent / OUTPUT_DIR
    output_path.mkdir(exist_ok=True)
    
    # Save markdown documents
    print(f"Saving markdown to {output_path}...")
    
    # Save as individual page files
    for i, doc in enumerate(markdown_documents):
        page_file = output_path / f"page_{i+1:03d}.md"
        with open(page_file, 'w', encoding='utf-8') as f:
            f.write(doc.text)
        print(f"  ✓ Saved page {i+1} to {page_file.name}")
    
    # Also save as a single combined file
    combined_file = output_path / "full_document.md"
    with open(combined_file, 'w', encoding='utf-8') as f:
        for i, doc in enumerate(markdown_documents):
            if i > 0:
                f.write("\n\n---\n\n")
            f.write(doc.text)
    print(f"  ✓ Saved combined document to {combined_file.name}")
    
    print()
    print("=" * 60)
    print("Parsing completed successfully!")
    print(f"Total pages: {len(markdown_documents)}")
    print(f"Output location: {output_path.absolute()}")
    print("=" * 60)
    
    return markdown_documents, output_path


def main():
    """Main execution function."""
    try:
        markdown_documents, output_path = parse_document()
        return 0
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
