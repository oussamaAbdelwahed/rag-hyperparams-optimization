#!/usr/bin/env python3
"""
Script to convert DOCX files to Markdown using MarkItDown library.

Usage:
    python markitdown.py <path_to_docx_file> [-o <output_file>]

Examples:
    python markitdown.py document.docx
    python markitdown.py document.docx -o output.md
"""

import argparse
import sys
from pathlib import Path

from markitdown import MarkItDown


def main():
    """Main function to convert DOCX to Markdown."""
    parser = argparse.ArgumentParser(
        description="Convert DOCX files to Markdown format using MarkItDown"
    )
    
    parser.add_argument(
        "docx_file",
        type=str,
        help="Path to the DOCX file to convert"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output markdown file path (optional). If not provided, prints to stdout"
    )
    
    args = parser.parse_args()
    
    # Verify the input file exists
    input_path = Path(args.docx_file)
    if not input_path.exists():
        print(f"Error: File '{args.docx_file}' not found", file=sys.stderr)
        sys.exit(1)
    
    if not input_path.suffix.lower() == ".docx":
        print(f"Warning: File does not have .docx extension", file=sys.stderr)
    
    try:
        # Initialize MarkItDown converter
        md = MarkItDown(enable_plugins=False)
        
        # Convert the DOCX file
        print(f"Converting '{args.docx_file}'...", file=sys.stderr)
        result = md.convert(str(input_path))
        
        # Output the result
        if args.output:
            output_path = Path(args.output)
            output_path.write_text(result.text_content, encoding="utf-8")
            print(f"Successfully converted to '{args.output}'", file=sys.stderr)
        else:
            print(result.text_content)
    
    except FileNotFoundError:
        print(f"Error: Could not find the file '{args.docx_file}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error during conversion: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
