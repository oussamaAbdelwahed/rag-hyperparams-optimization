"""
Example RAG query with hierarchical context expansion.

This script demonstrates how to:
1. Query Pinecone index with hierarchical chunks
2. Expand retrieved chunks using hierarchical relationships
3. Use expanded context for better RAG responses
"""

import os
import json
from dotenv import load_dotenv
from typing import List

# LlamaIndex imports
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.groq import Groq

# Pinecone
from pinecone import Pinecone

# Hierarchical retrieval utilities
from hierarchical_retrieval_utils import HierarchicalChunkNavigator, reconstruct_context

# Rich for output
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

load_dotenv()
console = Console()

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
INDEX_NAME = "chrono-docling-hierarchical"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"  # Same model used for indexing
METADATA_FILE = f"{INDEX_NAME}_hierarchical_metadata.json"


def initialize_components():
    """Initialize Pinecone, embeddings, and LLM."""
    console.print("[bold blue]Initializing RAG components...[/bold blue]")
    
    # Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(INDEX_NAME)
    
    # Embeddings
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    
    # Vector store
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    
    # Index
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model
    )
    
    # LLM
    llm = Groq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
    
    console.print("[green]✓ Components initialized[/green]\n")
    
    return index, llm, pinecone_index


def load_hierarchical_navigator():
    """Load the hierarchical navigator from exported metadata."""
    console.print(f"[bold blue]Loading hierarchical metadata from {METADATA_FILE}...[/bold blue]")
    
    try:
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chunks_metadata = [chunk['metadata'] for chunk in data['chunks']]
        navigator = HierarchicalChunkNavigator(chunks_metadata)
        
        console.print(f"[green]✓ Loaded {len(chunks_metadata)} chunks with hierarchical metadata[/green]\n")
        return navigator
    
    except FileNotFoundError:
        console.print(f"[red]Error: {METADATA_FILE} not found![/red]")
        console.print(f"[yellow]Make sure to run process_docx_with_docling.py with --export-metadata flag first.[/yellow]\n")
        return None


def query_with_hierarchical_context(
    query: str,
    index: VectorStoreIndex,
    llm: Groq,
    navigator: HierarchicalChunkNavigator,
    top_k: int = 5,
    expansion_strategy: str = "parent_and_siblings"
):
    """
    Query the RAG system with hierarchical context expansion.
    
    Args:
        query: User query
        index: VectorStoreIndex
        llm: Language model
        navigator: HierarchicalChunkNavigator for context expansion
        top_k: Number of chunks to retrieve
        expansion_strategy: How to expand context (see hierarchical_retrieval_utils.py)
    """
    console.print(Panel(f"[bold cyan]Query:[/bold cyan] {query}", expand=False))
    
    # Step 1: Basic retrieval
    console.print(f"\n[bold blue]Step 1: Retrieving top-{top_k} chunks...[/bold blue]")
    query_engine = index.as_query_engine(
        similarity_top_k=top_k,
        llm=llm
    )
    
    response = query_engine.query(query)
    
    # Extract retrieved chunk IDs
    retrieved_chunk_ids = [
        node.metadata.get('chunk_id') or node.metadata.get('chunk_index')
        for node in response.source_nodes
    ]
    
    console.print(f"[green]✓ Retrieved chunks: {retrieved_chunk_ids}[/green]\n")
    
    # Step 2: Expand with hierarchical context
    console.print(f"[bold blue]Step 2: Expanding context using '{expansion_strategy}' strategy...[/bold blue]")
    
    expanded_chunks = reconstruct_context(
        retrieved_chunk_ids,
        navigator,
        strategy=expansion_strategy
    )
    
    console.print(f"[green]✓ Expanded from {len(retrieved_chunk_ids)} to {len(expanded_chunks)} chunks[/green]\n")
    
    # Step 3: Display results
    console.print("[bold blue]Step 3: Response with hierarchical context:[/bold blue]\n")
    
    # Show the original response
    console.print(Panel(
        Markdown(str(response)),
        title="[bold green]RAG Response[/bold green]",
        expand=False
    ))
    
    # Show expanded chunks metadata
    console.print("\n[bold blue]Expanded chunks hierarchy:[/bold blue]")
    for i, chunk in enumerate(expanded_chunks[:10]):  # Show first 10
        chunk_id = chunk.get('chunk_id')
        headings = chunk.get('headings', [])
        level = chunk.get('level', 'N/A')
        console.print(f"  {i+1}. Chunk {chunk_id} - Level {level} - Headings: {headings}")
    
    if len(expanded_chunks) > 10:
        console.print(f"  ... and {len(expanded_chunks) - 10} more chunks")
    
    return response, expanded_chunks


def main():
    """Main execution."""
    console.print("[bold green]Hierarchical RAG Query Example[/bold green]\n")
    console.print("=" * 60)
    console.print()
    
    # Initialize
    index, llm, pinecone_index = initialize_components()
    navigator = load_hierarchical_navigator()
    
    if navigator is None:
        console.print("[red]Cannot proceed without hierarchical metadata. Exiting.[/red]")
        return
    
    # Example queries
    queries = [
        "What are the main features of the document?",
        "Explain the methodology section",
        "What are the key findings?"
    ]
    
    console.print("[bold cyan]Running example queries with hierarchical context expansion...[/bold cyan]\n")
    
    for i, query in enumerate(queries, 1):
        console.print(f"\n{'='*60}")
        console.print(f"[bold yellow]Query {i}/{len(queries)}[/bold yellow]")
        console.print(f"{'='*60}\n")
        
        response, expanded_chunks = query_with_hierarchical_context(
            query=query,
            index=index,
            llm=llm,
            navigator=navigator,
            top_k=3,  # Start with fewer chunks
            expansion_strategy="parent_and_siblings"  # Try different strategies!
        )
        
        console.print()
    
    console.print("\n" + "="*60)
    console.print("[bold green]✓ Example queries completed![/bold green]")
    console.print("\n[dim]Try different expansion strategies:[/dim]")
    console.print("[dim]  - 'parent_only': Minimal expansion[/dim]")
    console.print("[dim]  - 'parent_and_siblings': Balanced context[/dim]")
    console.print("[dim]  - 'full_ancestors': Complete hierarchy[/dim]")
    console.print("[dim]  - 'full_context': Maximum context[/dim]")


if __name__ == "__main__":
    main()
