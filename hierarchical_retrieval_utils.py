"""
Utility functions for hierarchical retrieval with Docling chunks.

These utilities help navigate parent/child relationships in hierarchical chunks
stored in Pinecone, enabling advanced retrieval strategies like:
- Retrieving parent chunks for broader context
- Retrieving child chunks for more specific details
- Building document trees for context-aware retrieval
"""

from typing import List, Dict, Any, Optional, Set
from collections import defaultdict


class HierarchicalChunkNavigator:
    """
    Navigate hierarchical relationships between chunks.
    
    This class builds a tree structure from chunk metadata and provides
    methods to traverse parent/child relationships.
    """
    
    def __init__(self, chunks_metadata: List[Dict[str, Any]]):
        """
        Initialize the navigator with chunk metadata.
        
        Args:
            chunks_metadata: List of metadata dictionaries from chunks
                            Each should contain: chunk_id, parent_id, doc_items, etc.
        """
        self.chunks = {chunk['chunk_id']: chunk for chunk in chunks_metadata}
        self.parent_to_children = defaultdict(list)
        self.child_to_parent = {}
        
        # Build parent-child relationships
        for chunk in chunks_metadata:
            chunk_id = chunk['chunk_id']
            parent_id = chunk.get('parent_id')
            
            if parent_id is not None:
                self.parent_to_children[parent_id].append(chunk_id)
                self.child_to_parent[chunk_id] = parent_id
    
    def get_parent(self, chunk_id: int) -> Optional[Dict[str, Any]]:
        """Get the parent chunk of a given chunk."""
        parent_id = self.child_to_parent.get(chunk_id)
        if parent_id is not None:
            return self.chunks.get(parent_id)
        return None
    
    def get_children(self, chunk_id: int) -> List[Dict[str, Any]]:
        """Get all direct children of a given chunk."""
        child_ids = self.parent_to_children.get(chunk_id, [])
        return [self.chunks[cid] for cid in child_ids if cid in self.chunks]
    
    def get_siblings(self, chunk_id: int) -> List[Dict[str, Any]]:
        """Get all sibling chunks (chunks with the same parent)."""
        parent_id = self.child_to_parent.get(chunk_id)
        if parent_id is None:
            return []
        
        sibling_ids = [
            cid for cid in self.parent_to_children.get(parent_id, [])
            if cid != chunk_id
        ]
        return [self.chunks[sid] for sid in sibling_ids if sid in self.chunks]
    
    def get_ancestors(self, chunk_id: int, max_depth: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all ancestor chunks (parent, grandparent, etc.).
        
        Args:
            chunk_id: ID of the starting chunk
            max_depth: Maximum number of levels to traverse (None for all)
        
        Returns:
            List of ancestor chunks, ordered from immediate parent to root
        """
        ancestors = []
        current_id = chunk_id
        depth = 0
        
        while True:
            if max_depth is not None and depth >= max_depth:
                break
            
            parent = self.get_parent(current_id)
            if parent is None:
                break
            
            ancestors.append(parent)
            current_id = parent['chunk_id']
            depth += 1
        
        return ancestors
    
    def get_descendants(self, chunk_id: int, max_depth: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all descendant chunks (children, grandchildren, etc.).
        
        Args:
            chunk_id: ID of the starting chunk
            max_depth: Maximum number of levels to traverse (None for all)
        
        Returns:
            List of all descendant chunks
        """
        descendants = []
        to_visit = [(chunk_id, 0)]
        visited = set()
        
        while to_visit:
            current_id, depth = to_visit.pop(0)
            
            if current_id in visited:
                continue
            
            if max_depth is not None and depth >= max_depth:
                continue
            
            visited.add(current_id)
            children = self.get_children(current_id)
            descendants.extend(children)
            
            for child in children:
                to_visit.append((child['chunk_id'], depth + 1))
        
        return descendants
    
    def get_context_window(
        self,
        chunk_id: int,
        parent_levels: int = 1,
        child_levels: int = 1,
        include_siblings: bool = False
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get a contextual window around a chunk.
        
        This is useful for retrieval - you can get the original chunk along
        with relevant surrounding context.
        
        Args:
            chunk_id: ID of the central chunk
            parent_levels: Number of parent levels to include
            child_levels: Number of child levels to include
            include_siblings: Whether to include sibling chunks
        
        Returns:
            Dictionary with 'ancestors', 'descendants', 'siblings', and 'center' keys
        """
        return {
            'center': self.chunks.get(chunk_id),
            'ancestors': self.get_ancestors(chunk_id, max_depth=parent_levels),
            'descendants': self.get_descendants(chunk_id, max_depth=child_levels),
            'siblings': self.get_siblings(chunk_id) if include_siblings else []
        }
    
    def get_chunks_by_heading(self, heading: str) -> List[Dict[str, Any]]:
        """Find all chunks that have a specific heading in their metadata."""
        matching_chunks = []
        
        for chunk in self.chunks.values():
            headings = chunk.get('headings', [])
            if heading in headings:
                matching_chunks.append(chunk)
        
        return matching_chunks
    
    def get_chunks_by_level(self, level: int) -> List[Dict[str, Any]]:
        """Get all chunks at a specific hierarchy level."""
        return [
            chunk for chunk in self.chunks.values()
            if chunk.get('level') == level
        ]
    
    def build_tree_structure(self) -> Dict[str, Any]:
        """
        Build a tree representation of the document structure.
        
        Returns:
            Nested dictionary representing the document tree
        """
        # Find root chunks (those without parents)
        roots = [
            chunk for chunk in self.chunks.values()
            if chunk['chunk_id'] not in self.child_to_parent
        ]
        
        def build_subtree(chunk_id: int) -> Dict[str, Any]:
            chunk = self.chunks[chunk_id]
            children = self.get_children(chunk_id)
            
            return {
                'chunk_id': chunk_id,
                'metadata': chunk,
                'children': [build_subtree(child['chunk_id']) for child in children]
            }
        
        return {
            'roots': [build_subtree(root['chunk_id']) for root in roots]
        }


def reconstruct_context(
    retrieved_chunk_ids: List[int],
    navigator: HierarchicalChunkNavigator,
    strategy: str = "parent_and_siblings"
) -> List[Dict[str, Any]]:
    """
    Reconstruct broader context for retrieved chunks using hierarchical information.
    
    Args:
        retrieved_chunk_ids: List of chunk IDs retrieved from vector search
        navigator: HierarchicalChunkNavigator instance
        strategy: Strategy for context expansion:
            - "parent_and_siblings": Include parent and sibling chunks
            - "full_ancestors": Include all ancestor chunks
            - "parent_only": Include only immediate parent
            - "children": Include child chunks
            - "full_context": Include ancestors, descendants, and siblings
    
    Returns:
        List of chunk metadata dictionaries with expanded context
    """
    result_chunks = []
    seen_ids = set()
    
    for chunk_id in retrieved_chunk_ids:
        if strategy == "parent_and_siblings":
            # Get parent and siblings for each retrieved chunk
            context = navigator.get_context_window(
                chunk_id,
                parent_levels=1,
                child_levels=0,
                include_siblings=True
            )
            
            if context['center'] and context['center']['chunk_id'] not in seen_ids:
                result_chunks.append(context['center'])
                seen_ids.add(context['center']['chunk_id'])
            
            for ancestor in context['ancestors']:
                if ancestor['chunk_id'] not in seen_ids:
                    result_chunks.append(ancestor)
                    seen_ids.add(ancestor['chunk_id'])
            
            for sibling in context['siblings']:
                if sibling['chunk_id'] not in seen_ids:
                    result_chunks.append(sibling)
                    seen_ids.add(sibling['chunk_id'])
        
        elif strategy == "full_ancestors":
            # Get all ancestors
            center = navigator.chunks.get(chunk_id)
            if center and center['chunk_id'] not in seen_ids:
                result_chunks.append(center)
                seen_ids.add(center['chunk_id'])
            
            ancestors = navigator.get_ancestors(chunk_id)
            for ancestor in ancestors:
                if ancestor['chunk_id'] not in seen_ids:
                    result_chunks.append(ancestor)
                    seen_ids.add(ancestor['chunk_id'])
        
        elif strategy == "parent_only":
            # Just get the parent
            center = navigator.chunks.get(chunk_id)
            if center and center['chunk_id'] not in seen_ids:
                result_chunks.append(center)
                seen_ids.add(center['chunk_id'])
            
            parent = navigator.get_parent(chunk_id)
            if parent and parent['chunk_id'] not in seen_ids:
                result_chunks.append(parent)
                seen_ids.add(parent['chunk_id'])
        
        elif strategy == "children":
            # Get children
            center = navigator.chunks.get(chunk_id)
            if center and center['chunk_id'] not in seen_ids:
                result_chunks.append(center)
                seen_ids.add(center['chunk_id'])
            
            children = navigator.get_children(chunk_id)
            for child in children:
                if child['chunk_id'] not in seen_ids:
                    result_chunks.append(child)
                    seen_ids.add(child['chunk_id'])
        
        elif strategy == "full_context":
            # Get everything
            context = navigator.get_context_window(
                chunk_id,
                parent_levels=None,
                child_levels=None,
                include_siblings=True
            )
            
            all_chunks = (
                [context['center']] + 
                context['ancestors'] + 
                context['descendants'] + 
                context['siblings']
            )
            
            for chunk in all_chunks:
                if chunk and chunk['chunk_id'] not in seen_ids:
                    result_chunks.append(chunk)
                    seen_ids.add(chunk['chunk_id'])
    
    return result_chunks


# Example usage function
def example_usage():
    """
    Example of how to use the hierarchical retrieval utilities.
    
    This would typically be called after retrieving chunks from Pinecone.
    """
    # Example: Load hierarchical metadata (from exported JSON)
    import json
    
    with open('chrono-docling-hierarchical_hierarchical_metadata.json', 'r') as f:
        exported_data = json.load(f)
    
    # Extract metadata list
    chunks_metadata = [chunk['metadata'] for chunk in exported_data['chunks']]
    
    # Create navigator
    navigator = HierarchicalChunkNavigator(chunks_metadata)
    
    # Example: Get context for retrieved chunk IDs
    retrieved_ids = [5, 12, 23]  # From Pinecone query
    
    # Expand with parent and siblings
    expanded_chunks = reconstruct_context(
        retrieved_ids,
        navigator,
        strategy="parent_and_siblings"
    )
    
    print(f"Retrieved {len(retrieved_ids)} chunks")
    print(f"Expanded to {len(expanded_chunks)} chunks with hierarchical context")
    
    # Example: Navigate the tree
    chunk_id = 5
    print(f"\nContext window for chunk {chunk_id}:")
    context = navigator.get_context_window(chunk_id, parent_levels=2, include_siblings=True)
    print(f"  - Parent levels: {len(context['ancestors'])}")
    print(f"  - Siblings: {len(context['siblings'])}")
    print(f"  - Children: {len(context['descendants'])}")


if __name__ == "__main__":
    # Run example if executed directly
    print("Hierarchical Retrieval Utilities")
    print("=" * 50)
    print("\nThis module provides utilities for navigating hierarchical chunks.")
    print("Import and use HierarchicalChunkNavigator and reconstruct_context")
    print("in your retrieval pipelines.\n")
