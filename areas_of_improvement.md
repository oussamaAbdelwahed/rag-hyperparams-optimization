# Areas of Improvement

1. use semantic chunking instead of fixed-size chunking for better context relevance.
2. couple the semantic chunking with LLM to summarize the retrieved chunks if their total length > LLM context window capacity and then feed that summarized context to the LLM for answer generation.
