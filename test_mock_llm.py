"""Test script to verify MockLLM configuration"""

from llama_index.core.llms import MockLLM
from llama_index.core import Settings

# Test MockLLM configuration
mock_llm = MockLLM(max_tokens=256)
print(f"MockLLM created: {mock_llm}")
print(f"Max tokens: {mock_llm.max_tokens}")

# Check if we can access metadata
if hasattr(mock_llm, 'metadata'):
    print(f"Metadata: {mock_llm.metadata}")

# Set global settings
Settings.llm = MockLLM(max_tokens=256)
Settings.context_window = 4096
Settings.num_output = 256

print(f"\nSettings configured:")
print(f"  LLM: {Settings.llm}")
print(f"  Context window: {Settings.context_window}")
print(f"  Num output: {Settings.num_output}")

print("\n✓ MockLLM configuration test passed!")
