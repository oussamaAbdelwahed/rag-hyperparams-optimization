# RAG Hyperparameter Optimization

This project implements hyperparameter optimization for Retrieval-Augmented Generation (RAG) systems using LlamaIndex.

Based on the official LlamaIndex tutorial: [Hyperparameter Optimization for RAG](https://developers.llamaindex.ai/python/examples/param_optimizer/param_optimizer/)

## Features

- Automated hyperparameter tuning for RAG systems
- Grid search over chunk size and top-k parameters
- Semantic similarity evaluation
- Both synchronous and asynchronous parameter tuning
- Uses the Llama 2 paper as test data

## Setup

1. **Activate virtual environment:**

```bash
source venv/bin/activate
```

2. **Set OpenAI API Key:**

```bash
export OPENAI_API_KEY='your-api-key-here'
```

Or create a `.env` file:

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

3. **Run the optimization:**

```bash
python main.py
```

## What It Does

The script performs the following:

1. **Downloads Data**: Automatically downloads the Llama 2 paper (PDF) and evaluation dataset
2. **Builds Indexes**: Creates vector indexes with different chunk sizes (256, 512, 1024)
3. **Tests Parameters**: Tests different combinations of:
   - Chunk sizes: 256, 512, 1024
   - Top-k values: 1, 2, 5
4. **Evaluates Performance**: Uses semantic similarity to evaluate response quality
5. **Finds Best Parameters**: Identifies the optimal parameter combination

## Parameters Being Optimized

- **chunk_size**: The size of text chunks for document splitting
  - Options: 256, 512, 1024 tokens
- **top_k**: Number of top relevant chunks to retrieve
  - Options: 1, 2, 5 chunks

## Output

The script will output:

- Progress during parameter tuning
- Best score achieved
- Optimal top-k value
- Optimal chunk size

Example output:

```
==================================================
SYNC PARAMETER TUNING RESULTS
==================================================
Score: 0.9521222054806685
Top-k: 2
Chunk size: 512
==================================================
```

## Project Structure

```
rag-hyperparams-optimization/
├── venv/                  # Virtual environment
├── data/                  # Downloaded data (auto-created)
│   ├── llama2.pdf
│   └── llama2_eval_qr_dataset.json
├── storage_*/             # Index storage (auto-created)
├── main.py               # Main optimization script
├── requirements.txt      # Python dependencies
├── .env.example         # Example environment variables
└── README.md            # This file
```

## Dependencies

All dependencies are listed in `requirements.txt` and include:

- llama-index-core
- llama-index-llms-openai
- llama-index-embeddings-openai
- llama-index-readers-file
- llama-index-experimental
- pymupdf
- And many more...

## Notes

- The script uses only 10 evaluation samples by default for faster experimentation
- Adjust `num_samples` in the code to use more evaluation data
- The async version is commented out but can be enabled
- Storage directories are created automatically for each chunk size
- Indexes are cached, so subsequent runs with the same chunk size will be faster

## Troubleshooting

**Import errors**: Make sure you've activated the virtual environment and installed all dependencies:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

**Missing API key**: Set the OPENAI_API_KEY environment variable before running

**Download issues**: If automatic downloads fail, manually download:

- Paper: https://arxiv.org/pdf/2307.09288.pdf
- Dataset: https://www.dropbox.com/scl/fi/fh9vsmmm8vu0j50l3ss38/llama2_eval_qr_dataset.json?rlkey=kkoaez7aqeb4z25gzc06ak6kb&dl=1
