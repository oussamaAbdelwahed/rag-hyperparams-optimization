# Ollama Docker Setup

## Quick Start

### 1. Start Ollama service

```bash
docker-compose up -d
```

### 2. Pull a chat model (choose one)

```bash
# Small, fast model (~2GB) - Recommended for testing
docker-compose exec ollama ollama pull llama3.2:3b

# Or smaller model (~1.6GB)
docker-compose exec ollama ollama pull phi3:mini

# Or larger, more capable model (~4.7GB)
docker-compose exec ollama ollama pull llama3.1:8b
```

### 3. Verify model is ready

```bash
docker-compose exec ollama ollama list
```

### 4. Run your optimization script

```bash
source venv/bin/activate
python ats_async_find_opt_params_sync_pinecone.py
```

## Management Commands

### Check service status

```bash
docker-compose ps
```

### View logs

```bash
docker-compose logs -f ollama
```

### Stop service

```bash
docker-compose down
```

### Stop and remove models (free up disk space)

```bash
docker-compose down -v
```

## Model Recommendations

- **llama3.2:3b** - Best balance of speed/quality for RAG (2GB)
- **phi3:mini** - Fastest, smallest (1.6GB)
- **llama3.1:8b** - Higher quality responses (4.7GB)

## GPU Support

If you have NVIDIA GPU, uncomment the `deploy` section in docker-compose.yml for faster inference.

## Notes

- Ollama API endpoint: `http://localhost:11434`
- Models are persisted in Docker volume `ollama-models`
- The Python script connects to localhost:11434 automatically
