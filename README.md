# f1-chatbot

A local LLM-powered Formula 1 chatbot built with Retrieval-Augmented Generation (RAG).

This project:
- Uses a FastAPI backend
- Uses ChromaDB for vector retrieval
- Uses sentence-transformers for embeddings
- Uses a local GGUF model via `llama-cpp-python`
- Is designed for later deployment on low-resource devices

## Project Structure

```text
f1-chatbot/
├── backend/
│   ├── server.py
│   ├── rag.py
│   └── ingest.py
├── data/
│   └── *.txt
├── embeddings/
├── model/
├── frontend/
│   └── index.html
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Ingestion

Index all files in `data/*.txt` into Chroma:

```bash
python -m backend.ingest
```

Embeddings are stored in `embeddings/chroma`.

## Model Setup
    
Place a GGUF model file in `model/` (or set `F1_MODEL_PATH` to an absolute/relative GGUF path).

## Run

```bash
uvicorn backend.server:app --reload
```

Open:
- App: `http://127.0.0.1:8000`
- Health: `http://127.0.0.1:8000/health`

## API

- `POST /ask` with `{ "question": "..." }`
- `GET /health`
- `POST /reload`

The bot is domain-restricted and refuses non-F1 questions.
