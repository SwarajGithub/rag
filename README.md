# Goal

**Build a RAG Pipeline and chat with your PDF**

## Learnings
* Parsing
* Chunking
* Embedding
* Vector DB
* LLM Integration

## Steps

### Step 1: PDF Selection
* NASA Artemis Plan
* Python 3.13 Release Notes

### Step 2: Architecture & Tech
| Task                                               | Tech                                    |
|----------------------------------------------------|-----------------------------------------|
| PDF Parsing                                        | pymupdf, Pillow                         |
| Chunking | LangChain                               |
| Embeddings | sentence-transformers(all-MiniLM-L6-v2) |
| Vector Database | Pgvector                                |
| Search | FAISS                                   |
| LLM Integration | Ollama(Local)                           |
| API | FastAPI                                 |
| UI | HTML                                    |

### Step 3: Implementation
| Task | File | Description                                  |
|----|-----|----------------------------------------------|
| Ingestion and Chunking | ingest.py | Load the PDF, extract and chunk with metadata |
| Embedding | vectorstore.py | sentence-transformer and pgvector       |
| RAG and Grounding | rag.py | connect to LLM |
| APIs | main.py | fast API for upload and chat |
| UI | ui | UI for this |


## Run

### DB

```
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    document_name TEXT,
    page INT,
    chunk_type TEXT, -- 'text' or 'image'
    content TEXT,
    image_path TEXT,
    embedding vector(384) -- Dimension matches all-MiniLM-L6-v2 (384)
);
```

```
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt 
```

* Copy .env.example as .env and populate

```
source .env
python main.py
```

* Open app.html in browser

## NOTE
* This is not a chat application
* Contexts are not saved after each chat but each question is independent.