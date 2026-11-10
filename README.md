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
CREATE EXTENSION IF NOT EXISTS age;

CREATE TABLE IF NOT EXISTS kb_advanced_chunks (
            id SERIAL PRIMARY KEY,
            document_id VARCHAR(255) NOT NULL,
            page_num INT NOT NULL DEFAULT 1,
            content_type VARCHAR(50) NOT NULL DEFAULT 'text',
            image_path TEXT DEFAULT NULL,
            content TEXT NOT NULL,
            embedding VECTOR(384),
            tsv_content TSVECTOR
        );
        
        CREATE INDEX IF NOT EXISTS idx_kb_chunks_tsv ON kb_advanced_chunks USING GIN(tsv_content);
        CREATE INDEX IF NOT EXISTS idx_kb_chunks_embedding ON kb_advanced_chunks USING hnsw (embedding vector_cosine_ops)
    
```

```
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt 
```

* Copy .env.example as .env and populate
* Go to the folder which needs to be run(tested)

```
source .env
python main.py
```

* Open app.html in browser

## NOTE
* This is not a chat application
* Contexts are not saved after each chat but each question is independent.