import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from vectorstore import VectorStore
from rag import RAGGenerator

app = FastAPI(title="RAG Chat With Your PDF API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)

# Initialize VectorStore and RAG Generator
vector_store = VectorStore()
rag_generator = RAGGenerator()

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class ChatRequest(BaseModel):
    query: str
    doc_name: str
    top_k: int = 3


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Uploads a PDF, saves it locally, extracts chunks and images,
    embeds them, and loads them into PostgreSQL pgvector.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Build index and store vectors in PostgreSQL
        vector_store.build_index(file_path)
        doc_name = os.path.basename(file_path).split(".")[0]
        return {
            "status": "success",
            "filename": file.filename,
            "doc_name": doc_name,
            "message": "PDF successfully ingested and embedded into PostgreSQL pgvector!"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat_with_pdf(request: ChatRequest):
    """
    Takes a query and doc_name, retrieves relevant context via pgvector,
    and generates a strictly grounded answer using Qwen with page citations.
    """
    try:
        answer, sources = rag_generator.generate_answer(
            question=request.query,
            doc_name=request.doc_name,
            top_k=request.top_k
        )
        return {
            "query": request.query,
            "doc_name": request.doc_name,
            "answer": answer,
            "citations": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)