import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from vectorstore import VectorStore
from adv_rag import RAGGenerator

app = FastAPI(title="Advanced RAG Chat API Engine", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vector_store = VectorStore()
rag_generator = RAGGenerator()

UPLOAD_DIR = "/Users/sshekhar/git1/rag/data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class ChatRequest(BaseModel):
    query: str
    doc_name: str
    top_k: int = 5

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print("PDF successfully uploaded")
    try:
        vector_store.build_index(file_path)
        doc_name = os.path.basename(file_path).split(".")[0]
        return {
            "status": "success",
            "filename": file.filename,
            "doc_name": doc_name,
            "message": "PDF successfully ingested using Sliding Windows and optimized via PostgreSQL TSVECTOR!"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_with_pdf(request: ChatRequest):
    try:
        answer, sources = rag_generator.generate_answer(
            question=request.query,
            doc_name=request.doc_name,
            top_k=request.top_k
        )
        return {
            "answer": answer,
            "citations": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
