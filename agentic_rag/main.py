import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from agentic_rag.vectorstore import VectorStore
from agentic_rag.rag import Rag

app = FastAPI(title="Graph RAG Chat API Engine", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "../data"
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
        vc = VectorStore(file_path)
        vc.ingest_pdf(file_path)
        doc_name = vc.doc_name
        return {
            "status": "success",
            "filename": file.filename,
            "doc_name": doc_name,
            "message": "PDF successfully ingested using Graph RAG!"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_with_pdf(request: ChatRequest):
    rag = Rag()
    try:
        answer, sources = rag.generate_answer(
            user_question=request.query,
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
