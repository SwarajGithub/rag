import os
import psycopg
import numpy as np
from sentence_transformers import SentenceTransformer
from traditional_rag.ingest import ingest_pdf
from utils import DB_Operations

class VectorStore:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_embedding_dimension()

    def build_index(self, pdf_path: str):
        chunks = ingest_pdf(pdf_path)
        doc_name = os.path.basename(pdf_path).split(".")[0]
        if not chunks:
            return

        texts = [chunk["content"] for chunk in chunks]

        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        embeddings = embeddings.astype(np.float32)

        db = DB_Operations()
        db.delete_from_db(doc_name)
        db.insert_in_db(doc_name, chunks, embeddings)



    def search(self, query: str, doc_name: str, top_k = 3):

        query_vector = self.model.encode(query, show_progress_bar=True, convert_to_numpy=True).astype(np.float32).tolist()
        db = DB_Operations()

        return db.get_from_db(doc_name, query_vector, query=query, candidate_pool_limit = top_k)

if __name__ == "__main__":
    pdf_path = "/data/artemis_plan-20200921.pdf"
    vectorstore = VectorStore()

    # Store
    vectorstore.build_index(pdf_path)

    # Search
    doc_name = os.path.basename(pdf_path).split(".")[0]
    question = "What is mission timeline?"
    answers  = vectorstore.search(query=question, doc_name=doc_name)
    for i, answer in enumerate(answers):
        print(i, answer)