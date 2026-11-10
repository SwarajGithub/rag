import os
import numpy as np
from sentence_transformers import SentenceTransformer
from ingest import ingest_pdf
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
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
        db = DB_Operations()
        try:
            db.delete_from_db(doc_name=doc_name)
            print("Doc was existing, deleted it.")
        except:
            pass
        for chunk, emb in zip(chunks, embeddings):
            db.insert_in_db(doc_name=doc_name, chunk=chunk, embeddings=emb)

    def hybrid_search_top_k(self, query: str, doc_name: str, candidate_pool_limit: int = 20):
        """Executes a combined semantic and text search using RRF optimization."""
        query_vector = self.model.encode(query, convert_to_numpy=True).astype(np.float32).tolist()
        db = DB_Operations()
        return db.get_from_db(query_vector, doc_name, query, candidate_pool_limit)