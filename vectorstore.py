import os
import psycopg
import numpy as np
from sentence_transformers import SentenceTransformer
from ingest import ingest_pdf
from utils import get_db_config

class VectorStore:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_embedding_dimension()
        self.db_config = get_db_config()
        self.conn = psycopg.connect(**self.db_config)

    def build_index(self, pdf_path: str):
        chunks = ingest_pdf(pdf_path)
        doc_name = os.path.basename(pdf_path).split(".")[0]
        if not chunks:
            return

        texts = [chunk["content"] for chunk in chunks]

        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        embeddings = embeddings.astype(np.float32)

        with self.conn.cursor() as cursor:
            cursor.execute('''
            DELETE FROM document_chunks WHERE document_name = %s;
            ''', (doc_name,))

            for chunk, emb in zip(chunks, embeddings):
                emb_list = emb.astype(float).tolist()
                cursor.execute('''
                INSERT INTO document_chunks (document_name, page, chunk_type, content, image_path, embedding) VALUES (%s, %s, %s, %s, %s, %s);'''
                               , (doc_name, chunk["page"], chunk["type"], chunk["content"], chunk["image_path"], emb_list,)
                               )

        self.conn.commit()

    def search(self, query: str, doc_name: str, top_k = 3):

        query_vector = self.model.encode(query, show_progress_bar=True, convert_to_numpy=True).astype(np.float32).tolist()

        with self.conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT page, chunk_type, content, image_path, (embedding <-> %s::vector) AS distance
                FROM document_chunks
                WHERE document_name = %s
                ORDER BY distance ASC
                LIMIT %s;
                """,
                (str(query_vector), doc_name, top_k)
            )
            rows = cursor.fetchall()
        results = []
        for row in rows:
            results.append({
                "page": row[0],
                "type": row[1],
                "content": row[2],
                "image_path": row[3],
                "score": float(row[4])
            })
        return results

if __name__ == "__main__":
    pdf_path = "/Users/sshekhar/git1/rag/data/artemis_plan-20200921.pdf"
    vectorstore = VectorStore()

    # Store
    vectorstore.build_index(pdf_path)

    # Search
    doc_name = os.path.basename(pdf_path).split(".")[0]
    question = "What is mission timeline?"
    answers  = vectorstore.search(query=question, doc_name=doc_name)
    for i, answer in enumerate(answers):
        print(i, answer)