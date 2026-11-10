import os

import numpy as np
from sentence_transformers import SentenceTransformer
from utils import DB_Operations, call_llm


class Rag:
    def __init__(self):
        self.embeddings = SentenceTransformer(os.getenv("EMBEDDING_MODEL"))

    def generate_answer(self, question: str, doc_name: str, top_k: int = 3):
        query_vector = self.embeddings.encode(question, convert_to_numpy=True).astype(np.float32).tolist()

        db = DB_Operations(is_age_enabled=True)

        # 2. Extract baseline text matches via traditional Vector HNSW Index
        results = db.get_from_db(query_vector, doc_name=doc_name, query=question)

        chunk_ids = [row['page'] for row in results]
        text_contexts = [row['content'] for row in results]
        citations = sorted({row['page'] for row in results})

        graph_contexts = []
        if chunk_ids:
            chunk_ids_str = ", ".join(map(str, chunk_ids))
            graph_contexts = db.query_graph(chunk_ids_str)

        context_str = "\n--- Semantic Text Snippets ---\n" + "\n".join(text_contexts)
        if graph_contexts:
            context_str += "\n--- Knowledge Network Structural Links ---\n" + "\n".join(graph_contexts)

        system_prompt = f"""
        Answer the user question using both text snippets and relational network links retrieved below:"""

        user_prompt = f"""
        Context Data:
        {context_str}

        Question: {question}
        Answer:
        """

        answer_text = call_llm(system_prompt, user_prompt, 4096)
        return answer_text, citations

