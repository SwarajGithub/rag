import os
from vectorstore import VectorStore
from utils import call_llm

class RAGGenerator:
    def __init__(self):
        self.vector_store = VectorStore()

    def generate_answer(self, question: str, doc_name: str, top_k = 3):
        retrieved_chunks = self.vector_store.search(question, doc_name)
        if not retrieved_chunks:
            return "Not available", []

        context_blocks = []
        citations = set()

        for chunk in retrieved_chunks:
            page = chunk['page']
            content = chunk['content']
            citations.add(page)
            context_blocks.append(f"[Page {page}]: {content}")
            print(chunk)
            print()

        combined_context = "\n".join(context_blocks)

        system_prompt = (
            "You are an expert research assistant. Use ONLY the provided context to answer the user's question. "
            "You must strictly ground your answer in the text. If the answer cannot be found in the context or if you "
            "do not know, you MUST output exactly: 'I don't know based on the provided document.' "
            "Always cite the source page numbers in your response."
        )

        user_prompt = (
            f"Context:\n{combined_context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )

        answer_text = call_llm(system_prompt, user_prompt)
        return answer_text, sorted(list(citations))


