from vectorstore import VectorStore
from utils import call_llm
from sentence_transformers import CrossEncoder

class RAGGenerator:
    def __init__(self):
        encoder  = "BAAI/bge-reranker-base"
        self.vectorstore = VectorStore()
        self.reranker = CrossEncoder(encoder)

    def rewrite_query(self, question: str) -> str:
        system_prompt = (
            "You are an expert search engine optimizer. Rewrite the user's question into an optimized, "
            "expanded search query containing alternative terms and core keywords found in technical documents. "
            "Output ONLY the final rewritten text phrase."
            "Do not ask back questions, this is not a chat agent. Generate a response."
        )
        user_prompt = f"Original Question: {question}\nRewritten Query:"
        try:
            return call_llm(system_prompt, user_prompt, max_tokens=1536).strip()
        except Exception:
            return question

    def generate_answer(self, question: str, doc_name: str, top_k=5):
        optimized_query = self.rewrite_query(question)
        print(f"Optimized query is {optimized_query}")
        candidates = self.vectorstore.hybrid_search_top_k(optimized_query, doc_name, candidate_pool_limit=20)
        if not candidates:
            return "Not available", []
        pairs = [[question, item['content']] for item in candidates]
        scores = self.reranker.predict(pairs)
        for idx, score in enumerate(scores):
            candidates[idx]['rerank_score'] = float(score)
        reranked_chunks = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)[:top_k]
        context_blocks = []
        citations = set()
        for chunk in reranked_chunks:
            page = chunk['page']
            citations.add(page)
            context_blocks.append(f"[Page {page}]: {chunk['content']}")

        combined_context = "\n".join(context_blocks)

        system_prompt = (
            "You are an expert research assistant. Use ONLY the provided context to answer the user's question. "
            "You must strictly ground your answer in the text. If the answer cannot be found in the context or if you "
            "do not know, you MUST output exactly: 'I don't know based on the provided document.' "
            "Always cite the source page numbers in your response."
            "Do not ask back questions, this is not a chat agent. Generate a response."
        )

        user_prompt = f"Context:\n{combined_context}\n\nQuestion: {question}\n\nAnswer:"
        answer_text = call_llm(system_prompt, user_prompt, max_tokens=3592)

        return answer_text, sorted(list(citations))