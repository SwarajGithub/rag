import os

import numpy as np
from sentence_transformers import SentenceTransformer
from utils import DB_Operations, call_llm


class Rag:
    def __init__(self, max_loop_count: int = 3):
        self.embeddings = SentenceTransformer(os.getenv("EMBEDDING_MODEL"))
        self.max_loop_count = max_loop_count


    def collect_context(self, user_question: str, doc_name: str):
        db = DB_Operations(is_age_enabled=True)
        accumulated_context_blocks = []
        seen_chunks = set()
        cited_pages = set()

        current_internal_state = ""
        loop_count = 0

        while loop_count < self.max_loop_count:
            loop_count += 1
            print(f"--- [Loop Iteration {loop_count}/{self.max_loop_count}] ---")
            routing_system_prompt = (
                "You are the central routing engine of an autonomous research agent. Your job is to analyze "
                "the user's master question and the research data gathered so far, then determine the next best step.\n\n"
                "Options:\n"
                "1. If info is missing, output: SEARCH: <highly specific keyword or semantic phrase to look up>\n"
                "2. If you have gathered sufficient context to fully resolve the user's prompt, output: COMPLETE\n\n"
                "Be critical. If the current gathered facts do not explicitly answer the question, issue another SEARCH query."
            )

            routing_user_prompt = (
                f"Master User Question: {user_question}\n\n"
                f"Currently Gathered Information Store:\n{current_internal_state if current_internal_state else '[Empty - Loop Start]'}\n\n"
                f"Determine your next routing step (SEARCH: <phrase> or COMPLETE):"
            )

            decision = call_llm(routing_system_prompt, routing_user_prompt).strip()
            print(f"Agent Decision Strategy: {decision}")

            if "COMPLETE" in decision or not decision.startswith("SEARCH:"):
                print("Agent declared info store complete. Moving to final synthesis...")
                break

            search_query = decision.replace("SEARCH:", "").strip().strip("'\"")
            query_vector = self.embeddings.encode(search_query, convert_to_numpy=True).astype(np.float32).tolist()
            retrieved_chunks = db.get_from_db(query_vector=query_vector, doc_name=doc_name, query=search_query, candidate_pool_limit=3)

            if not retrieved_chunks:
                print("Search yielded zero database matches. Adding warning to state layer.")
                current_internal_state += f"\n- Tried searching for '{search_query}' but found nothing."
                continue

            new_info_added = False
            for chunk in retrieved_chunks:
                chunk_signature = (chunk['page'], chunk['content'][:100])

                if chunk_signature not in seen_chunks:
                    seen_chunks.add(chunk_signature)
                    cited_pages.add(chunk['page'])

                    block = f"[Page {chunk['page']}]: {chunk['content']}"
                    accumulated_context_blocks.append(block)
                    current_internal_state += f"\n- {block}"
                    new_info_added = True

            if not new_info_added:
                print("Redundancy Guard Triggered: This search provided no new information. Breaking loop.")
                break

        return "\n".join(accumulated_context_blocks), cited_pages

    def generate_answer(self, user_question: str, doc_name: str, top_k: int = 3):

        context, cited_pages = self.collect_context(user_question, doc_name)

        synthesis_system_prompt = (
            "You are an expert research assistant. Synthesize a definitive answer to the user's question "
            "using ONLY the provided document context blocks. You must remain strictly grounded. If the "
            "provided context does not contain the answer, state that you do not know based on the document.\n"
            "Always include source page citations (e.g., [Page X]) seamlessly in your response text."
        )

        synthesis_user_prompt = (
            f"Accumulated Document Context:\n{context if context else 'No matching entries found.'}\n\n"
            f"User Question: {user_question}\n\n"
            f"Final Answer:"
        )

        final_answer = call_llm(synthesis_system_prompt, synthesis_user_prompt)

        return final_answer, sorted(list(cited_pages))

