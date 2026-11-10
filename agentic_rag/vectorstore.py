import os
from sentence_transformers import SentenceTransformer
import fitz

from utils import DB_Operations
from agentic_rag.ingestion import extract_entities_and_relationship

class VectorStore:

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.doc_name = os.path.basename(pdf_path).split(".")[0]

    def ingest_document(self, document_id, chunk, embedding_model = SentenceTransformer(os.getenv("EMBEDDING_MODEL"))):
        text = chunk['content']
        vector_embedding = embedding_model.encode(text, show_progress_bar=True)
        db = DB_Operations(is_age_enabled=True)
        chunk_id = db.insert_in_db(document_id, chunk, vector_embedding)

        graph_payload = extract_entities_and_relationship(text)
        try:
            db.execute_graph_insert(graph_payload, chunk_id)
        except Exception as e:
            print(f"Graph insert failed for chunk {chunk_id}: {e}")

    def ingest_pdf(self, pdf_path, output_image_dir="extracted_images"):
        if not os.path.exists(output_image_dir):
            os.makedirs(output_image_dir)

        doc = fitz.open(pdf_path)
        document_chunks = []
        doc_name = os.path.basename(pdf_path).split(".")[0]

        print("Processing PDF...")
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_chunks = []

            text = page.get_text()
            if text.strip():
                page_chunks.append(
                    {
                        "page": page_num + 1,
                        "type": "text",
                        "content": text.strip(),
                        "image_path": "",
                    }
                )

            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                image_filename = os.path.join(output_image_dir,
                                              f"{doc_name}_page_{page_num + 1}_img{img_index}.{image_ext}")
                with open(image_filename, "wb") as f:
                    f.write(image_bytes)

                page_chunks.append(
                    {
                        "page": page_num + 1,
                        "type": "image",
                        "content": f"[Image from {page_num + 1}]({img[0]})]",
                        "image_path": image_filename,
                    }
                )

            for chunk in page_chunks:
                self.ingest_document(self.doc_name, chunk)
            document_chunks.extend(page_chunks)
        print("Extraction completed")
        return document_chunks

