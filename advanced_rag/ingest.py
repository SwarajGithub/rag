import os
import fitz

def sliding_window_chunking(text: str, chunk_size: int = 600, overlap: int = 100) -> list:
    chunks = []
    start = 0
    cleaned_text = " ".join(text.split())
    if not cleaned_text:
        return []
    while start < len(cleaned_text):
        end = start + chunk_size
        chunks.append(cleaned_text[start:end])
        start = end - overlap
    return chunks

def ingest_pdf(pdf_path: str, output_image_dir: str = "extracted_images"):

    doc_name= os.path.basename(pdf_path)
    doc = fitz.open(pdf_path)
    document_chunks = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        raw_text = page.get_text()

        text_chunks = sliding_window_chunking(raw_text)
        for window in text_chunks:
            document_chunks.append({
                "page": page_num + 1,
                "type": "text",
                "content": window,
                "image_path": "",
            })

            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    image_filename = os.path.join(output_image_dir,
                                                  f"{doc_name}_page_{page_num + 1}_img{img_index}.{image_ext}")
                    with open(image_filename, "wb") as f:
                        f.write(image_bytes)

                    document_chunks.append({
                        "page": page_num + 1,
                        "type": "image",
                        "content": f"[Extracted Graphical Asset from page {page_num + 1}] Name referent: {img[0]}",
                        "image_path": image_filename,
                    })
                except Exception:
                    continue

        print(f"Extraction completed successfully. Size is {len(document_chunks)}")
        return document_chunks



