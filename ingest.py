import os
import fitz
from PIL import Image

def ingest_pdf(pdf_path, output_image_dir="extracted_images"):
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    doc = fitz.open(pdf_path)
    document_chunks = []

    print("Processing PDF...")
    for page_num in range(len(doc)):
        page = doc[page_num]

        text = page.get_text()
        if text.strip():
            document_chunks.append(
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

            image_filename = os.path.join(output_image_dir, f"page_{page_num+1}_img{img_index}.{image_ext}")
            with open(image_filename, "wb") as f:
                f.write(image_bytes)

            document_chunks.append(
                {
                    "page": page_num + 1,
                    "type": "image",
                    "content": f"[Image from {page_num+1}]({img[0]})]",
                    "image_path": image_filename,
                }
            )

    print("Extraction completed")
    return document_chunks

if __name__ == "__main__":
    pdf_file = "/Users/sshekhar/git1/rag/data/artemis_plan-20200921.pdf"
    chunks = ingest_pdf(pdf_file)
    print(chunks)