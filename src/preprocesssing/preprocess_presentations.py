import os
import re
import fitz  # PyMuPDF
from tqdm import tqdm

PDF_FOLDER = "raw-data/presentations"
TXT_OUTPUT_FOLDER = "preprocessed-data/presentations"

def extract_slide_text(pdf_path):
    doc = fitz.open(pdf_path)
    slide_texts = []

    for i, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'[ \t]+', ' ', text)
        if text:
            slide = f"[Slide {i}]\n{text}\n"
            slide_texts.append(slide)

    return "\n".join(slide_texts)

def main():
    os.makedirs(TXT_OUTPUT_FOLDER, exist_ok=True)
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

    for filename in tqdm(pdf_files, desc="Processing presentations"):
        pdf_path = os.path.join(PDF_FOLDER, filename)
        base_name = os.path.splitext(filename)[0]
        txt_path = os.path.join(TXT_OUTPUT_FOLDER, base_name + ".txt")

        try:
            extracted = extract_slide_text(pdf_path)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(extracted)
        except Exception as e:
            print(f"[ERROR] Failed to process {filename}: {e}")

    print(f"Processed {len(pdf_files)} PDFs into: {TXT_OUTPUT_FOLDER}/")

if __name__ == "__main__":
    main()