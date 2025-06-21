import os
import re
import fitz  # PyMuPDF
from tqdm import tqdm


PDF_FOLDER = "raw-data/papers"
TXT_OUTPUT_FOLDER = "preprocessed-data/papers"

def extract_text_with_placeholders(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""

    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text()
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if "image" in block:
                page_text += f"\n[IMAGE: Figure on page {page_num}]\n"
            elif block["type"] == 0:
                text = block.get("text", "")
                if re.search(r'table\s*\d+', text, re.IGNORECASE):
                    page_text += f"\n[TABLE: Table on page {page_num}]\n"

        full_text += page_text + "\n"

    return full_text

def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def process_pdf_file(file_path):
    raw_text = extract_text_with_placeholders(file_path)
    cleaned = clean_text(raw_text)
    return cleaned

def main():
    os.makedirs(TXT_OUTPUT_FOLDER, exist_ok=True)

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

    for filename in tqdm(pdf_files, desc="Processing PDFs"):
        file_path = os.path.join(PDF_FOLDER, filename)
        try:
            cleaned_text = process_pdf_file(file_path)
            base_name = os.path.splitext(filename)[0]
            txt_path = os.path.join(TXT_OUTPUT_FOLDER, base_name + ".txt")

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)

        except Exception as e:
            print(f"[ERROR] Failed to process {filename}: {e}")

    print(f"\n✅ Done! Processed {len(pdf_files)} PDFs.")
    print(f"📄 Output saved to: {TXT_OUTPUT_FOLDER}/")

if __name__ == "__main__":
    main()
