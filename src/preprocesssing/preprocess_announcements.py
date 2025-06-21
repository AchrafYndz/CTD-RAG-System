import os
import re

RAW_FILE = "raw-data/announcements.txt"
OUTPUT_DIR = "preprocessed-data/announcements"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(RAW_FILE, "r", encoding="utf-8") as f:
    raw_text = f.read()
announcements = re.split(r"\n\s*Dear students,?\s*\n", raw_text)
announcements = [a.strip() for a in announcements if a.strip()]


# Helper to guess title/date from text
def infer_metadata(text):
    lines = text.splitlines()
    title = next((l for l in lines if l.strip()), "General Announcement")

    # Try to extract a date (e.g., 20/3, 21/5, etc.)
    date_match = re.search(r"(\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}/\d{1,2})", text)
    date = date_match.group(1) if date_match else "Unknown"

    return title.strip()[:80], "Toon Calders", date


# Process and save each announcement
for i, content in enumerate(announcements):
    title, author, date = infer_metadata(content)

    doc_text = f"[Title] {title}\n[Author] {author}\n[Date] {date}\n\n{content.strip()}"
    filename = f"announcement_{i + 1:02d}.txt"

    with open(os.path.join(OUTPUT_DIR, filename), "w", encoding="utf-8") as f:
        f.write(doc_text)

print(f"✅ Processed {len(announcements)} announcements into: {OUTPUT_DIR}/")
