import os
import re

RAW_FILE = "raw-data/announcements.txt"
OUTPUT_DIR = "preprocessed-data/announcements"

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(RAW_FILE, "r", encoding="utf-8") as f:
    raw_text = f.read()

announcement_pattern = r"(?=^[^\n]+\nDear students,?)"
announcements = re.split(announcement_pattern, raw_text, flags=re.MULTILINE)
announcements = [a.strip() for a in announcements if a.strip()]


def infer_metadata(text):
    lines = text.splitlines()
    title = next((l.strip() for l in lines if l.strip()), "General Announcement")

    return title[:80], "Toon Calders"


for i, content in enumerate(announcements):
    title, author = infer_metadata(content)

    doc_text = f"[Title] {title}\n[Author] {author}\n\n{content.strip()}"
    filename = f"announcement_{i + 1:02d}.txt"

    with open(os.path.join(OUTPUT_DIR, filename), "w", encoding="utf-8") as f:
        f.write(doc_text)

print(f"Processed {len(announcements)} announcements into: {OUTPUT_DIR}/")
