import os

INPUT_PATH = "data/raw/course_information.txt"
OUTPUT_PATH = "data/processed/course_information_cleaned.txt"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

def preprocess_course_info(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        # Remove all blank lines
        lines = [line.strip() for line in f if line.strip()]

    processed = []
    skip_next = False

    for i in range(len(lines)):
        if skip_next:
            skip_next = False
            continue

        current = lines[i]
        if current.endswith(":") and i + 1 < len(lines):
            # Join with next line
            combined = f"{current} {lines[i + 1]}"
            processed.append(combined)
            skip_next = True
        else:
            processed.append(current)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(processed))

    print(f"✅ Course information cleaned and saved to: {output_path}")

if __name__ == "__main__":
    preprocess_course_info(INPUT_PATH, OUTPUT_PATH)
