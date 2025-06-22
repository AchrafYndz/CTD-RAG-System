import os
import re
from collections import defaultdict

RAW_FILE = "raw-data/announcements.txt"
OUTPUT_DIR = "preprocessed-data/announcements"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load full announcement file
with open(RAW_FILE, "r", encoding="utf-8") as f:
    raw_text = f.read()

# Split on "Dear students" with preceding title
announcement_pattern = r"(?=^[^\n]+\nDear students,?)"
announcements = re.split(announcement_pattern, raw_text, flags=re.MULTILINE)
announcements = [a.strip() for a in announcements if a.strip()]


def infer_metadata(text):
    lines = text.splitlines()
    title = next((l.strip() for l in lines if l.strip()), "General Announcement")
    return title[:80], "Toon Calders"


def write_group_summary_batch1(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.splitlines()
    try:
        start_idx = next(i for i, line in enumerate(lines) if line.strip() == "Best regards,")
    except StopIteration:
        start_idx = 0

    schedule_lines = lines[start_idx + 1:]
    schedule_lines = [line.strip() for line in schedule_lines if line.strip() and line.strip() != "Toon Calders"]

    date_pattern = re.compile(r"^\d{1,2}/\d{1,2}$|^TBD:?$", re.IGNORECASE)
    date = None
    groups = []
    current_entries = []

    for line in schedule_lines:
        if date_pattern.match(line):
            if current_entries:
                groups.append((date, current_entries))
                current_entries = []
            date = line
        else:
            current_entries.append(line)
    if current_entries:
        groups.append((date, current_entries))

    paper_groups = defaultdict(lambda: {"date": None, "members": []})
    for date, entries in groups:
        for i in range(0, len(entries) - 1, 2):
            person = entries[i]
            paper = entries[i + 1]
            key = (paper, date)
            paper_groups[key]["members"].append(person)
            paper_groups[key]["date"] = date

    output_lines = ["[Title] Group assignments first presentation\n"]
    for idx, ((paper, date), info) in enumerate(paper_groups.items(), start=1):
        output_lines.append(f"[Group] {idx} — Presentation Date: {date}")
        output_lines.append("Members:")
        for member in info["members"]:
            output_lines.append(f"- {member}")
        output_lines.append("Paper:")
        output_lines.append(paper)
        output_lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))


def write_group_summary_batch2(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.splitlines()
    try:
        start_idx = next(i for i, line in enumerate(lines) if line.strip() == "Best regards,")
    except StopIteration:
        start_idx = 0

    list_lines = lines[start_idx + 1:]
    list_lines = [line.strip() for line in list_lines if line.strip() and line.strip() not in {"Toon Calders", "---"}]

    paper_groups = []
    current_paper = None
    current_students = []

    for line in list_lines:
        if re.match(r"^[A-Z].{10,}", line) and len(line.split()) > 4:
            if current_paper and current_students:
                paper_groups.append((current_paper, current_students))
            current_paper = line
            current_students = []
        else:
            current_students.append(line)
    if current_paper and current_students:
        paper_groups.append((current_paper, current_students))

    output_lines = ["[Title] Group assignments second presentation (video)\n"]
    for idx, (paper, members) in enumerate(paper_groups, start=1):
        output_lines.append(f"[Group] {idx} — Presentation Date: 2/4")
        output_lines.append("Members:")
        for member in members:
            output_lines.append(f"- {member}")
        output_lines.append("Paper:")
        output_lines.append(paper)
        output_lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

def write_commenting_assignments(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.splitlines()
    try:
        start_idx = next(i for i, line in enumerate(lines) if line.strip() == "Best regards,")
    except StopIteration:
        start_idx = 0

    data_lines = lines[start_idx + 1:]
    data_lines = [line.strip() for line in data_lines if line.strip() and line.strip() not in {"Toon Calders"}]

    paper_assignments = []
    current_paper = None
    current_reviewers = []

    for line in data_lines:
        # Heuristic: long lines = paper title
        if re.match(r"^[A-Z].{10,}", line) and len(line.split()) > 4:
            if current_paper and current_reviewers:
                paper_assignments.append((current_paper, current_reviewers))
            current_paper = line
            current_reviewers = []
        else:
            # Combine split names if needed
            name = line.replace("\t", " ").strip()
            current_reviewers.append(name)

    if current_paper and current_reviewers:
        paper_assignments.append((current_paper, current_reviewers))

    output_lines = ["[Title] Assignments for commenting or commenting and posing one question for a presentation video of batch 2\n"]

    for idx, (paper, reviewers) in enumerate(paper_assignments, start=1):
        output_lines.append(f"[Assignment] {idx}")
        output_lines.append("Paper:")
        output_lines.append(paper)
        output_lines.append("Assigned to:")
        for reviewer in reviewers:
            output_lines.append(f"- {reviewer}")
        output_lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"✅ Commenting assignments written to: {output_path}")

def write_poster_schedule(input_path, output_path):
    import re
    from collections import defaultdict

    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.splitlines()
    try:
        start_idx = next(i for i, line in enumerate(lines) if line.strip() == "Best regards,")
    except StopIteration:
        start_idx = 0

    data_lines = lines[start_idx + 1:]
    data_lines = [line.strip() for line in data_lines if line.strip().lower() not in {
        "toon", "group", "name", "poster presentation date"
    }]

    # Grouping logic
    schedule = defaultdict(lambda: {"members": [], "raw_slot": None})

    i = 0
    while i + 2 < len(data_lines):
        group = data_lines[i]
        first = data_lines[i + 1]
        last = data_lines[i + 2]
        full_name = f"{first} {last}"

        # Look ahead for date
        if i + 3 < len(data_lines):
            next_line = data_lines[i + 3]
            if re.match(r"\d{1,2}/\d{1,2};", next_line):
                schedule[group]["raw_slot"] = next_line
                i += 4
            else:
                i += 3
        else:
            i += 3

        schedule[group]["members"].append(full_name)

    # Formatter
    def split_slot(slot):
        if not slot:
            return "TBD", "TBD", "TBD"
        parts = [p.strip() for p in slot.split(";")]
        date = parts[0] if len(parts) > 0 else "TBD"
        time = parts[1] if len(parts) > 1 else "TBD"
        room = parts[2] if len(parts) > 2 else "TBD"
        return date, time, room

    # Write output
    output_lines = ["[Title] Poster (and third) presentation schedule\n"]

    for group_num, entry in schedule.items():
        date, time, room = split_slot(entry["raw_slot"])
        output_lines.append(f"[Group] {group_num}")
        output_lines.append(f"Date: {date}")
        output_lines.append(f"Time: {time}")
        output_lines.append(f"Room: {room}")
        output_lines.append("Members:")
        for member in entry["members"]:
            output_lines.append(f"- {member}")
        output_lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"✅ Poster schedule written to: {output_path}")

# Process announcements and write .txt files
for i, content in enumerate(announcements):
    title, author = infer_metadata(content)
    clean_lines = [line.strip() for line in content.splitlines() if line.strip()]
    cleaned_text = "\n".join(clean_lines)

    doc_text = f"[Title] {title}\n[Author] {author}\n\n{cleaned_text}"
    filename = f"announcement_{i + 1:02d}.txt"
    txt_path = os.path.join(OUTPUT_DIR, filename)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(doc_text)

# Directly post-process specific announcements
write_group_summary_batch1(
    input_path=os.path.join(OUTPUT_DIR, "announcement_02.txt"),
    output_path=os.path.join(OUTPUT_DIR, "announcement_02_groups.txt")
)

write_group_summary_batch2(
    input_path=os.path.join(OUTPUT_DIR, "announcement_06.txt"),
    output_path=os.path.join(OUTPUT_DIR, "announcement_06_groups.txt")
)

write_commenting_assignments(
    input_path="preprocessed-data/announcements/announcement_07.txt",
    output_path="preprocessed-data/announcements/announcement_07_assignments.txt"
)

write_poster_schedule(
    input_path="preprocessed-data/announcements/announcement_13.txt",
    output_path="preprocessed-data/announcements/announcement_13_schedule.txt"
)

print(f"✅ Processed {len(announcements)} announcements into: {OUTPUT_DIR}/")
