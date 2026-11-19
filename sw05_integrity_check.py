# audit how the partitions were created


from pathlib import Path

ROOT = Path(
    r"C:\Users\usher\Documents\NDHS-Fl-Research\NDHS_FL_Research"
)

terms = [
    "fixed_partition_assignments",
    "split_seed",
    "GroupShuffleSplit",
    "StratifiedGroupKFold",
    "GroupKFold",
    "train_test_split",
    "v021",
    "source_row_id",
]

extensions = {".py", ".md", ".txt", ".json", ".ipynb"}

print("=" * 75)
print("SW-01: PARTITION GENERATION PROVENANCE SEARCH")
print("=" * 75)

for path in sorted(ROOT.rglob("*")):
    if not path.is_file() or path.suffix.lower() not in extensions:
        continue

    # Skip virtual environments and Git internals.
    if any(
        part.lower() in {".venv", "venv", ".git", "__pycache__"}
        for part in path.parts
    ):
        continue

    try:
        lines = path.read_text(
            encoding="utf-8",
            errors="ignore"
        ).splitlines()
    except Exception:
        continue

    matches = []

    for i, line in enumerate(lines):
        found = [
            term for term in terms
            if term.lower() in line.lower()
        ]

        if found:
            matches.append((i, line.strip(), found))

    if matches:
        print(f"\nFILE: {path.relative_to(ROOT)}")

        shown = set()

        for line_number, line, found in matches:
            # Show a small context window around each match.
            start = max(0, line_number - 2)
            end = min(len(lines), line_number + 3)

            if any(n in shown for n in range(start, end)):
                continue

            shown.update(range(start, end))

            print(f"\n  Lines {start + 1}-{end}:")
            for n in range(start, end):
                print(f"  {n + 1}: {lines[n][:250]}")

print("\n" + "=" * 75)
print("SEARCH COMPLETE — NO FILES MODIFIED")
print("=" * 75)
