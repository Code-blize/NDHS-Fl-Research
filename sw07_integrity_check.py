# verify the original split implementation


from pathlib import Path

ROOT = Path(
    r"C:\Users\usher\Documents\NDHS-Fl-Research\NDHS_FL_Research"
)

terms = [
    "StratifiedGroupKFold(",
    "StratifiedGroupKFold",
    "fixed_partition_assignments.csv",
    "fold_assignment",
    "random_state=2026",
]

print("=" * 70)
print("SW-01: ORIGINAL SPLIT IMPLEMENTATION SEARCH")
print("=" * 70)

for path in sorted(ROOT.rglob("*.py")):
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

    matches = [
        i for i, line in enumerate(lines)
        if any(term.lower() in line.lower() for term in terms)
    ]

    if matches:
        print(f"\nFILE: {path.relative_to(ROOT)}")

        shown = set()

        for i in matches:
            start = max(0, i - 5)
            end = min(len(lines), i + 9)

            if any(n in shown for n in range(start, end)):
                continue

            shown.update(range(start, end))

            for n in range(start, end):
                print(f"{n + 1}: {lines[n]}")

print("\n" + "=" * 70)
print("SEARCH COMPLETE — NO FILES MODIFIED")
print("=" * 70)
