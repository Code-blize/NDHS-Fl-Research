# inspecting the saved partition evidence

 
from pathlib import Path
import pandas as pd

ROOT = Path(
    r"C:\Users\usher\Documents\NDHS-Fl-Research\NDHS_FL_Research"
)

PARTITION_DIR = ROOT / "03_partitions"

print("=" * 70)
print("SW-01: SAVED PARTITION EVIDENCE AUDIT")
print("=" * 70)

# 1. Inspect the saved regional summary
summary_path = PARTITION_DIR / "partition_summary_by_region.csv"

print("\n--- PARTITION SUMMARY BY REGION ---")

if summary_path.exists():
    summary = pd.read_csv(summary_path)
    print("Columns:", summary.columns.tolist())
    print(summary.to_string(index=False))
else:
    print("File not found.")

# 2. Inspect the partitioned baseline and expanded datasets
for filename in [
    "baseline_partitioned.csv",
    "expanded_partitioned.csv"
]:
    path = PARTITION_DIR / filename

    print(f"\n--- {filename} ---")

    if not path.exists():
        print("File not found.")
        continue

    df = pd.read_csv(path)

    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # Show likely partition and grouping fields
    for col in [
        "partition", "fold", "source_row_id",
        "v021", "v024", "facility_delivery"
    ]:
        if col in df.columns:
            print(f"\n{col}:")
            print(df[col].value_counts(dropna=False).to_string())

    if {"partition", "v024", "facility_delivery"}.issubset(df.columns):
        print("\nRegion x outcome by partition:")
        print(
            df.groupby(
                ["partition", "v024", "facility_delivery"]
            ).size().to_string()
        )

# 3. Search parent research folder for alternative split scripts
print("\n--- BROADER PROVENANCE SEARCH ---")

SEARCH_ROOT = ROOT.parent

extensions = {
    ".py", ".ipynb", ".r", ".rmd",
    ".ps1", ".md", ".txt", ".log"
}

terms = [
    "StratifiedGroupKFold",
    "fixed_partition_assignments",
    "fold_assignment",
    "random_state=2026",
    "random_state = 2026"
]

found = 0

for path in SEARCH_ROOT.rglob("*"):
    if not path.is_file() or path.suffix.lower() not in extensions:
        continue

    if any(
        part.lower() in {
            ".venv", "venv", ".git", "__pycache__",
            "site-packages", "node_modules"
        }
        for part in path.parts
    ):
        continue

    try:
        content = path.read_text(
            encoding="utf-8",
            errors="ignore"
        )
    except Exception:
        continue

    matches = [
        term for term in terms
        if term.lower() in content.lower()
    ]

    if matches:
        found += 1
        print(path)
        print("  Matching terms:", matches)

print("\nFiles with matching terms:", found)

print("\n" + "=" * 70)
print("AUDIT COMPLETE — NO FILES MODIFIED")
print("=" * 70)
