# inspect the original notebook


from pathlib import Path
import json

NOTEBOOK = Path(
    r"C:\Users\usher\Documents\NDHS-Fl-Research\notebook.ipynb"
)

terms = [
    "StratifiedGroupKFold",
    "fixed_partition_assignments",
    "fold_assignment",
    "random_state",
    "split_seed",
    "GroupKFold",
    "partition_manifest",
]

print("=" * 75)
print("SW-01: ORIGINAL NOTEBOOK PARTITION IMPLEMENTATION")
print("=" * 75)

if not NOTEBOOK.exists():
    print("Notebook not found:", NOTEBOOK)
    raise SystemExit

with open(NOTEBOOK, "r", encoding="utf-8") as f:
    notebook = json.load(f)

cells = notebook.get("cells", [])
found = 0

for index, cell in enumerate(cells):
    source = cell.get("source", [])
    if isinstance(source, list):
        source = "".join(source)

    matches = [
        term for term in terms
        if term.lower() in source.lower()
    ]

    if matches:
        found += 1
        print(f"\n{'=' * 65}")
        print("CELL:", index)
        print("TYPE:", cell.get("cell_type"))
        print("MATCHES:", ", ".join(matches))
        print("-" * 65)
        print(source[:12000])

print("\n" + "=" * 75)
print("MATCHING CELLS:", found)
print("NOTEBOOK CELLS:", len(cells))
print("INSPECTION COMPLETE — NOTEBOOK NOT MODIFIED")
print("=" * 75)
