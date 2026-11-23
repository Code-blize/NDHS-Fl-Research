from pathlib import Path
import json
from collections import Counter

ROOT = Path(
    r"C:\Users\usher\Documents\NDHS-Fl-Research\NDHS_FL_Research"
)

OUT = (
    ROOT
    / "09_flower_implementation_v1_2_1"
    / "outputs"
    / "confirmatory_v2_seeds_42_51"
)

print("Confirmatory directory exists:", OUT.exists())

if not OUT.exists():
    raise SystemExit("STOP: Confirmatory output directory not found.")

run_dirs = sorted(p for p in OUT.iterdir() if p.is_dir())
manifests = sorted(OUT.rglob("run_manifest.json"))
checkpoints = sorted(OUT.rglob("final_model_state_dict.pt"))

print("\n=== FILE COUNTS ===")
print("Run directories:", len(run_dirs))
print("Run manifests:", len(manifests))
print("Final checkpoints:", len(checkpoints))

# Check each run directory for its expected files.
print("\n=== PER-RUN FILE CHECK ===")
missing_manifest = []
missing_checkpoint = []

for folder in run_dirs:
    if not (folder / "run_manifest.json").is_file():
        missing_manifest.append(folder.name)
    if not (folder / "final_model_state_dict.pt").is_file():
        missing_checkpoint.append(folder.name)

print("Runs missing manifest:", missing_manifest)
print("Runs missing checkpoint:", missing_checkpoint)

# Read manifests without modifying them.
records = []
errors = []

for path in manifests:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        records.append((path, data))
    except Exception as exc:
        errors.append((str(path), repr(exc)))

print("\n=== MANIFEST READ CHECK ===")
print("Successfully parsed:", len(records))
print("Read/JSON errors:", errors)

# Compare top-level manifest structures.
key_counts = Counter(
    tuple(sorted(data.keys()))
    for _, data in records
)

print("\n=== MANIFEST STRUCTURES ===")
for keys, count in key_counts.items():
    print(f"\nStructure occurs {count} times:")
    print(list(keys))

# Show one representative manifest in full.
if records:
    sample_path, sample_data = records[0]

    print("\n=== REPRESENTATIVE MANIFEST ===")
    print("Path:", sample_path)
    print(json.dumps(sample_data, indent=2, ensure_ascii=False))

# Show all run folder names.
print("\n=== CONFIRMATORY RUN FOLDERS ===")
for folder in run_dirs:
    print(folder.name)

print("\nAudit complete. No files modified.")