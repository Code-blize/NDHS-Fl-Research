# inspect the partition manifest


from pathlib import Path
import json

ROOT = Path(
    r"C:\Users\usher\Documents\NDHS-Fl-Research\NDHS_FL_Research"
)

PARTITION_DIR = ROOT / "03_partitions"
manifest_path = PARTITION_DIR / "partition_manifest.json"

print("=" * 70)
print("SW-01: PARTITION MANIFEST AND PROVENANCE")
print("=" * 70)

print("\n--- COMPLETE PARTITION MANIFEST ---")

with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

print(json.dumps(manifest, indent=2))

print("\n--- FILES IN PARTITION DIRECTORY ---")

for path in sorted(PARTITION_DIR.iterdir()):
    if path.is_file():
        print(path.name, "-", path.stat().st_size, "bytes")

print("\n--- SPLIT-RELATED MANIFEST FIELDS ---")

for key, value in manifest.items():
    if any(term in key.lower() for term in [
        "split", "seed", "fold", "strat", "group",
        "train", "test", "valid", "random"
    ]):
        print(f"{key}: {value}")

print("\n" + "=" * 70)
print("INSPECTION COMPLETE — NO FILES MODIFIED")
print("=" * 70)
