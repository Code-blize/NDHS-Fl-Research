from pathlib import Path
import json
import numpy as np

ROOT = Path(
    r"C:\Users\usher\Documents\NDHS-Fl-Research\NDHS_FL_Research"
)

PACKAGE = ROOT / "09_flower_implementation_v1_2_1"
OUTPUTS = PACKAGE / "outputs" / "confirmatory_v2_seeds_42_51"

TEST_FILES = [
    ROOT / "05_preprocessed" / "baseline_test.npz",
    ROOT / "05_preprocessed" / "expanded_test.npz",
]

print("=" * 70)
print("SW-01: READ-ONLY INTEGRITY DIAGNOSTIC")
print("=" * 70)

# 1. Inspect test archives
for path in TEST_FILES:
    print(f"\n--- TEST ARCHIVE: {path.name} ---")

    if not path.exists():
        print("MISSING:", path)
        continue

    try:
        with np.load(path, allow_pickle=False) as data:
            print("Archive keys:", data.files)

            for key in data.files:
                arr = data[key]
                print(
                    f"{key}: shape={arr.shape}, "
                    f"dtype={arr.dtype}"
                )

                if key.lower() in ("source_row_id", "source_ids"):
                    print("  Unique IDs:", len(np.unique(arr)))
                    print("  Total IDs:", len(arr))
                    print("  All IDs unique:", len(np.unique(arr)) == len(arr))

    except Exception as exc:
        print("ERROR reading archive:", repr(exc))


# 2. Inspect confirmatory run folders and manifests
print("\n--- CONFIRMATORY OUTPUT INVENTORY ---")

if not OUTPUTS.exists():
    print("OUTPUT DIRECTORY MISSING:", OUTPUTS)
else:
    checkpoints = list(OUTPUTS.rglob("final_model_state_dict.pt"))
    manifests = list(OUTPUTS.rglob("run_manifest.json"))

    print("Output folders:", sum(p.is_dir() for p in OUTPUTS.iterdir()))
    print("Checkpoint count:", len(checkpoints))
    print("Manifest count:", len(manifests))

    print("\n--- RUN MANIFEST DETAILS ---")

    for manifest_path in sorted(manifests):
        print(f"\nManifest: {manifest_path.parent.name}")

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            print("Available fields:", list(manifest.keys()))

            # Display likely experiment configuration fields.
            wanted = (
                "seed",
                "algorithm",
                "feature_config",
                "local_epochs",
                "num_server_rounds",
                "status",
                "success",
                "completed",
            )

            for key in wanted:
                if key in manifest:
                    print(f"  {key}: {manifest[key]}")

        except Exception as exc:
            print("ERROR reading manifest:", repr(exc))

        checkpoint = manifest_path.parent / "final_model_state_dict.pt"
        print("Matching checkpoint exists:", checkpoint.exists())


# 3. Check expected metadata files
print("\n--- SURVEY METADATA FILES ---")

metadata_files = [
    ROOT / "02_clean_datasets" / "survey_design_metadata.csv",
    ROOT / "03_partitions" / "fixed_partition_assignments.csv",
    ROOT / "05_preprocessed" / "preprocessing_manifest.json",
]

for path in metadata_files:
    print(path.name, "EXISTS" if path.exists() else "MISSING")


print("\n" + "=" * 70)
print("DIAGNOSTIC FINISHED — NO FILES MODIFIED")
print("=" * 70)
