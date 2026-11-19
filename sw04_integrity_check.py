# Inspecting the preprocessing and partition design


from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(
    r"C:\Users\usher\Documents\NDHS-Fl-Research\NDHS_FL_Research"
)

MANIFEST = ROOT / "05_preprocessed" / "preprocessing_manifest.json"
METADATA = ROOT / "02_clean_datasets" / "survey_design_metadata.csv"
PARTITIONS = ROOT / "03_partitions" / "fixed_partition_assignments.csv"

print("=" * 75)
print("SW-01: ANALYSIS POPULATION AND PARTITION DESIGN AUDIT")
print("=" * 75)

# 1. Preprocessing manifest
print("\n--- PREPROCESSING MANIFEST ---")

if MANIFEST.exists():
    with open(MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print(json.dumps(manifest, indent=2))
else:
    print("Manifest not found:", MANIFEST)

# 2. Load data
metadata = pd.read_csv(
    METADATA,
    dtype={"source_row_id": "string"}
)

partitions = pd.read_csv(
    PARTITIONS,
    dtype={"source_row_id": "string"}
)

print("\n--- DATASET SIZES ---")
print("Survey metadata rows:", len(metadata))
print("Partition assignment rows:", len(partitions))
print("Unique metadata IDs:", metadata["source_row_id"].nunique())
print("Unique partition IDs:", partitions["source_row_id"].nunique())
print("Duplicate metadata IDs:", metadata["source_row_id"].duplicated().sum())
print("Duplicate partition IDs:", partitions["source_row_id"].duplicated().sum())

# 3. Partition completeness and consistency
print("\n--- PARTITION COMPLETENESS ---")
print(partitions["partition"].value_counts(dropna=False).to_string())

print("Missing partition labels:", partitions["partition"].isna().sum())
print(
    "Unexpected partition labels:",
    sorted(set(partitions["partition"].dropna()) -
           {"train", "validation", "test"})
)

metadata_ids = set(metadata["source_row_id"])
partition_ids = set(partitions["source_row_id"])

print("Metadata IDs absent from partitions:", len(metadata_ids - partition_ids))
print("Partition IDs absent from metadata:", len(partition_ids - metadata_ids))

# 4. Verify metadata-to-partition field agreement
print("\n--- METADATA / PARTITION AGREEMENT ---")

joined = metadata.merge(
    partitions,
    on="source_row_id",
    how="outer",
    suffixes=("_metadata", "_partition"),
    indicator=True,
    validate="one_to_one"
)

print("Merge results:")
print(joined["_merge"].value_counts().to_string())

for field in ["v024", "v021"]:
    left = joined[f"{field}_metadata"]
    right = joined[f"{field}_partition"]

    mismatch = (
        left.notna() & right.notna() &
        (left.astype("string") != right.astype("string"))
    )

    print(f"{field} mismatches:", mismatch.sum())

# 5. Distribution across partitions
print("\n--- REGION DISTRIBUTION BY PARTITION ---")

print(
    pd.crosstab(
        partitions["partition"],
        partitions["v024"],
        margins=True
    ).to_string()
)

print("\n--- STRATA DISTRIBUTION BY PARTITION ---")

print(
    pd.crosstab(
        partitions["partition"],
        partitions["v022"]
    ).to_string()
)

# 6. PSU overlap between partitions
print("\n--- PSU OVERLAP BETWEEN PARTITIONS ---")

psu_partition_counts = (
    partitions.groupby("v021")["partition"].nunique()
)

print("Total unique PSUs:", len(psu_partition_counts))
print(
    "PSUs appearing in multiple partitions:",
    (psu_partition_counts > 1).sum()
)
print(
    "PSUs appearing in only one partition:",
    (psu_partition_counts == 1).sum()
)

print("\nPSU counts by number of partitions represented:")
print(psu_partition_counts.value_counts().sort_index().to_string())

# 7. Number of PSUs per stratum and partition
print("\n--- PSUs PER STRATUM BY PARTITION ---")

for partition_name in ["train", "validation", "test"]:
    subset = partitions[
        partitions["partition"] == partition_name
    ]

    counts = subset.groupby("v022")["v021"].nunique()

    print(f"\n{partition_name.upper()}")
    print("Represented strata:", len(counts))
    print("Minimum PSUs per represented stratum:", counts.min())
    print("Strata with one PSU:", (counts == 1).sum())
    print("Strata with at least two PSUs:", (counts >= 2).sum())
    print("Strata with no observations:", 74 - len(counts))

# 8. Weight totals and effective sample size by partition
print("\n--- WEIGHT SUMMARY BY PARTITION ---")

weighted = partitions.merge(
    metadata[["source_row_id", "survey_weight"]],
    on="source_row_id",
    how="left",
    validate="one_to_one"
)

for partition_name in ["train", "validation", "test"]:
    subset = weighted[
        weighted["partition"] == partition_name
    ]

    w = subset["survey_weight"].to_numpy(dtype=float)

    effective_n = (w.sum() ** 2) / np.square(w).sum()

    print(f"\n{partition_name.upper()}")
    print("Respondents:", len(subset))
    print("Sum of weights:", w.sum())
    print("Minimum weight:", w.min())
    print("Maximum weight:", w.max())
    print("Weight-based effective sample size:", effective_n)

print("\n" + "=" * 75)
print("AUDIT COMPLETE — NO SOURCE FILES MODIFIED")
print("=" * 75)
