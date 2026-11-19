# inspecting the preprocessing and partition badge


from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(
    r"C:\Users\usher\Documents\NDHS-Fl-Research\NDHS_FL_Research"
)

METADATA = ROOT / "02_clean_datasets" / "survey_design_metadata.csv"
PARTITIONS = ROOT / "03_partitions" / "fixed_partition_assignments.csv"

metadata = pd.read_csv(
    METADATA,
    dtype={"source_row_id": "string"}
)

partitions = pd.read_csv(
    PARTITIONS,
    dtype={"source_row_id": "string"}
)

# Merge only the survey-design fields not already in partitions.
design_fields = [
    "source_row_id",
    "survey_weight",
    "v005",
    "v022",
    "v023"
]

df = partitions.merge(
    metadata[design_fields],
    on="source_row_id",
    how="left",
    validate="one_to_one"
)

print("=" * 70)
print("SW-01: FINAL PARTITION AND SURVEY DESIGN AUDIT")
print("=" * 70)

print("\n--- MERGED DATASET ---")
print("Rows:", len(df))

for col in [
    "survey_weight", "v005", "v021",
    "v022", "v023", "v024"
]:
    print(col, "missing:", df[col].isna().sum())

# Confirm partition and metadata identifiers agree.
print("\n--- IDENTIFIER CONSISTENCY ---")
print(
    "v021 and v001 identical:",
    (metadata["v021"].to_numpy() ==
     metadata["v001"].to_numpy()).all()
)

# Confirm region and PSU fields agree between source files.
check = partitions.merge(
    metadata[["source_row_id", "v021", "v024"]],
    on="source_row_id",
    how="inner",
    suffixes=("_partition", "_metadata"),
    validate="one_to_one"
)

for col in ["v021", "v024"]:
    mismatch = (
        check[f"{col}_partition"] !=
        check[f"{col}_metadata"]
    )
    print(f"{col} mismatches:", mismatch.sum())

# PSU overlap across partitions.
print("\n--- PSU OVERLAP ---")

psu_partition_counts = (
    df.groupby("v021")["partition"].nunique()
)

print("Total unique PSUs:", len(psu_partition_counts))
print("PSUs in multiple partitions:", (psu_partition_counts > 1).sum())
print("PSUs in one partition only:", (psu_partition_counts == 1).sum())

print("\nPSUs by number of partitions:")
print(psu_partition_counts.value_counts().sort_index().to_string())

# Regional distribution.
print("\n--- REGION DISTRIBUTION BY PARTITION ---")
print(
    pd.crosstab(
        df["partition"],
        df["v024"]
    ).to_string()
)

# Strata distribution.
print("\n--- STRATA DISTRIBUTION BY PARTITION ---")
print(
    pd.crosstab(
        df["partition"],
        df["v022"]
    ).to_string()
)

# PSU counts per stratum in each partition.
print("\n--- PSUs PER STRATUM BY PARTITION ---")

for name in ["train", "validation", "test"]:
    subset = df[df["partition"] == name]
    counts = subset.groupby("v022")["v021"].nunique()

    print(f"\n{name.upper()}")
    print("Respondents:", len(subset))
    print("Represented strata:", len(counts))
    print("Minimum PSUs per represented stratum:", counts.min())
    print("Single-PSU strata:", (counts == 1).sum())
    print("Strata with at least two PSUs:", (counts >= 2).sum())
    print("Unrepresented strata:", 74 - len(counts))

# Weight summaries.
print("\n--- WEIGHT SUMMARY BY PARTITION ---")

for name in ["train", "validation", "test"]:
    subset = df[df["partition"] == name]
    w = subset["survey_weight"].to_numpy(dtype=float)

    effective_n = (w.sum() ** 2) / np.square(w).sum()

    print(f"\n{name.upper()}")
    print("Respondents:", len(subset))
    print("Weight sum:", w.sum())
    print("Minimum weight:", w.min())
    print("Maximum weight:", w.max())
    print("Effective sample size:", effective_n)

print("\n" + "=" * 70)
print("AUDIT COMPLETE — NO SOURCE FILES MODIFIED")
print("=" * 70)
