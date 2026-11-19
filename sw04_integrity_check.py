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

# Attach survey-design variables to partition assignments.
df = partitions.merge(
    metadata[
        ["source_row_id", "survey_weight", "v005",
         "v021", "v022", "v023", "v001", "v024"]
    ],
    on="source_row_id",
    how="left",
    validate="one_to_one",
    suffixes=("_partition", "_metadata")
)

print("=" * 70)
print("SW-01: CORRECTED PARTITION AND SURVEY DESIGN AUDIT")
print("=" * 70)

print("\n--- MERGED DATASET ---")
print("Rows:", len(df))
print("Missing survey weights:", df["survey_weight"].isna().sum())
print("Missing v022:", df["v022"].isna().sum())
print("Missing v021:", df["v021"].isna().sum())

# PSU overlap between partitions
print("\n--- PSU OVERLAP ---")

psu_partition_counts = (
    df.groupby("v021")["partition"].nunique()
)

print("Total unique PSUs:", len(psu_partition_counts))
print("PSUs in multiple partitions:", (psu_partition_counts > 1).sum())
print("PSUs in one partition only:", (psu_partition_counts == 1).sum())

print("\nPSUs by number of partitions:")
print(psu_partition_counts.value_counts().sort_index().to_string())

# Region and strata distribution
print("\n--- REGION DISTRIBUTION BY PARTITION ---")
print(
    pd.crosstab(
        df["partition"],
        df["v024_partition"]
    ).to_string()
)

print("\n--- STRATA DISTRIBUTION BY PARTITION ---")
print(
    pd.crosstab(
        df["partition"],
        df["v022"]
    ).to_string()
)

# PSUs represented in each partition and stratum
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

# Weight summaries
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
print("CORRECTED AUDIT COMPLETE — NO SOURCE FILES MODIFIED")
print("=" * 70)
