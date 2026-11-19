# verifying the survey design variables

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(
    r"C:\Users\usher\Documents\NDHS-Fl-Research\NDHS_FL_Research"
)

metadata_path = ROOT / "02_clean_datasets" / "survey_design_metadata.csv"
partition_path = ROOT / "03_partitions" / "fixed_partition_assignments.csv"

metadata = pd.read_csv(metadata_path)
partitions = pd.read_csv(partition_path)

# Join partition labels to the survey metadata.
df = metadata.merge(
    partitions[["source_row_id", "partition"]],
    on="source_row_id",
    how="left",
    validate="one_to_one"
)

print("=" * 70)
print("SW-01: SURVEY DESIGN VARIABLE DIAGNOSTIC")
print("=" * 70)

# 1. Weight verification
print("\n--- WEIGHT VERIFICATION ---")

df["calculated_weight"] = df["v005"] / 1_000_000

print("Minimum v005:", df["v005"].min())
print("Maximum v005:", df["v005"].max())
print("Minimum survey_weight:", df["survey_weight"].min())
print("Maximum survey_weight:", df["survey_weight"].max())
print(
    "Maximum absolute weight difference:",
    (df["survey_weight"] - df["calculated_weight"]).abs().max()
)
print("All weights positive:", (df["survey_weight"] > 0).all())
print("All weights finite:", np.isfinite(df["survey_weight"]).all())

# 2. PSU and cluster relationship
print("\n--- PSU / CLUSTER VERIFICATION ---")

print("v021 unique:", df["v021"].nunique())
print("v001 unique:", df["v001"].nunique())
print("v021 equals v001 for every row:", (df["v021"] == df["v001"]).all())

print(
    "Number of v021 values mapped to multiple v001 values:",
    df.groupby("v021")["v001"].nunique().gt(1).sum()
)

print(
    "Number of v001 values mapped to multiple v021 values:",
    df.groupby("v001")["v021"].nunique().gt(1).sum()
)

# 3. Strata verification
print("\n--- STRATA VARIABLES ---")

for col in ["v022", "v023"]:
    print(f"\n{col}")
    print("Unique values:", df[col].nunique())
    print("Missing:", df[col].isna().sum())
    print("Value counts:")
    print(df[col].value_counts().sort_index().to_string())

print(
    "\nNumber of distinct v022-v023 combinations:",
    df[["v022", "v023"]].drop_duplicates().shape[0]
)

print(
    "Does each v022 map to exactly one v023?",
    df.groupby("v022")["v023"].nunique().eq(1).all()
)

print(
    "Does each v023 map to exactly one v022?",
    df.groupby("v023")["v022"].nunique().eq(1).all()
)

# 4. PSU counts per stratum in the full dataset
print("\n--- PSU COUNTS BY STRATUM: FULL DATA ---")

for stratum in ["v022", "v023"]:
    counts = df.groupby(stratum)["v021"].nunique()

    print(f"\nUsing {stratum}:")
    print("Number of strata:", len(counts))
    print("Minimum PSUs per stratum:", counts.min())
    print("Strata with only one PSU:", (counts == 1).sum())
    print("Strata with two or more PSUs:", (counts >= 2).sum())

# 5. PSU counts in the test subset
test = df[df["partition"] == "test"].copy()

print("\n--- PSU COUNTS BY STRATUM: TEST SUBSET ---")

for stratum in ["v022", "v023"]:
    counts = test.groupby(stratum)["v021"].nunique()

    print(f"\nUsing {stratum}:")
    print("Number of represented strata:", len(counts))
    print("Minimum PSUs per represented stratum:", counts.min())
    print("Strata with only one PSU:", (counts == 1).sum())
    print("Strata with two or more PSUs:", (counts >= 2).sum())

# 6. Missing design values
print("\n--- MISSING DESIGN VALUES ---")

for col in ["v005", "survey_weight", "v021", "v022", "v023", "v001", "v024"]:
    print(col, "missing:", df[col].isna().sum())

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE — NO FILES MODIFIED")
print("=" * 70)
