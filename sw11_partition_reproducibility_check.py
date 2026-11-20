# verifying exact split reproducibility

from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(
    r"C:\Users\usher\Documents\NDHS-Fl-Research\NDHS_FL_Research"
)

CLEAN = ROOT / "02_clean_datasets"
SPLIT_DIR = ROOT / "03_partitions"

baseline = pd.read_csv(CLEAN / "baseline_features.csv")
metadata = pd.read_csv(CLEAN / "survey_design_metadata.csv")

saved = pd.read_csv(
    SPLIT_DIR / "fixed_partition_assignments.csv"
)

# Reconstruct the original notebook's core dataset.
core = baseline[
    ["source_row_id", "facility_delivery", "v024"]
].copy()

core = core.merge(
    metadata[
        ["source_row_id", "v021", "v005",
         "v022", "v023", "survey_weight"]
    ],
    on="source_row_id",
    how="left",
    validate="one_to_one"
)

assert core["v021"].notna().all()

# Reproduce the original notebook's split logic.
core["stratum"] = (
    core["v024"].astype(str)
    + "_"
    + core["facility_delivery"].astype(str)
)

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=2026
)

fold_assignment = np.full(len(core), -1, dtype=int)

for fold, (_, fold_index) in enumerate(
    splitter.split(
        X=np.zeros((len(core), 1)),
        y=core["stratum"],
        groups=core["v021"]
    )
):
    fold_assignment[fold_index] = fold

assert (fold_assignment >= 0).all()

core["partition_reproduced"] = np.select(
    [
        fold_assignment == 0,
        fold_assignment == 1
    ],
    [
        "test",
        "validation"
    ],
    default="train"
)

# Compare with the saved assignments.
comparison = core[
    ["source_row_id", "v024", "v021", "facility_delivery",
     "partition_reproduced"]
].merge(
    saved,
    on="source_row_id",
    how="outer",
    suffixes=("_reproduced", "_saved"),
    indicator=True,
    validate="one_to_one"
)

print("=" * 65)
print("SW-01: PARTITION REPRODUCIBILITY CHECK")
print("=" * 65)

print("scikit-learn version:", sklearn.__version__)
print("Reproduced rows:", len(core))
print("Saved assignment rows:", len(saved))
print("ID merge counts:")
print(comparison["_merge"].value_counts().to_string())

matched = comparison["_merge"].eq("both")

print(
    "\nRegion mismatches:",
    (
        comparison.loc[matched, "v024_reproduced"]
        != comparison.loc[matched, "v024_saved"]
    ).sum()
)

print(
    "PSU mismatches:",
    (
        comparison.loc[matched, "v021_reproduced"]
        != comparison.loc[matched, "v021_saved"]
    ).sum()
)

print(
    "Outcome mismatches:",
    (
        comparison.loc[matched, "facility_delivery_reproduced"]
        != comparison.loc[matched, "facility_delivery_saved"]
    ).sum()
)

partition_mismatches = (
    comparison.loc[matched, "partition_reproduced"]
    != comparison.loc[matched, "partition"]
).sum()

print("Partition mismatches:", partition_mismatches)

print("\nReproduced partition counts:")
print(core["partition_reproduced"].value_counts().to_string())

print("\nSaved partition counts:")
print(saved["partition"].value_counts().to_string())

if (
    comparison["_merge"].eq("both").all()
    and partition_mismatches == 0
):
    print("\nRESULT: EXACT REPRODUCTION PASS")
else:
    print("\nRESULT: EXACT REPRODUCTION NOT CONFIRMED")

print("\nNo files modified.")
