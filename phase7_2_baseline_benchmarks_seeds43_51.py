from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import phase7_1_baseline_benchmarks as bench
import canonical_federated_implementation_v1_1 as canonical


SEEDS = range(42, 52)
CONFIGURATIONS = ("baseline", "expanded")

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "phase7_2_baseline_benchmarks_seeds_42_51"


def run_seed(seed: int) -> None:
    seed_file = OUTPUT_DIR / f"validation_metrics_seed{seed}.csv"
    metadata_file = OUTPUT_DIR / f"run_metadata_seed{seed}.json"

    if seed_file.exists() or metadata_file.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing outputs for seed {seed}."
        )

    rows = []

    for configuration in CONFIGURATIONS:
        X_train, y_train, train_regions, _ = canonical.load_partition(
            configuration, "train"
        )
        X_val, y_val, val_regions, _ = canonical.load_partition(
            configuration, "validation"
        )

        canonical.validate_training_data(
            X_train,
            y_train,
            train_regions,
            canonical.CONFIGURATIONS[configuration],
        )
        if set(np.unique(val_regions)) != set(canonical.REGIONS):
            raise ValueError(
                f"{configuration} validation data must contain all regions."
            )

        # Centralized logistic regression
        logistic = LogisticRegression(
            max_iter=2000,
            random_state=seed,
            solver="lbfgs",
        )
        logistic.fit(X_train, y_train.astype(int))
        bench.evaluate_all_regions(
            logistic,
            X_val,
            y_val,
            val_regions,
            configuration=configuration,
            model_name="centralized_logistic_regression",
            seed=seed,
            rows=rows,
        )

        # Centralized MLP
        central_mlp = bench.train_mlp(X_train, y_train, seed=seed)
        bench.evaluate_mlp(
            central_mlp,
            X_val,
            y_val,
            val_regions,
            configuration=configuration,
            model_name="centralized_mlp",
            seed=seed,
            rows=rows,
        )

        # One independently trained local-only MLP per region
        for region_id in canonical.REGIONS:
            train_mask = train_regions == region_id
            val_mask = val_regions == region_id

            local_model = bench.train_mlp(
                X_train[train_mask],
                y_train[train_mask],
                seed=seed + region_id,
            )
            probabilities = bench.mlp_predict_proba(
                local_model, X_val[val_mask]
            )

            rows.append(
                bench.metric_row(
                    y_val[val_mask],
                    probabilities,
                    configuration=configuration,
                    model_name="local_only_mlp",
                    seed=seed,
                    scope="region",
                    region_id=region_id,
                )
            )

        print(f"Seed {seed}: completed {configuration}")

    results = pd.DataFrame(rows)
    if len(results) != 40:
        raise RuntimeError(
            f"Expected 40 rows for seed {seed}; got {len(results)}."
        )

    results.to_csv(seed_file, index=False)

    metadata = {
        "phase": "7.2",
        "seed": seed,
        "configurations": list(CONFIGURATIONS),
        "centralized_and_local_mlp_epochs": bench.EPOCHS,
        "federated_reference_budget": "10 rounds x 5 local epochs",
        "batch_size": bench.BATCH_SIZE,
        "learning_rate": bench.LEARNING_RATE,
        "threshold": bench.THRESHOLD,
        "evaluation_partition": "validation",
        "survey_weights_used": False,
        "optimizer_reset_interval_epochs": 5,
        "optimizer_protocol": (
            "Fresh Adam optimizer every 5 epochs; 50 epochs total."
        ),
        "local_only_protocol": (
            "One independently trained MLP per region; each evaluated "
            "on that region's validation subset."
        ),
        "metrics_rows": len(results),
    }
    with metadata_file.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved seed {seed}: {seed_file}")


def main():
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {OUTPUT_DIR}"
        )

    OUTPUT_DIR.mkdir(parents=True)

    for seed in SEEDS:
        run_seed(seed)

    print(f"\nAll requested seeds completed. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()