"""
NDHS Federated Learning — Phase 5 Pilot Runner v1.0
=================================================
Stage 1 benchmark:
- Baseline and expanded feature configurations
- FedAvg and FedProx (mu=0.01)
- 10 communication rounds
- 1 local epoch
- Controlled pilot seeds 42 and 43
- Validation-only evaluation

This is a computational pilot, NOT a confirmatory experiment.
It imports the canonical v1.1 implementation without modifying it.
It never loads the test partition.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform
import time

import numpy as np
import pandas as pd
import torch

import canonical_federated_implementation_v1_1 as canonical


# ---------------------------------------------------------------------
# Frozen Stage 1 pilot settings
# ---------------------------------------------------------------------

PILOT_VERSION = "1.0"
PILOT_STAGE = "stage1_10rounds_E1"
PILOT_SEEDS = (42, 43)
PILOT_ROUNDS = 10
PILOT_LOCAL_EPOCHS = 1
PILOT_ALGORITHMS = ("FedAvg", "FedProx")
PILOT_CONFIGURATIONS = ("baseline", "expanded")
PILOT_MU = 0.01

PROJECT_ROOT = canonical.ROOT
OUTPUT_ROOT = PROJECT_ROOT / "10_phase5_pilot"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_finite_state(state_dict, label: str) -> None:
    for name, tensor in state_dict.items():
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"Non-finite parameter in {label}: {name}")


def assert_matched_initialization(result_a, result_b, label: str) -> float:
    passed, maximum_difference = canonical.compare_state_dicts(
        result_a["initial_state"], result_b["initial_state"]
    )
    if not passed:
        raise RuntimeError(
            f"Matched initial parameters failed for {label}; "
            f"maximum difference={maximum_difference}"
        )
    return maximum_difference


def save_one_run(result: dict, run_dir: Path, metadata: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=False)

    pd.DataFrame(result["round_records"]).to_csv(
        run_dir / "round_summary.csv", index=False
    )
    pd.DataFrame(result["evaluation_records"]).to_csv(
        run_dir / "validation_metrics_pooled_and_regional.csv", index=False
    )
    pd.DataFrame(result["confusion_records"]).to_csv(
        run_dir / "validation_confusion_counts.csv", index=False
    )
    pd.concat(result["probability_records"], ignore_index=True).to_csv(
        run_dir / "validation_probabilities_with_source_ids.csv", index=False
    )
    pd.DataFrame(result["participation_records"]).to_csv(
        run_dir / "client_participation.csv", index=False
    )

    torch.save(result["initial_state"], run_dir / "initial_model_state.pt")
    torch.save(result["final_state"], run_dir / "final_model_state.pt")

    manifest = {
        "pilot_version": PILOT_VERSION,
        "pilot_stage": PILOT_STAGE,
        "confirmatory_experiment": False,
        "test_partition_accessed": False,
        "validation_only_evaluation": True,
        "metadata": metadata,
        "canonical_implementation_version": canonical.VERSION,
        "canonical_implementation_sha256": sha256_file(
            Path(canonical.__file__).resolve()
        ),
        "evaluation_module_sha256": sha256_file(
            Path(canonical.__file__).with_name("evaluate_model_v1_1.py")
        ),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "device": str(canonical.DEVICE),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    with (run_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, default=str)


def run_stage1() -> Path:
    if not canonical.DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Preprocessed data directory not found: {canonical.DATA_DIR}"
        )
    evaluator_path = Path(canonical.__file__).with_name("evaluate_model_v1_1.py")
    if not evaluator_path.is_file():
        raise FileNotFoundError(
            "evaluate_model_v1_1.py must be beside the canonical implementation."
        )

    # Verify all required train/validation arrays before starting any model run.
    partition_sizes = {}
    for configuration in PILOT_CONFIGURATIONS:
        X_train, y_train, train_regions, train_ids = canonical.load_partition(
            configuration, "train"
        )
        X_val, y_val, val_regions, val_ids = canonical.load_partition(
            configuration, "validation"
        )
        canonical.validate_training_data(
            X_train, y_train, train_regions,
            canonical.CONFIGURATIONS[configuration]
        )
        if set(np.unique(val_regions)) != set(canonical.REGIONS):
            raise RuntimeError(
                f"{configuration}: validation partition lacks one or more regions."
            )
        partition_sizes[configuration] = {
            "training_n": int(len(y_train)),
            "validation_n": int(len(y_val)),
            "training_source_ids_unique": bool(len(np.unique(train_ids)) == len(train_ids)),
            "validation_source_ids_unique": bool(len(np.unique(val_ids)) == len(val_ids)),
        }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = OUTPUT_ROOT / f"{PILOT_STAGE}_{stamp}"
    run_root.mkdir(parents=True, exist_ok=False)

    run_summaries = []
    matched_initialization_checks = []

    total_start = time.perf_counter()

    for configuration in PILOT_CONFIGURATIONS:
        for seed in PILOT_SEEDS:
            paired_results = {}

            # Run both algorithms with the same configuration and seed.
            for algorithm in PILOT_ALGORITHMS:
                mu = 0.0 if algorithm == "FedAvg" else PILOT_MU
                print(
                    f"\nStarting {configuration} | {algorithm} | seed={seed} | "
                    f"rounds={PILOT_ROUNDS} | local_epochs={PILOT_LOCAL_EPOCHS}",
                    flush=True,
                )

                started = time.perf_counter()
                result = canonical.run_federated_experiment(
                    configuration=configuration,
                    seed=seed,
                    algorithm=algorithm,
                    rounds=PILOT_ROUNDS,
                    local_epochs=PILOT_LOCAL_EPOCHS,
                    mu=mu,
                )
                elapsed = time.perf_counter() - started

                assert_finite_state(
                    result["final_state"],
                    f"{configuration}/{algorithm}/seed{seed}"
                )

                probabilities = pd.concat(
                    result["probability_records"], ignore_index=True
                )
                if not np.isfinite(
                    probabilities["predicted_probability"].to_numpy(dtype=float)
                ).all():
                    raise RuntimeError(
                        f"Non-finite validation probabilities: "
                        f"{configuration}/{algorithm}/seed{seed}"
                    )
                if len(probabilities) != (
                    partition_sizes[configuration]["validation_n"] * PILOT_ROUNDS
                ):
                    raise RuntimeError(
                        "Saved validation probability count is inconsistent "
                        "with validation sample size and number of rounds."
                    )

                metadata = {
                    "configuration": configuration,
                    "input_features": canonical.CONFIGURATIONS[configuration],
                    "algorithm": algorithm,
                    "seed": seed,
                    "communication_rounds": PILOT_ROUNDS,
                    "local_epochs": PILOT_LOCAL_EPOCHS,
                    "learning_rate": canonical.LEARNING_RATE,
                    "fedprox_mu": mu,
                    "batch_size": canonical.BATCH_SIZE,
                    **partition_sizes[configuration],
                    "elapsed_seconds": elapsed,
                    "mean_seconds_per_round": elapsed / PILOT_ROUNDS,
                }

                run_dir = (
                    run_root
                    / configuration
                    / f"seed_{seed}"
                    / algorithm.lower()
                )
                # Create parent directories only; run directory itself must be new.
                run_dir.parent.mkdir(parents=True, exist_ok=True)
                save_one_run(result, run_dir, metadata)

                final_round = result["round_records"][-1]
                run_summaries.append({
                    **metadata,
                    "final_validation_loss": final_round[
                        "validation_binary_cross_entropy_loss"
                    ],
                    "final_validation_accuracy": final_round[
                        "validation_accuracy"
                    ],
                    "final_validation_macro_f1": final_round[
                        "validation_macro_f1"
                    ],
                    "final_validation_auroc": final_round["validation_auroc"],
                    "final_validation_auprc": final_round["validation_auprc"],
                    "final_validation_brier": final_round[
                        "validation_brier_score"
                    ],
                    "final_validation_ece": final_round["validation_ece"],
                    "all_parameters_finite": True,
                    "all_probabilities_finite": True,
                    "outputs_saved": True,
                })
                paired_results[algorithm] = result

                print(
                    f"Completed in {elapsed:.2f}s "
                    f"({elapsed / PILOT_ROUNDS:.2f}s/round); "
                    f"final validation loss="
                    f"{final_round['validation_binary_cross_entropy_loss']:.5f}; "
                    f"accuracy={final_round['validation_accuracy']:.5f}; "
                    f"macro-F1={final_round['validation_macro_f1']:.5f}",
                    flush=True,
                )

            max_diff = assert_matched_initialization(
                paired_results["FedAvg"],
                paired_results["FedProx"],
                f"{configuration}/seed{seed}",
            )
            matched_initialization_checks.append({
                "configuration": configuration,
                "seed": seed,
                "maximum_initial_parameter_difference": max_diff,
                "passed": True,
            })

    total_elapsed = time.perf_counter() - total_start

    summary_df = pd.DataFrame(run_summaries)
    summary_df.to_csv(run_root / "pilot_run_summary.csv", index=False)

    pd.DataFrame(matched_initialization_checks).to_csv(
        run_root / "matched_initialization_checks.csv", index=False
    )

    # Combined per-round trajectory and regional metric tables for review.
    round_tables = []
    regional_tables = []
    confusion_tables = []
    for configuration in PILOT_CONFIGURATIONS:
        for seed in PILOT_SEEDS:
            for algorithm in PILOT_ALGORITHMS:
                run_dir = (
                    run_root / configuration / f"seed_{seed}" / algorithm.lower()
                )
                round_tables.append(pd.read_csv(run_dir / "round_summary.csv"))
                regional_tables.append(pd.read_csv(
                    run_dir / "validation_metrics_pooled_and_regional.csv"
                ))
                confusion_tables.append(pd.read_csv(
                    run_dir / "validation_confusion_counts.csv"
                ))

    pd.concat(round_tables, ignore_index=True).to_csv(
        run_root / "all_runs_round_trajectories.csv", index=False
    )
    pd.concat(regional_tables, ignore_index=True).to_csv(
        run_root / "all_runs_pooled_and_regional_metrics.csv", index=False
    )
    pd.concat(confusion_tables, ignore_index=True).to_csv(
        run_root / "all_runs_confusion_counts.csv", index=False
    )

    pilot_manifest = {
        "pilot_version": PILOT_VERSION,
        "stage": PILOT_STAGE,
        "purpose": "Computational feasibility and validation-behaviour pilot",
        "confirmatory_experiment": False,
        "test_partition_accessed": False,
        "configurations": list(PILOT_CONFIGURATIONS),
        "algorithms": list(PILOT_ALGORITHMS),
        "seeds": list(PILOT_SEEDS),
        "communication_rounds": PILOT_ROUNDS,
        "local_epochs": PILOT_LOCAL_EPOCHS,
        "learning_rate": canonical.LEARNING_RATE,
        "fedprox_mu": PILOT_MU,
        "threshold": canonical.PREDICTION_THRESHOLD,
        "ece_bins": canonical.CALIBRATION_BINS,
        "partition_sizes": partition_sizes,
        "matched_initialization_checks": matched_initialization_checks,
        "total_wall_clock_seconds_including_output_writing": total_elapsed,
        "run_count": len(run_summaries),
        "all_runs_completed": len(run_summaries) == (
            len(PILOT_CONFIGURATIONS) * len(PILOT_SEEDS) * len(PILOT_ALGORITHMS)
        ),
        "canonical_implementation_sha256": sha256_file(
            Path(canonical.__file__).resolve()
        ),
        "evaluation_module_sha256": sha256_file(evaluator_path),
        "environment": canonical.get_environment(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    with (run_root / "pilot_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(pilot_manifest, stream, indent=2, default=str)

    print("\nPHASE 5 STAGE 1 COMPLETED", flush=True)
    print(f"Runs completed: {len(run_summaries)}", flush=True)
    print(f"Total elapsed including output writing: {total_elapsed:.2f}s", flush=True)
    print(f"Pilot outputs: {run_root}", flush=True)
    print("These are pilot results, not confirmatory findings.", flush=True)
    return run_root


if __name__ == "__main__":
    run_stage1()
