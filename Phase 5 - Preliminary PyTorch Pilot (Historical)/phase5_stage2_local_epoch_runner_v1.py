"""
NDHS Federated Learning — Phase 5 Stage 2 Runner v1.0
====================================================
Purpose: compare local epochs E=5 and E=10 at 10 communication rounds.

Design:
- Baseline and expanded feature configurations
- FedAvg and FedProx (mu=0.01)
- 10 communication rounds
- E=5 and E=10
- One controlled pilot seed: 42
- Validation-only evaluation
- Paired initial global parameters within each configuration/epoch

This is a computational pilot, NOT confirmatory training.
It imports canonical implementation v1.1 without modifying it.
It never loads the test partition.

Important metric-recording safeguard:
The canonical v1.1 round summary currently requests "auprc", whereas the
v1.1 evaluator names that field "auprc_average_precision". This runner
constructs its round trajectory directly from the evaluator's detailed
metric records and uses the evaluator's exact field name. It does not
silently substitute or infer missing AUPRC values.
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


PILOT_VERSION = "1.0"
PILOT_STAGE = "stage2_local_epoch_comparison"
PILOT_SEED = 42
PILOT_ROUNDS = 10
LOCAL_EPOCH_SETTINGS = (5, 10)
ALGORITHMS = ("FedAvg", "FedProx")
CONFIGURATIONS = ("baseline", "expanded")
FEDPROX_MU = 0.01

OUTPUT_ROOT = canonical.ROOT / "10_phase5_pilot"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_finite_state(state_dict, label: str) -> None:
    for name, tensor in state_dict.items():
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"Non-finite model parameter in {label}: {name}")


def assert_matched_initialization(fedavg_result, fedprox_result, label: str) -> float:
    passed, maximum_difference = canonical.compare_state_dicts(
        fedavg_result["initial_state"], fedprox_result["initial_state"]
    )
    if not passed:
        raise RuntimeError(
            f"Matched initial parameters failed for {label}; "
            f"max difference={maximum_difference}"
        )
    return float(maximum_difference)


def pooled_round_trajectory(metrics: pd.DataFrame) -> pd.DataFrame:
    """Build pooled round trajectory from evaluator output, not cached summaries."""
    required = {
        "region_id", "communication_round", "binary_cross_entropy_loss",
        "accuracy", "macro_f1", "auroc", "auprc_average_precision",
        "brier_score", "ece", "calibration_intercept", "calibration_slope",
    }
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise RuntimeError(
            "Evaluator metrics are missing required columns: " + ", ".join(missing)
        )

    pooled = metrics.loc[metrics["region_id"] == 0].copy()
    if pooled.empty:
        raise RuntimeError("No pooled evaluation rows (region_id=0) in evaluator output.")
    if pooled.duplicated(["communication_round"]).any():
        raise RuntimeError("Duplicate pooled rows found for a communication round.")

    trajectory = pooled[[
        "communication_round",
        "binary_cross_entropy_loss",
        "accuracy",
        "macro_f1",
        "auroc",
        "auprc_average_precision",
        "brier_score",
        "ece",
        "calibration_intercept",
        "calibration_slope",
    ]].copy()

    # Preserve calibration fit status/reason rather than treating NaN as a value.
    for column in ("calibration_status", "calibration_reason"):
        if column in pooled.columns:
            trajectory[column] = pooled[column].values

    trajectory = trajectory.rename(columns={
        "communication_round": "round",
        "binary_cross_entropy_loss": "validation_binary_cross_entropy_loss",
        "accuracy": "validation_accuracy",
        "macro_f1": "validation_macro_f1",
        "auroc": "validation_auroc",
        "auprc_average_precision": "validation_auprc_average_precision",
        "brier_score": "validation_brier_score",
        "ece": "validation_ece",
        "calibration_intercept": "validation_calibration_intercept",
        "calibration_slope": "validation_calibration_slope",
    })
    trajectory = trajectory.sort_values("round").reset_index(drop=True)

    if len(trajectory) != PILOT_ROUNDS:
        raise RuntimeError(
            f"Expected {PILOT_ROUNDS} pooled round rows; found {len(trajectory)}."
        )
    if trajectory["validation_auprc_average_precision"].isna().any():
        raise RuntimeError(
            "AUPRC is NaN in evaluator pooled records. Inspect evaluator output; "
            "do not infer or fill this metric."
        )
    return trajectory


def run_stage2() -> Path:
    canonical_file = Path(canonical.__file__).resolve()
    evaluator_file = canonical_file.with_name("evaluate_model_v1_1.py")
    if not evaluator_file.is_file():
        raise FileNotFoundError(
            "evaluate_model_v1_1.py must be in the same directory as canonical v1.1."
        )

    # Preflight: verify train/validation arrays and IDs, without opening test data.
    preflight = {}
    for config in CONFIGURATIONS:
        Xtr, ytr, rtr, ids_tr = canonical.load_partition(config, "train")
        Xva, yva, rva, ids_va = canonical.load_partition(config, "validation")
        canonical.validate_training_data(
            Xtr, ytr, rtr, canonical.CONFIGURATIONS[config]
        )
        if Xva.shape[1] != canonical.CONFIGURATIONS[config]:
            raise RuntimeError(f"{config}: validation feature dimension mismatch.")
        if len(set(np.asarray(ids_tr).astype(str))) != len(ids_tr):
            raise RuntimeError(f"{config}: duplicate training source IDs.")
        if len(set(np.asarray(ids_va).astype(str))) != len(ids_va):
            raise RuntimeError(f"{config}: duplicate validation source IDs.")
        if set(np.unique(rva)) != set(canonical.REGIONS):
            raise RuntimeError(f"{config}: validation does not contain all six regions.")
        preflight[config] = {
            "training_n": int(len(ytr)),
            "validation_n": int(len(yva)),
            "training_features": int(Xtr.shape[1]),
            "validation_features": int(Xva.shape[1]),
            "training_source_ids_unique": True,
            "validation_source_ids_unique": True,
        }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"{PILOT_STAGE}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    summaries = []
    paired_checks = []
    all_metrics = []
    all_confusions = []
    all_probabilities = []
    all_participation = []
    total_start = time.perf_counter()

    # For each feature/epoch setting, compare algorithms with matched initialization.
    for config in CONFIGURATIONS:
        for epochs in LOCAL_EPOCH_SETTINGS:
            paired_results = {}
            for algorithm in ALGORITHMS:
                mu = 0.0 if algorithm == "FedAvg" else FEDPROX_MU
                label = f"{config}/{algorithm}/E{epochs}/seed{PILOT_SEED}"
                print(f"Starting {label}", flush=True)
                start = time.perf_counter()

                result = canonical.run_federated_experiment(
                    configuration=config,
                    seed=PILOT_SEED,
                    algorithm=algorithm,
                    rounds=PILOT_ROUNDS,
                    local_epochs=epochs,
                    mu=mu,
                )
                elapsed = time.perf_counter() - start

                assert_finite_state(result["final_state"], label)

                metrics = pd.DataFrame(result["evaluation_records"])
                confusion = pd.DataFrame(result["confusion_records"])
                probabilities = pd.concat(
                    result["probability_records"], ignore_index=True
                )
                participation = pd.DataFrame(result["participation_records"])

                # Verify evaluator produced pooled and six regional rows each round.
                expected_metric_rows = PILOT_ROUNDS * 7
                if len(metrics) != expected_metric_rows:
                    raise RuntimeError(
                        f"{label}: expected {expected_metric_rows} metrics rows, "
                        f"found {len(metrics)}."
                    )
                if len(confusion) != expected_metric_rows:
                    raise RuntimeError(
                        f"{label}: expected {expected_metric_rows} confusion rows, "
                        f"found {len(confusion)}."
                    )
                expected_probability_rows = (
                    preflight[config]["validation_n"] * PILOT_ROUNDS
                )
                if len(probabilities) != expected_probability_rows:
                    raise RuntimeError(
                        f"{label}: expected {expected_probability_rows} probability "
                        f"rows, found {len(probabilities)}."
                    )
                if not np.isfinite(
                    probabilities["predicted_probability"].to_numpy(dtype=float)
                ).all():
                    raise RuntimeError(f"{label}: non-finite validation probabilities.")

                trajectory = pooled_round_trajectory(metrics)
                # Explicit metadata, retaining the exact evaluator field names.
                for frame in (metrics, confusion, probabilities, participation, trajectory):
                    frame["configuration"] = config
                    frame["algorithm"] = algorithm
                    frame["seed"] = PILOT_SEED
                    frame["local_epochs"] = epochs
                    frame["mu"] = mu

                run_dir = (
                    output_dir / config / f"E{epochs}" / algorithm.lower()
                )
                run_dir.mkdir(parents=True, exist_ok=False)
                trajectory.to_csv(run_dir / "pooled_validation_round_trajectory.csv", index=False)
                metrics.to_csv(run_dir / "validation_metrics_pooled_and_regional.csv", index=False)
                confusion.to_csv(run_dir / "validation_confusion_counts.csv", index=False)
                probabilities.to_csv(
                    run_dir / "validation_probabilities_with_source_ids.csv", index=False
                )
                participation.to_csv(run_dir / "client_participation.csv", index=False)
                torch.save(result["initial_state"], run_dir / "initial_model_state.pt")
                torch.save(result["final_state"], run_dir / "final_model_state.pt")

                final = trajectory.iloc[-1]
                summaries.append({
                    "configuration": config,
                    "algorithm": algorithm,
                    "seed": PILOT_SEED,
                    "communication_rounds": PILOT_ROUNDS,
                    "local_epochs": epochs,
                    "learning_rate": canonical.LEARNING_RATE,
                    "fedprox_mu": mu,
                    "input_features": canonical.CONFIGURATIONS[config],
                    "training_n": preflight[config]["training_n"],
                    "validation_n": preflight[config]["validation_n"],
                    "elapsed_seconds": elapsed,
                    "mean_seconds_per_round": elapsed / PILOT_ROUNDS,
                    "final_validation_loss": final["validation_binary_cross_entropy_loss"],
                    "final_validation_accuracy": final["validation_accuracy"],
                    "final_validation_macro_f1": final["validation_macro_f1"],
                    "final_validation_auroc": final["validation_auroc"],
                    "final_validation_auprc_average_precision": final[
                        "validation_auprc_average_precision"
                    ],
                    "final_validation_brier_score": final["validation_brier_score"],
                    "final_validation_ece": final["validation_ece"],
                    "all_parameters_finite": True,
                    "all_probabilities_finite": True,
                    "test_partition_accessed": False,
                })

                all_metrics.append(metrics)
                all_confusions.append(confusion)
                all_probabilities.append(probabilities)
                all_participation.append(participation)
                paired_results[algorithm] = result

                print(
                    f"Completed {label}: {elapsed:.2f}s total, "
                    f"{elapsed / PILOT_ROUNDS:.2f}s/round; "
                    f"val loss={final['validation_binary_cross_entropy_loss']:.5f}; "
                    f"Macro-F1={final['validation_macro_f1']:.5f}; "
                    f"AUPRC={final['validation_auprc_average_precision']:.5f}",
                    flush=True,
                )

            max_diff = assert_matched_initialization(
                paired_results["FedAvg"], paired_results["FedProx"],
                f"{config}/E{epochs}/seed{PILOT_SEED}"
            )
            paired_checks.append({
                "configuration": config,
                "local_epochs": epochs,
                "seed": PILOT_SEED,
                "maximum_initial_parameter_difference": max_diff,
                "passed": True,
            })

    total_elapsed = time.perf_counter() - total_start

    pd.DataFrame(summaries).to_csv(output_dir / "stage2_run_summary.csv", index=False)
    pd.DataFrame(paired_checks).to_csv(
        output_dir / "matched_initialization_checks.csv", index=False
    )
    pd.concat(all_metrics, ignore_index=True).to_csv(
        output_dir / "all_validation_metrics_pooled_and_regional.csv", index=False
    )
    pd.concat(all_confusions, ignore_index=True).to_csv(
        output_dir / "all_validation_confusion_counts.csv", index=False
    )
    pd.concat(all_probabilities, ignore_index=True).to_csv(
        output_dir / "all_validation_probabilities_with_source_ids.csv", index=False
    )
    pd.concat(all_participation, ignore_index=True).to_csv(
        output_dir / "all_client_participation.csv", index=False
    )

    manifest = {
        "pilot_version": PILOT_VERSION,
        "stage": PILOT_STAGE,
        "purpose": "Compare E=5 and E=10 for computational feasibility and validation behaviour",
        "confirmatory_experiment": False,
        "test_partition_accessed": False,
        "evaluation_partition": "validation",
        "configurations": list(CONFIGURATIONS),
        "algorithms": list(ALGORITHMS),
        "seed": PILOT_SEED,
        "communication_rounds": PILOT_ROUNDS,
        "local_epoch_settings": list(LOCAL_EPOCH_SETTINGS),
        "learning_rate": canonical.LEARNING_RATE,
        "fedprox_mu": FEDPROX_MU,
        "threshold": canonical.PREDICTION_THRESHOLD,
        "ece_bins": canonical.CALIBRATION_BINS,
        "partition_preflight": preflight,
        "matched_initialization_checks": paired_checks,
        "run_count": len(summaries),
        "all_runs_completed": len(summaries) == 8,
        "total_wall_clock_seconds_including_output_writing": total_elapsed,
        "canonical_implementation_sha256": sha256_file(canonical_file),
        "evaluation_module_sha256": sha256_file(evaluator_file),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "device": str(canonical.DEVICE),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "metric_recording_note": (
            "Round trajectory is reconstructed from evaluator detailed records; "
            "AUPRC uses the evaluator field auprc_average_precision."
        ),
    }
    with (output_dir / "stage2_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, default=str)

    print("\nPHASE 5 STAGE 2 COMPLETED", flush=True)
    print(f"Runs completed: {len(summaries)}", flush=True)
    print(f"Total elapsed including output writing: {total_elapsed:.2f}s", flush=True)
    print(f"Outputs: {output_dir}", flush=True)
    print("These are pilot results, not confirmatory findings.", flush=True)
    return output_dir


if __name__ == "__main__":
    run_stage2()
