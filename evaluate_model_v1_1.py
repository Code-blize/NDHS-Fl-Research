"""
NDHS Federated Learning — Evaluation Module v1.1
------------------------------------------------
Purpose:
    A separately versioned validation evaluator for pooled and six-region
    evaluation. This module does not train models and does not load the test
    partition.

Usage:
    1. Save this file as evaluate_model_v1_1.py.
    2. Import evaluate_model_v1_1 in the canonical implementation.
    3. Call it with a trained model and VALIDATION arrays only.
    4. Run this file directly to execute controlled tests.

Expected model interface:
    model(X_tensor) -> one logit per observation, shape (n,) or (n, 1).

Dependencies:
    numpy, pandas, torch, scipy

Metric conventions:
    - Threshold = 0.5.
    - AUPRC is average precision (sklearn-compatible definition).
    - ECE uses 10 equal-width probability bins on [0, 1].
    - Undefined metrics are NaN, never silently replaced with zero.
    - Calibration slope/intercept use unpenalized logistic recalibration
      of outcome on logit(predicted probability).
    - Confidence intervals and inferential comparisons are out of scope.
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


EVALUATOR_VERSION = "1.1"
DEFAULT_THRESHOLD = 0.5
DEFAULT_ECE_BINS = 10
DEFAULT_REGIONS = {
    1: "North Central",
    2: "North East",
    3: "North West",
    4: "South East",
    5: "South South",
    6: "South West",
}


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return NaN when the denominator is zero."""
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def _calibration_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    eps: float = 1e-7,
) -> tuple[float, float, str]:
    """
    Estimate calibration intercept and slope by logistic recalibration:
        logit(P(Y=1)) = intercept + slope * logit(predicted_probability)

    Returns (intercept, slope, status).
    A failed or non-identifiable fit is explicitly reported as NaN.
    """
    if np.unique(y_true).size < 2:
        return float("nan"), float("nan"), "undefined_single_class"

    p = np.clip(probabilities.astype(float), eps, 1.0 - eps)
    x = np.log(p / (1.0 - p))
    design = np.column_stack([np.ones(len(x)), x])

    if not np.all(np.isfinite(design)):
        return float("nan"), float("nan"), "failed_nonfinite_design"

    def objective(beta):
        linear_predictor = design @ beta
        # Stable negative log likelihood.
        loss = np.logaddexp(0.0, linear_predictor) - y_true * linear_predictor
        return float(np.sum(loss))

    def gradient(beta):
        linear_predictor = design @ beta
        return design.T @ (expit(linear_predictor) - y_true)

    try:
        result = minimize(
            objective,
            x0=np.array([0.0, 1.0]),
            jac=gradient,
            method="BFGS",
            options={"gtol": 1e-8, "maxiter": 2000},
        )
        beta = np.asarray(result.x, dtype=float)

        # A non-converged, non-finite, or extreme solution is not reported
        # as a valid calibration estimate.
        if (
            not result.success
            or beta.shape != (2,)
            or not np.all(np.isfinite(beta))
            or np.max(np.abs(beta)) > 1e6
        ):
            return float("nan"), float("nan"), "failed_optimizer"

        return float(beta[0]), float(beta[1]), "estimated"
    except Exception as exc:
        return float("nan"), float("nan"), f"failed_{type(exc).__name__}"


def _ece(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = DEFAULT_ECE_BINS,
) -> float:
    """Expected calibration error using equal-width bins."""
    if len(y_true) == 0:
        return float("nan")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_true)
    ece = 0.0

    for bin_index in range(n_bins):
        # Include p=1 in the final bin.
        if bin_index == n_bins - 1:
            mask = (
                (probabilities >= edges[bin_index])
                & (probabilities <= edges[bin_index + 1])
            )
        else:
            mask = (
                (probabilities >= edges[bin_index])
                & (probabilities < edges[bin_index + 1])
            )

        if np.any(mask):
            mean_probability = float(np.mean(probabilities[mask]))
            observed_frequency = float(np.mean(y_true[mask]))
            ece += (np.sum(mask) / total) * abs(
                observed_frequency - mean_probability
            )

    return float(ece)


def _calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    ece_bins: int,
) -> dict[str, Any]:
    """Calculate one complete metric record."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    if len(y_true) == 0:
        raise ValueError("Cannot evaluate an empty subset.")
    if y_true.shape != probabilities.shape:
        raise ValueError("y_true and probabilities must have identical shapes.")
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only binary values 0 and 1.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Predicted probabilities contain non-finite values.")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Predicted probabilities must be in [0, 1].")

    predictions = (probabilities >= threshold).astype(int)

    tn = int(np.sum((y_true == 0) & (predictions == 0)))
    fp = int(np.sum((y_true == 0) & (predictions == 1)))
    fn = int(np.sum((y_true == 1) & (predictions == 0)))
    tp = int(np.sum((y_true == 1) & (predictions == 1)))

    n = len(y_true)
    n_negative = tn + fp
    n_positive = tp + fn
    n_predicted_negative = tn + fn
    n_predicted_positive = tp + fp

    accuracy = (tp + tn) / n
    precision = _safe_divide(tp, n_predicted_positive)
    sensitivity = _safe_divide(tp, n_positive)
    specificity = _safe_divide(tn, n_negative)

    # Positive-class F1 is undefined when there are no actual positives.
    # Negative-class F1 is undefined when there are no actual negatives.
    f1_positive = (
        _safe_divide(2 * tp, 2 * tp + fp + fn)
        if n_positive > 0 else float("nan")
    )
    f1_negative = (
        _safe_divide(2 * tn, 2 * tn + fp + fn)
        if n_negative > 0 else float("nan")
    )

    if np.isnan(f1_positive) or np.isnan(f1_negative):
        macro_f1 = float("nan")
    else:
        macro_f1 = float((f1_positive + f1_negative) / 2.0)

    if np.isnan(sensitivity) or np.isnan(specificity):
        balanced_accuracy = float("nan")
    else:
        balanced_accuracy = float((sensitivity + specificity) / 2.0)

    if np.unique(y_true).size < 2:
        auroc = float("nan")
        auprc = float("nan")
        discrimination_status = "undefined_single_class"
    else:
        auroc = float(roc_auc_score(y_true, probabilities))
        auprc = float(average_precision_score(y_true, probabilities))
        discrimination_status = "estimated"

    # Clip only for numerical stability in log loss. Original probabilities
    # are retained for Brier score, ECE and discrimination metrics.
    eps = 1e-15
    p_for_log_loss = np.clip(probabilities, eps, 1.0 - eps)
    log_loss = float(
        -np.mean(
            y_true * np.log(p_for_log_loss)
            + (1 - y_true) * np.log(1.0 - p_for_log_loss)
        )
    )

    brier = float(np.mean((probabilities - y_true) ** 2))
    observed_prevalence = float(np.mean(y_true))
    predicted_positive_prevalence = float(np.mean(predictions))

    calibration_intercept, calibration_slope, calibration_status = (
        _calibration_metrics(y_true, probabilities)
    )

    return {
        "n": int(n),
        "n_actual_negative": int(n_negative),
        "n_actual_positive": int(n_positive),
        "n_predicted_negative": int(n_predicted_negative),
        "n_predicted_positive": int(n_predicted_positive),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "accuracy": float(accuracy),
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1_positive,
        "f1_negative_class": f1_negative,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "auroc": auroc,
        "auprc_average_precision": auprc,
        "brier_score": brier,
        "binary_cross_entropy_loss": log_loss,
        "observed_positive_prevalence": observed_prevalence,
        "predicted_positive_prevalence": predicted_positive_prevalence,
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
        "ece": _ece(y_true, probabilities, n_bins=ece_bins),
        "discrimination_status": discrimination_status,
        "calibration_status": calibration_status,
    }


def evaluate_model_v1_1(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    regions: np.ndarray,
    *,
    feature_configuration: str = "unspecified",
    algorithm: str = "unspecified",
    seed: int | None = None,
    communication_round: int | None = None,
    partition: str = "validation",
    threshold: float = DEFAULT_THRESHOLD,
    ece_bins: int = DEFAULT_ECE_BINS,
    region_names: dict[int, str] | None = None,
    device: str | torch.device = "cpu",
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Evaluate a model on one supplied partition.

    IMPORTANT:
        This function defaults to validation and rejects any partition label
        other than 'validation'. It does not load data or access a test set.

    Returns:
        metrics_df: pooled + six regional metric records.
        confusion_df: pooled + six regional confusion-count records.
        probabilities: predicted probabilities in input row order.
    """
    if partition.lower() != "validation":
        raise ValueError(
            "This evaluator is restricted to validation data. "
            f"Received partition={partition!r}."
        )
    if not (0.0 < threshold < 1.0):
        raise ValueError("threshold must be strictly between 0 and 1.")
    if ece_bins < 1:
        raise ValueError("ece_bins must be at least 1.")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).reshape(-1).astype(int)
    regions = np.asarray(regions).reshape(-1).astype(int)

    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional feature matrix.")
    if len(X) != len(y) or len(y) != len(regions):
        raise ValueError("X, y and regions must have matching row counts.")
    if len(y) == 0:
        raise ValueError("Validation data is empty.")
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y must contain only binary values 0 and 1.")

    names = region_names or DEFAULT_REGIONS
    expected_region_ids = list(names.keys())
    observed_region_ids = set(np.unique(regions).tolist())

    unknown_regions = observed_region_ids - set(expected_region_ids)
    if unknown_regions:
        raise ValueError(f"Unknown region IDs found: {sorted(unknown_regions)}")

    model_device = torch.device(device)
    model = model.to(model_device)
    model.eval()

    with torch.no_grad():
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=model_device)
        logits = model(X_tensor)

        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        logits = logits.reshape(-1)
        if logits.numel() != len(y):
            raise ValueError(
                "Model must return exactly one logit per observation."
            )

        probabilities = torch.sigmoid(logits).detach().cpu().numpy()

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Model generated non-finite probabilities.")

    records = []
    confusion_records = []

    subsets = [(0, "Pooled", np.ones(len(y), dtype=bool))]
    for region_id, region_name in names.items():
        subsets.append(
            (region_id, region_name, regions == region_id)
        )

    metadata = {
        "feature_configuration": feature_configuration,
        "algorithm": algorithm,
        "seed": seed,
        "communication_round": communication_round,
        "partition": "validation",
        "threshold": threshold,
        "evaluator_version": EVALUATOR_VERSION,
    }

    for region_id, region_name, mask in subsets:
        if not np.any(mask):
            raise ValueError(
                f"Validation subset is empty for region {region_name} "
                f"(region_id={region_id})."
            )

        metrics = _calculate_metrics(
            y[mask],
            probabilities[mask],
            threshold=threshold,
            ece_bins=ece_bins,
        )

        record = {
            **metadata,
            "region_id": region_id,
            "region": region_name,
            **metrics,
        }
        records.append(record)

        confusion_records.append({
            **metadata,
            "region_id": region_id,
            "region": region_name,
            "n": metrics["n"],
            "TN": metrics["TN"],
            "FP": metrics["FP"],
            "FN": metrics["FN"],
            "TP": metrics["TP"],
        })

    return (
        pd.DataFrame(records),
        pd.DataFrame(confusion_records),
        probabilities,
    )


def save_evaluation_outputs(
    metrics_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
    output_dir: str | Path,
    *,
    probabilities: np.ndarray | None = None,
    source_row_ids: np.ndarray | None = None,
    manifest_extra: dict[str, Any] | None = None,
) -> dict[str, str]:
    """
    Save evaluation outputs without overwriting prior versions by default.
    Use a new, empty output directory for each evaluation run.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "validation_metrics_v1_1.csv"
    confusion_path = output_dir / "validation_confusion_counts_v1_1.csv"
    manifest_path = output_dir / "evaluation_manifest_v1_1.json"

    if metrics_path.exists() or confusion_path.exists() or manifest_path.exists():
        raise FileExistsError(
            "At least one output already exists in this directory. "
            "Choose a new output directory to preserve previous results."
        )

    metrics_df.to_csv(metrics_path, index=False)
    confusion_df.to_csv(confusion_path, index=False)

    saved = {
        "metrics": str(metrics_path),
        "confusion_counts": str(confusion_path),
        "manifest": str(manifest_path),
    }

    if probabilities is not None:
        if source_row_ids is None:
            raise ValueError(
                "source_row_ids are required when saving row-level probabilities."
            )
        source_row_ids = np.asarray(source_row_ids).reshape(-1)
        probabilities = np.asarray(probabilities).reshape(-1)
        if len(source_row_ids) != len(probabilities):
            raise ValueError("source_row_ids and probabilities length mismatch.")

        probability_path = output_dir / "validation_probabilities_v1_1.csv"
        if probability_path.exists():
            raise FileExistsError(probability_path)
        pd.DataFrame({
            "source_row_id": source_row_ids,
            "predicted_probability": probabilities,
        }).to_csv(probability_path, index=False)
        saved["probabilities"] = str(probability_path)

    manifest = {
        "evaluator_version": EVALUATOR_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "partition_policy": "validation_only",
        "threshold": DEFAULT_THRESHOLD,
        "ece_bins": DEFAULT_ECE_BINS,
        "auprc_definition": "average_precision_score",
        "calibration_method": (
            "unpenalized logistic recalibration of outcome on "
            "logit(predicted probability)"
        ),
        "undefined_metric_policy": (
            "undefined metrics are stored as NaN with status fields"
        ),
        "confidence_intervals": "not calculated in evaluator",
        "statistical_comparisons": "out of scope",
        "additional_metadata": manifest_extra or {},
    }

    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, default=str)

    return saved


def run_controlled_tests() -> pd.DataFrame:
    """Run deterministic controlled tests for the v1.1 metric implementation."""

    # Controlled metric example previously used in Phase 4E.1.
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.8, 0.2, 0.9, 0.4, 0.7])
    m = _calculate_metrics(y, p, threshold=0.5, ece_bins=10)

    expected = {
        "TN": 2,
        "FP": 1,
        "FN": 1,
        "TP": 2,
        "accuracy": 2 / 3,
        "precision": 2 / 3,
        "sensitivity": 2 / 3,
        "specificity": 2 / 3,
        "f1": 2 / 3,
        "macro_f1": 2 / 3,
        "balanced_accuracy": 2 / 3,
        "auroc": 7 / 9,
        "auprc_average_precision": 29 / 36,
        "brier_score": 1.15 / 6,
        "observed_positive_prevalence": 0.5,
        "predicted_positive_prevalence": 0.5,
    }

    rows = []
    for metric, expected_value in expected.items():
        observed = m[metric]
        passed = bool(np.isclose(observed, expected_value, atol=1e-10))
        rows.append({
            "test": f"controlled_{metric}",
            "expected": expected_value,
            "observed": observed,
            "passed": passed,
        })

    # Single-class: AUROC/AUPRC, sensitivity or specificity may be undefined.
    single_y = np.array([0, 0, 0])
    single_p = np.array([0.1, 0.2, 0.3])
    single = _calculate_metrics(
        single_y, single_p, threshold=0.5, ece_bins=10
    )

    for metric in [
        "auroc", "auprc_average_precision",
        "sensitivity", "macro_f1", "balanced_accuracy",
        "calibration_intercept", "calibration_slope",
    ]:
        rows.append({
            "test": f"single_class_{metric}_is_nan",
            "expected": "NaN",
            "observed": single[metric],
            "passed": bool(np.isnan(single[metric])),
        })

    # No predicted positives: precision is undefined.
    no_positive_predictions = _calculate_metrics(
        np.array([0, 1, 1]),
        np.array([0.1, 0.2, 0.3]),
        threshold=0.5,
        ece_bins=10,
    )
    rows.append({
        "test": "no_predicted_positives_precision_is_nan",
        "expected": "NaN",
        "observed": no_positive_predictions["precision"],
        "passed": bool(np.isnan(no_positive_predictions["precision"])),
    })

    # Actual evaluator interface: six regions and one single-class region.
    class ControlledModel(torch.nn.Module):
        def __init__(self, logits):
            super().__init__()
            self.register_buffer(
                "logits",
                torch.tensor(logits, dtype=torch.float32)
            )

        def forward(self, X):
            return self.logits

    region_y = np.array([
        0, 0,       # region 1: single class
        0, 1,
        0, 1,
        0, 1,
        0, 1,
        0, 1,
    ])
    region_ids = np.repeat(np.arange(1, 7), 2)
    region_p = np.array([
        0.1, 0.2,
        0.1, 0.8,
        0.2, 0.9,
        0.1, 0.9,
        0.4, 0.7,
        0.2, 0.8,
    ])
    region_logits = np.log(region_p / (1 - region_p))
    model = ControlledModel(region_logits)
    X_dummy = np.zeros((len(region_y), 1), dtype=np.float32)

    metrics_df, confusion_df, returned_p = evaluate_model_v1_1(
        model,
        X_dummy,
        region_y,
        region_ids,
        feature_configuration="controlled_test",
        algorithm="controlled_test",
        seed=1,
        communication_round=1,
        partition="validation",
    )

    rows.append({
        "test": "actual_evaluator_returns_7_rows",
        "expected": 7,
        "observed": len(metrics_df),
        "passed": len(metrics_df) == 7,
    })
    rows.append({
        "test": "actual_evaluator_returns_7_confusion_rows",
        "expected": 7,
        "observed": len(confusion_df),
        "passed": len(confusion_df) == 7,
    })
    rows.append({
        "test": "actual_evaluator_returns_probabilities_in_row_order",
        "expected": len(region_y),
        "observed": len(returned_p),
        "passed": (
            len(returned_p) == len(region_y)
            and np.allclose(returned_p, region_p, atol=1e-6)
        ),
    })

    north_central = metrics_df.loc[
        metrics_df["region_id"] == 1
    ].iloc[0]
    rows.append({
        "test": "single_class_region_auroc_is_nan",
        "expected": "NaN",
        "observed": north_central["auroc"],
        "passed": bool(np.isnan(north_central["auroc"])),
    })
    rows.append({
        "test": "single_class_region_accuracy_is_defined",
        "expected": 1.0,
        "observed": north_central["accuracy"],
        "passed": bool(np.isclose(north_central["accuracy"], 1.0)),
    })

    # Explicitly reject test partition labels.
    try:
        evaluate_model_v1_1(
            model, X_dummy, region_y, region_ids, partition="test"
        )
        rejected_test_partition = False
    except ValueError:
        rejected_test_partition = True

    rows.append({
        "test": "test_partition_label_rejected",
        "expected": True,
        "observed": rejected_test_partition,
        "passed": rejected_test_partition,
    })

    report = pd.DataFrame(rows)
    if not report["passed"].all():
        failed = report.loc[~report["passed"]]
        raise AssertionError(
            "Controlled v1.1 tests failed:\n" + failed.to_string(index=False)
        )

    return report


if __name__ == "__main__":
    report = run_controlled_tests()
    print("\nEVALUATOR v1.1 CONTROLLED TEST REPORT")
    print("=" * 88)
    print(report.to_string(index=False))
    print("\nAll evaluator v1.1 controlled tests passed.")
