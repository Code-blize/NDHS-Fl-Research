"""
Independent audit of evaluate_model_v1_1.py
-------------------------------------------
Run from the same folder as evaluate_model_v1_1.py:
    python audit_evaluation_v1_1.py

This script does not modify the evaluator or train a model.
It independently checks:
  1. Calibration intercept/slope using score-equation root finding.
  2. Binary cross-entropy against sklearn.metrics.log_loss.
  3. ECE using an independently written bin assignment.
  4. CSV/manifest output integrity and overwrite protection.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import root
from scipy.special import expit
from sklearn.metrics import log_loss

import evaluate_model_v1_1 as evaluator


def independent_calibration_fit(y, probabilities, eps=1e-7):
    """Solve logistic recalibration score equations with scipy.root."""
    p = np.clip(np.asarray(probabilities, dtype=float), eps, 1 - eps)
    y = np.asarray(y, dtype=float)
    z = np.log(p / (1 - p))
    design = np.column_stack([np.ones(len(z)), z])

    def score_equations(beta):
        fitted = expit(design @ beta)
        return design.T @ (y - fitted)

    result = root(score_equations, np.array([0.0, 1.0]), method="hybr")
    if not result.success or not np.all(np.isfinite(result.x)):
        raise AssertionError(
            f"Independent calibration fit failed: {result.message}"
        )
    if np.max(np.abs(score_equations(result.x))) > 1e-5:
        raise AssertionError("Independent calibration score residual too large.")
    return float(result.x[0]), float(result.x[1])


def independent_ece(y, probabilities, bins=10):
    """Independent ECE using integer bin assignment."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(probabilities, dtype=float)

    # Bins are [0,.1), [.1,.2), ... [.9,1], with p=1 in final bin.
    bin_ids = np.minimum((p * bins).astype(int), bins - 1)
    total_error = 0.0

    for b in range(bins):
        selected = bin_ids == b
        if selected.any():
            confidence = p[selected].mean()
            frequency = y[selected].mean()
            total_error += selected.sum() / len(y) * abs(
                frequency - confidence
            )
    return float(total_error)


class FixedLogitModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer(
            "fixed_logits",
            torch.tensor(logits, dtype=torch.float32)
        )

    def forward(self, X):
        return self.fixed_logits


def main():
    checks = []

    # Non-perfect predictions and both classes, repeated to provide
    # enough observations for a stable calibration fit.
    y_one = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    p_one = np.array([0.1, 0.8, 0.2, 0.9, 0.4, 0.7], dtype=float)
    y = np.tile(y_one, 20)
    probabilities = np.tile(p_one, 20)

    # 1. Calibration: compare independent score-equation solution.
    expected_intercept, expected_slope = independent_calibration_fit(
        y, probabilities
    )
    observed_intercept, observed_slope, status = (
        evaluator._calibration_metrics(y, probabilities)
    )

    checks.append({
        "check": "calibration_fit_status",
        "expected": "estimated",
        "observed": status,
        "passed": status == "estimated",
    })
    checks.append({
        "check": "calibration_intercept",
        "expected": expected_intercept,
        "observed": observed_intercept,
        "passed": np.isclose(
            expected_intercept, observed_intercept, atol=1e-5
        ),
    })
    checks.append({
        "check": "calibration_slope",
        "expected": expected_slope,
        "observed": observed_slope,
        "passed": np.isclose(
            expected_slope, observed_slope, atol=1e-5
        ),
    })

    # 2. Log-loss: compare with sklearn's independent implementation.
    observed_metrics = evaluator._calculate_metrics(
        y, probabilities, threshold=0.5, ece_bins=10
    )
    expected_logloss = log_loss(
        y, probabilities, labels=[0, 1]
    )
    checks.append({
        "check": "binary_cross_entropy_vs_sklearn_log_loss",
        "expected": expected_logloss,
        "observed": observed_metrics["binary_cross_entropy_loss"],
        "passed": np.isclose(
            expected_logloss,
            observed_metrics["binary_cross_entropy_loss"],
            atol=1e-12,
        ),
    })

    # 3. ECE: independent bin assignment.
    expected_ece = independent_ece(y, probabilities, bins=10)
    checks.append({
        "check": "ece_independent_bin_calculation",
        "expected": expected_ece,
        "observed": observed_metrics["ece"],
        "passed": np.isclose(
            expected_ece, observed_metrics["ece"], atol=1e-12
        ),
    })

    # 4. Exercise public evaluator and output saver.
    # Six regions, each receives 20 repetitions of the 6-row pattern.
    y_all = np.tile(y_one, 20 * 6)
    p_all = np.tile(p_one, 20 * 6)
    region_ids = np.repeat(np.arange(1, 7), len(y) )
    logits_all = np.log(p_all / (1 - p_all))
    X = np.zeros((len(y_all), 1), dtype=np.float32)
    model = FixedLogitModel(logits_all)

    metrics_df, confusion_df, returned_probabilities = (
        evaluator.evaluate_model_v1_1(
            model,
            X,
            y_all,
            region_ids,
            feature_configuration="audit_controlled",
            algorithm="audit_controlled",
            seed=2026,
            communication_round=1,
            partition="validation",
        )
    )

    checks.append({
        "check": "output_metrics_has_7_rows",
        "expected": 7,
        "observed": len(metrics_df),
        "passed": len(metrics_df) == 7,
    })
    checks.append({
        "check": "output_confusion_has_7_rows",
        "expected": 7,
        "observed": len(confusion_df),
        "passed": len(confusion_df) == 7,
    })

    with tempfile.TemporaryDirectory(prefix="ndhs_eval_v11_audit_") as tmp:
        output_dir = Path(tmp) / "fresh_output"
        ids = np.arange(len(returned_probabilities))

        saved = evaluator.save_evaluation_outputs(
            metrics_df,
            confusion_df,
            output_dir,
            probabilities=returned_probabilities,
            source_row_ids=ids,
            manifest_extra={"audit_purpose": "controlled_output_integrity"},
        )

        metrics_read = pd.read_csv(saved["metrics"])
        confusion_read = pd.read_csv(saved["confusion_counts"])
        probabilities_read = pd.read_csv(saved["probabilities"])
        with open(saved["manifest"], "r", encoding="utf-8") as f:
            manifest = json.load(f)

        checks.append({
            "check": "saved_metrics_csv_rows",
            "expected": 7,
            "observed": len(metrics_read),
            "passed": len(metrics_read) == 7,
        })
        checks.append({
            "check": "saved_confusion_csv_rows",
            "expected": 7,
            "observed": len(confusion_read),
            "passed": len(confusion_read) == 7,
        })
        checks.append({
            "check": "saved_probability_csv_rows",
            "expected": len(returned_probabilities),
            "observed": len(probabilities_read),
            "passed": len(probabilities_read) == len(returned_probabilities),
        })
        checks.append({
            "check": "manifest_evaluator_version",
            "expected": "1.1",
            "observed": manifest.get("evaluator_version"),
            "passed": manifest.get("evaluator_version") == "1.1",
        })
        checks.append({
            "check": "manifest_validation_only",
            "expected": "validation_only",
            "observed": manifest.get("partition_policy"),
            "passed": manifest.get("partition_policy") == "validation_only",
        })

        # The same output directory must not be overwritten.
        try:
            evaluator.save_evaluation_outputs(
                metrics_df, confusion_df, output_dir
            )
            overwrite_rejected = False
        except FileExistsError:
            overwrite_rejected = True

        checks.append({
            "check": "existing_outputs_not_overwritten",
            "expected": True,
            "observed": overwrite_rejected,
            "passed": overwrite_rejected,
        })

    report = pd.DataFrame(checks)
    print("\nPHASE 4E.4 — INDEPENDENT EVALUATOR AUDIT")
    print("=" * 100)
    print(report.to_string(index=False))

    if not report["passed"].all():
        failed = report.loc[~report["passed"]]
        raise AssertionError(
            "\nIndependent audit failures:\n" + failed.to_string(index=False)
        )

    print("\nALL INDEPENDENT AUDIT CHECKS PASSED.")


if __name__ == "__main__":
    main()
