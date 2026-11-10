
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)

# ============================================================
# PHASE 4E.1 — CONTROLLED EVALUATION METRIC VALIDATION
# ============================================================

THRESHOLD = 0.5
TOLERANCE = 1e-10

# Known outcomes
y_true = np.array([0, 0, 0, 1, 1, 1])

# Known predicted probabilities
probabilities = np.array([
    0.1, 0.8, 0.2,
    0.9, 0.4, 0.7
])

predictions = (
    probabilities >= THRESHOLD
).astype(int)

# ============================================================
# 1. CONFUSION MATRIX
# ============================================================

tn, fp, fn, tp = confusion_matrix(
    y_true,
    predictions,
    labels=[0, 1]
).ravel()

# Expected:
# TN = 2, FP = 1, FN = 1, TP = 2

assert (tn, fp, fn, tp) == (2, 1, 1, 2)

# ============================================================
# 2. CLASSIFICATION METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    predictions
)

precision = precision_score(
    y_true,
    predictions
)

sensitivity = recall_score(
    y_true,
    predictions
)

specificity = tn / (tn + fp)

f1 = f1_score(
    y_true,
    predictions
)

macro_f1 = f1_score(
    y_true,
    predictions,
    average="macro"
)

balanced_accuracy = balanced_accuracy_score(
    y_true,
    predictions
)

# ============================================================
# 3. PROBABILITY METRICS
# ============================================================

auroc = roc_auc_score(
    y_true,
    probabilities
)

auprc = average_precision_score(
    y_true,
    probabilities
)

brier = brier_score_loss(
    y_true,
    probabilities
)

# ============================================================
# 4. MANUAL REFERENCE VALUES
# ============================================================

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
    "auprc": 29 / 36,
    "brier_score": 1.15 / 6,

    "observed_prevalence": 0.5,
    "predicted_positive_prevalence": 0.5

}

observed = {

    "TN": tn,
    "FP": fp,
    "FN": fn,
    "TP": tp,

    "accuracy": accuracy,
    "precision": precision,
    "sensitivity": sensitivity,
    "specificity": specificity,
    "f1": f1,
    "macro_f1": macro_f1,
    "balanced_accuracy": balanced_accuracy,

    "auroc": auroc,
    "auprc": auprc,
    "brier_score": brier,

    "observed_prevalence": y_true.mean(),
    "predicted_positive_prevalence": predictions.mean()

}

# ============================================================
# 5. AUTOMATED ASSERTIONS
# ============================================================

validation_results = []

for metric, expected_value in expected.items():

    observed_value = observed[metric]

    passed = np.isclose(
        observed_value,
        expected_value,
        atol=TOLERANCE,
        rtol=0
    )

    validation_results.append({

        "metric": metric,
        "expected": expected_value,
        "observed": observed_value,
        "passed": bool(passed)

    })

    assert passed, (
        f"FAILED: {metric}. "
        f"Expected {expected_value}, "
        f"observed {observed_value}"
    )

# ============================================================
# 6. SINGLE-CLASS EDGE CASE
# ============================================================

single_class_y = np.array([0, 0, 0, 0])

single_class_probabilities = np.array([
    0.1, 0.2, 0.3, 0.4
])

single_class_auroc = (
    roc_auc_score(
        single_class_y,
        single_class_probabilities
    )
    if len(np.unique(single_class_y)) == 2
    else np.nan
)

single_class_auprc = (
    average_precision_score(
        single_class_y,
        single_class_probabilities
    )
    if len(np.unique(single_class_y)) == 2
    else np.nan
)

assert np.isnan(single_class_auroc)
assert np.isnan(single_class_auprc)

print("\nSINGLE-CLASS EDGE CASE")
print("AUROC:", single_class_auroc)
print("AUPRC:", single_class_auprc)

# ============================================================
# 7. RESULTS
# ============================================================

results_df = pd.DataFrame(
    validation_results
)

print("\nCONTROLLED METRIC VALIDATION")
print("=" * 75)

print(
    results_df.to_string(index=False)
)

print("\nAll controlled metric assertions passed.")

print(
    "\nSingle-class AUROC and AUPRC correctly "
    "reported as undefined (NaN)."
)

print("\nPHASE 4E.1 PASSED")
