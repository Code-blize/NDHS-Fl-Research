import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, log_loss,
)
from sklearn.linear_model import LogisticRegression
from .model import NDHSMLP, set_parameters
from .reproducibility import REGION_CODES, REGION_NAMES

def _calibration(y, prob):
    eps = np.finfo(float).eps
    p = np.clip(np.asarray(prob, dtype=float), eps, 1 - eps)
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan"), "undefined_one_class"
    try:
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
        lr.fit(np.log(p / (1 - p)).reshape(-1, 1), y)
        return float(lr.intercept_[0]), float(lr.coef_[0, 0]), "estimated"
    except Exception as exc:
        return float("nan"), float("nan"), f"fit_failed:{type(exc).__name__}"

def _ece(y, prob, bins=10):
    y = np.asarray(y, dtype=int)
    prob = np.asarray(prob, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    value = 0.0
    for i in range(bins):
        mask = (prob >= edges[i]) & ((prob < edges[i + 1]) if i < bins - 1 else (prob <= edges[i + 1]))
        if mask.any():
            value += float(mask.mean()) * abs(float(y[mask].mean()) - float(prob[mask].mean()))
    return float(value)

def _metrics(y, prob):
    y = np.asarray(y, dtype=int)
    prob = np.asarray(prob, dtype=float)
    if len(y) == 0:
        raise ValueError("Cannot evaluate an empty partition")
    if not np.isfinite(prob).all() or np.any((prob < 0) | (prob > 1)):
        raise ValueError("Probabilities must be finite and in [0,1]")
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    two_classes = len(np.unique(y)) == 2
    ci, cs, calibration_status = _calibration(y, prob)
    return {
        "n": int(len(y)), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "auroc": float(roc_auc_score(y, prob)) if two_classes else float("nan"),
        "auprc_average_precision": float(average_precision_score(y, prob)) if two_classes else float("nan"),
        "brier": float(brier_score_loss(y, prob)),
        "binary_cross_entropy": float(log_loss(y, np.column_stack([1 - prob, prob]), labels=[0, 1])),
        "observed_prevalence": float(y.mean()),
        "predicted_prevalence": float(pred.mean()),
        "calibration_intercept": ci, "calibration_slope": cs,
        "calibration_status": calibration_status, "ece_10": _ece(y, prob),
    }

def evaluate_pooled_and_regions(parameters, validation_npz, input_dim):
    with np.load(validation_npz, allow_pickle=True) as z:
        X = np.asarray(z["X"], dtype=np.float32)
        y = np.asarray(z["y"], dtype=np.int64)
        regions = np.asarray(z["v024"], dtype=np.int64)
    if X.ndim != 2 or X.shape[1] != int(input_dim) or len(X) != len(y) or len(y) != len(regions):
        raise ValueError("Validation archive dimensions do not match the configured model")
    if not np.isfinite(X).all() or not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Invalid validation data")
    if set(np.unique(regions)) != set(REGION_CODES):
        raise ValueError("Validation data must contain all six regions")
    model = NDHSMLP(input_dim)
    set_parameters(model, parameters)
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.from_numpy(X))).numpy()
    rows = []
    pooled = _metrics(y, prob)
    pooled.update({"scope": "pooled", "region_code": 0, "region": "Pooled"})
    rows.append(pooled)
    for code in REGION_CODES:
        mask = regions == code
        row = _metrics(y[mask], prob[mask])
        row.update({"scope": "regional", "region_code": code, "region": REGION_NAMES[code]})
        rows.append(row)
    return rows
