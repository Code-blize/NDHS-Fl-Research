
# ================================================================
# NDHS FEDERATED LEARNING RESEARCH
# CANONICAL FEDERATED IMPLEMENTATION — VERSION 1.0
#
# Algorithms: FedAvg and FedProx
# Clients: Nigeria's six geopolitical zones
# Architecture: Input -> 16 -> 8 -> 1
#
# Purpose:
#   1. Establish one canonical FL implementation.
#   2. Implement reproducibility controls.
#   3. Run DET-01, DET-02 and DET-03.
#   4. Validate pooled and regional evaluation metrics.
#   5. Save implementation and test reports.
#
# IMPORTANT:
#   - This is NOT a confirmatory experiment.
#   - The test partition is never loaded.
#   - Existing historical and pilot files are not overwritten.
#   - No hyperparameter tuning is performed.
# ================================================================


# ================================================================
# 1. IMPORTS
# ================================================================

from pathlib import Path
from datetime import datetime, timezone
import json
import random
import hashlib
import platform
import copy

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

import sklearn
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


# ================================================================
# 2. PROJECT CONFIGURATION
# ================================================================

ROOT = Path(
    r"C:\Users\usher\Documents\NDHS-Fl-Research\NDHS_FL_Research"
)

DATA_DIR = ROOT / "05_preprocessed"

# New versioned output directory.
# Does not overwrite historical pilot outputs.
OUTPUT_DIR = ROOT / "08_canonical_implementation_v1"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VERSION = "1.0"

DEVICE = torch.device("cpu")

CONFIGURATIONS = {
    "baseline": 15,
    "expanded": 51
}

REGIONS = {
    1: "North Central",
    2: "North East",
    3: "North West",
    4: "South East",
    5: "South South",
    6: "South West"
}

# Approved pilot settings.
SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 0.001
FEDPROX_MU = 0.01

# Sanity-test settings.
SANITY_ROUNDS = 1
SANITY_LOCAL_EPOCHS = 1

# Evaluation settings.
PREDICTION_THRESHOLD = 0.5
CALIBRATION_BINS = 10

# Numerical tolerance.
PARAMETER_TOLERANCE = 1e-7

# This implementation retains the existing optimizer lifecycle:
# Adam is recreated for each client in each communication round.


# ================================================================
# 3. REPRODUCIBILITY CONTROLS
# ================================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)


def get_environment():

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "device": str(DEVICE),
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        )
    }


# ================================================================
# 4. DATA LOADING
# ================================================================

def load_partition(configuration, partition):

    if partition not in ["train", "validation"]:
        raise ValueError(
            "Only training and validation partitions "
            "are permitted in this implementation."
        )

    path = DATA_DIR / f"{configuration}_{partition}.npz"

    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=True) as data:

        X = np.asarray(
            data["X"],
            dtype=np.float32
        )

        y = np.asarray(
            data["y"],
            dtype=np.float32
        ).reshape(-1)

        regions = np.asarray(
            data["v024"],
            dtype=np.int64
        )

    assert X.ndim == 2
    assert len(X) == len(y) == len(regions)

    assert np.isfinite(X).all()
    assert np.isfinite(y).all()

    assert set(np.unique(y)).issubset({0.0, 1.0})

    assert set(np.unique(regions)).issubset(
        set(REGIONS.keys())
    )

    return X, y, regions


def validate_training_data(X, y, regions, expected_features):

    assert X.shape[1] == expected_features

    assert set(np.unique(regions)) == set(REGIONS.keys())

    assert set(np.unique(y)) == {0.0, 1.0}

    for region_id in REGIONS:

        mask = regions == region_id

        assert mask.sum() > 0

        assert set(np.unique(y[mask])) == {0.0, 1.0}


# ================================================================
# 5. CANONICAL MLP ARCHITECTURE
# ================================================================

class MLP(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_dim, 16),

            nn.ReLU(),

            nn.Linear(16, 8),

            nn.ReLU(),

            nn.Linear(8, 1)

        )

    def forward(self, x):

        return self.network(x).squeeze(1)


def create_model(input_dim, seed):

    set_seed(seed)

    model = MLP(input_dim).to(DEVICE)

    return model


# ================================================================
# 6. INITIAL PARAMETER COMPARISON
# ================================================================

def compare_model_parameters(model_a, model_b):

    state_a = model_a.state_dict()
    state_b = model_b.state_dict()

    if state_a.keys() != state_b.keys():
        return False, float("inf")

    differences = []

    for key in state_a:

        if state_a[key].shape != state_b[key].shape:
            return False, float("inf")

        difference = torch.max(
            torch.abs(state_a[key] - state_b[key])
        ).item()

        differences.append(difference)

    maximum_difference = max(differences)

    return (
        maximum_difference <= PARAMETER_TOLERANCE,
        maximum_difference
    )


def compare_state_dicts(state_a, state_b):

    if state_a.keys() != state_b.keys():
        return False, float("inf")

    differences = []

    for key in state_a:

        difference = torch.max(
            torch.abs(state_a[key] - state_b[key])
        ).item()

        differences.append(difference)

    maximum_difference = max(differences)

    return (
        maximum_difference <= PARAMETER_TOLERANCE,
        maximum_difference
    )


# ================================================================
# 7. CANONICAL CLIENT TRAINING
# ================================================================

def train_client(
    global_model,
    X_client,
    y_client,
    mu,
    client_seed,
    local_epochs
):

    if mu < 0:
        raise ValueError("FedProx mu cannot be negative.")

    set_seed(client_seed)

    X_tensor = torch.tensor(
        X_client,
        dtype=torch.float32
    )

    y_tensor = torch.tensor(
        y_client,
        dtype=torch.float32
    )

    generator = torch.Generator()

    generator.manual_seed(client_seed)

    loader = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0
    )

    # Initialize local model from global parameters.
    local_model = MLP(
        X_client.shape[1]
    ).to(DEVICE)

    local_model.load_state_dict(
        global_model.state_dict()
    )

    # Fixed global reference parameters.
    global_parameters = [
        parameter.detach().clone()
        for parameter in global_model.parameters()
    ]

    # Approved optimizer lifecycle:
    # A new Adam optimizer for each client-round.
    optimizer = torch.optim.Adam(
        local_model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.BCEWithLogitsLoss()

    local_model.train()

    for epoch in range(local_epochs):

        for batch_X, batch_y in loader:

            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            logits = local_model(batch_X)

            loss = criterion(logits, batch_y)

            # FedProx proximal term.
            if mu > 0:

                proximal_penalty = torch.zeros(
                    (),
                    device=DEVICE
                )

                for local_parameter, global_parameter in zip(
                    local_model.parameters(),
                    global_parameters
                ):

                    proximal_penalty += torch.sum(
                        (local_parameter - global_parameter) ** 2
                    )

                loss = loss + (
                    mu / 2.0
                ) * proximal_penalty

            loss.backward()

            optimizer.step()

    return copy.deepcopy(
        local_model.state_dict()
    )


# ================================================================
# 8. CANONICAL GLOBAL AGGREGATION
# ================================================================

def aggregate_models(client_states, client_sizes):

    if len(client_states) != 6:
        raise ValueError(
            "Exactly six client updates are required."
        )

    if len(client_sizes) != 6:
        raise ValueError(
            "Exactly six client sample sizes are required."
        )

    if any(size <= 0 for size in client_sizes):
        raise ValueError(
            "Every client must have a positive sample size."
        )

    total_samples = sum(client_sizes)

    if total_samples <= 0:
        raise ValueError("Invalid total sample size.")

    reference_keys = client_states[0].keys()

    for state in client_states:

        if state.keys() != reference_keys:
            raise ValueError(
                "Client parameter keys do not match."
            )

    global_state = {}

    for key in reference_keys:

        reference_tensor = client_states[0][key]

        for state in client_states:

            if state[key].shape != reference_tensor.shape:
                raise ValueError(
                    f"Parameter shape mismatch: {key}"
                )

            if not torch.isfinite(state[key]).all():
                raise ValueError(
                    f"Non-finite client parameter: {key}"
                )

        aggregated_tensor = torch.zeros_like(
            reference_tensor
        )

        for state, size in zip(
            client_states,
            client_sizes
        ):

            weight = size / total_samples

            aggregated_tensor += (
                state[key] * weight
            )

        if not torch.isfinite(aggregated_tensor).all():
            raise ValueError(
                f"Non-finite aggregated parameter: {key}"
            )

        global_state[key] = aggregated_tensor

    return global_state


# ================================================================
# 9. CALIBRATION SUMMARY
# ================================================================

def calculate_calibration(y_true, probabilities):

    y_true = np.asarray(y_true, dtype=int)

    probabilities = np.asarray(
        probabilities,
        dtype=float
    )

    bin_edges = np.linspace(
        0.0,
        1.0,
        CALIBRATION_BINS + 1
    )

    bin_results = []

    weighted_absolute_error = 0.0

    for bin_index in range(CALIBRATION_BINS):

        lower = bin_edges[bin_index]
        upper = bin_edges[bin_index + 1]

        if bin_index == CALIBRATION_BINS - 1:

            mask = (
                (probabilities >= lower)
                & (probabilities <= upper)
            )

        else:

            mask = (
                (probabilities >= lower)
                & (probabilities < upper)
            )

        count = int(mask.sum())

        if count == 0:

            mean_probability = np.nan
            observed_frequency = np.nan

        else:

            mean_probability = float(
                probabilities[mask].mean()
            )

            observed_frequency = float(
                y_true[mask].mean()
            )

            weighted_absolute_error += (
                count / len(y_true)
            ) * abs(
                mean_probability - observed_frequency
            )

        bin_results.append({

            "bin": bin_index + 1,
            "lower_probability": lower,
            "upper_probability": upper,
            "n": count,
            "mean_predicted_probability": mean_probability,
            "observed_positive_frequency": observed_frequency

        })

    return (
        float(weighted_absolute_error),
        bin_results
    )


# ================================================================
# 10. COMPLETE EVALUATION METRICS
# ================================================================

def calculate_metrics(y_true, probabilities):

    y_true = np.asarray(
        y_true,
        dtype=int
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float
    )

    if len(y_true) == 0:
        raise ValueError("Cannot evaluate an empty dataset.")

    if len(y_true) != len(probabilities):
        raise ValueError(
            "Outcome and probability lengths differ."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Non-finite predicted probabilities."
        )

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(
            "Predicted probabilities must be between 0 and 1."
        )

    predictions = (
        probabilities >= PREDICTION_THRESHOLD
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    sensitivity = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        predictions,
        labels=[0, 1],
        average="macro",
        zero_division=0
    )

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        predictions
    )

    # AUROC requires both outcome classes.
    if len(np.unique(y_true)) == 2:

        auroc = roc_auc_score(
            y_true,
            probabilities
        )

        auprc = average_precision_score(
            y_true,
            probabilities
        )

    else:

        auroc = np.nan
        auprc = np.nan

    brier = brier_score_loss(
        y_true,
        probabilities
    )

    ece, calibration_bins = calculate_calibration(
        y_true,
        probabilities
    )

    observed_prevalence = float(
        y_true.mean()
    )

    predicted_prevalence = float(
        predictions.mean()
    )

    return {

        "n": len(y_true),

        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),

        "accuracy": float(accuracy),

        "precision": float(precision),

        "sensitivity_recall": float(sensitivity),

        "specificity": float(specificity),

        "f1": float(f1),

        "macro_f1": float(macro_f1),

        "balanced_accuracy": float(
            balanced_accuracy
        ),

        "auroc": float(auroc),

        "auprc": float(auprc),

        "brier_score": float(brier),

        "observed_positive_prevalence": (
            observed_prevalence
        ),

        "predicted_positive_prevalence": (
            predicted_prevalence
        ),

        "mean_predicted_probability": float(
            probabilities.mean()
        ),

        "calibration_ece_10_bins": ece,

        "prediction_threshold": (
            PREDICTION_THRESHOLD
        ),

        "calibration_method": (
            "Expected calibration error using "
            "10 equal-width probability bins"
        ),

        "calibration_bins": calibration_bins

    }


# ================================================================
# 11. POOLED AND REGIONAL EVALUATION
# ================================================================

def evaluate_model(
    model,
    X,
    y,
    regions,
    configuration,
    algorithm,
    seed,
    round_number
):

    model.eval()

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32
    ).to(DEVICE)

    with torch.no_grad():

        logits = model(X_tensor)

        probabilities = torch.sigmoid(
            logits
        ).cpu().numpy()

    results = []

    # Pooled evaluation.
    pooled_metrics = calculate_metrics(
        y,
        probabilities
    )

    results.append({

        "configuration": configuration,
        "algorithm": algorithm,
        "seed": seed,
        "round": round_number,
        "region_id": 0,
        "region": "Pooled",
        **{
            key: value
            for key, value in pooled_metrics.items()
            if key != "calibration_bins"
        }

    })

    calibration_results = []

    for item in pooled_metrics["calibration_bins"]:

        calibration_results.append({

            "configuration": configuration,
            "algorithm": algorithm,
            "seed": seed,
            "round": round_number,
            "region": "Pooled",
            **item

        })

    # Regional evaluation.
    for region_id, region_name in REGIONS.items():

        mask = regions == region_id

        if mask.sum() == 0:
            raise ValueError(
                f"Empty validation region: {region_name}"
            )

        regional_metrics = calculate_metrics(
            y[mask],
            probabilities[mask]
        )

        results.append({

            "configuration": configuration,
            "algorithm": algorithm,
            "seed": seed,
            "round": round_number,
            "region_id": region_id,
            "region": region_name,
            **{
                key: value
                for key, value in regional_metrics.items()
                if key != "calibration_bins"
            }

        })

        for item in regional_metrics["calibration_bins"]:

            calibration_results.append({

                "configuration": configuration,
                "algorithm": algorithm,
                "seed": seed,
                "round": round_number,
                "region": region_name,
                **item

            })

    return results, calibration_results


# ================================================================
# 12. CANONICAL FEDERATED EXPERIMENT FUNCTION
# ================================================================

def run_federated_experiment(
    configuration,
    seed,
    algorithm,
    rounds,
    local_epochs,
    mu
):

    if algorithm not in ["FedAvg", "FedProx"]:
        raise ValueError("Unknown algorithm.")

    if algorithm == "FedAvg" and mu != 0:
        raise ValueError(
            "FedAvg must use mu = 0."
        )

    if algorithm == "FedProx" and mu < 0:
        raise ValueError(
            "FedProx mu cannot be negative."
        )

    X_train, y_train, train_regions = load_partition(
        configuration,
        "train"
    )

    X_val, y_val, val_regions = load_partition(
        configuration,
        "validation"
    )

    expected_features = CONFIGURATIONS[configuration]

    validate_training_data(
        X_train,
        y_train,
        train_regions,
        expected_features
    )

    if X_val.shape[1] != expected_features:
        raise ValueError(
            "Validation feature dimension mismatch."
        )

    if not np.isfinite(X_val).all():
        raise ValueError(
            "Validation matrix contains non-finite values."
        )

    input_dim = X_train.shape[1]

    global_model = create_model(
        input_dim,
        seed
    )

    initial_state = copy.deepcopy(
        global_model.state_dict()
    )

    round_records = []
    evaluation_records = []
    calibration_records = []
    participation_records = []

    for round_number in range(1, rounds + 1):

        client_states = []
        client_sizes = []

        # Fixed order: region IDs 1 through 6.
        for region_id in REGIONS:

            mask = train_regions == region_id

            X_client = X_train[mask]
            y_client = y_train[mask]

            client_seed = (
                seed
                + round_number * 1000
                + region_id
            )

            client_state = train_client(
                global_model=global_model,
                X_client=X_client,
                y_client=y_client,
                mu=mu,
                client_seed=client_seed,
                local_epochs=local_epochs
            )

            client_states.append(
                client_state
            )

            client_sizes.append(
                len(y_client)
            )

            participation_records.append({

                "configuration": configuration,
                "algorithm": algorithm,
                "seed": seed,
                "round": round_number,
                "region_id": region_id,
                "region": REGIONS[region_id],
                "training_n": len(y_client),
                "client_seed": client_seed

            })

        # Verify six participating clients.
        assert len(client_states) == 6

        # Sample-size-weighted aggregation.
        new_global_state = aggregate_models(
            client_states,
            client_sizes
        )

        global_model.load_state_dict(
            new_global_state
        )

        # Validation evaluation only.
        round_evaluations, round_calibration = evaluate_model(
            model=global_model,
            X=X_val,
            y=y_val,
            regions=val_regions,
            configuration=configuration,
            algorithm=algorithm,
            seed=seed,
            round_number=round_number
        )

        evaluation_records.extend(
            round_evaluations
        )

        calibration_records.extend(
            round_calibration
        )

        pooled = next(
            item for item in round_evaluations
            if item["region"] == "Pooled"
        )

        round_records.append({

            "configuration": configuration,
            "algorithm": algorithm,
            "seed": seed,
            "round": round_number,
            "mu": mu,
            "local_epochs": local_epochs,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "validation_loss_not_calculated": True,
            "validation_accuracy": pooled["accuracy"],
            "validation_macro_f1": pooled["macro_f1"],
            "validation_auroc": pooled["auroc"],
            "validation_auprc": pooled["auprc"]

        })

    return {

        "model": global_model,
        "initial_state": initial_state,
        "final_state": copy.deepcopy(
            global_model.state_dict()
        ),
        "round_records": round_records,
        "evaluation_records": evaluation_records,
        "calibration_records": calibration_records,
        "participation_records": participation_records

    }


# ================================================================
# 13. DET-03 — INITIAL PARAMETER EQUALITY
# ================================================================

def run_det03(configuration):

    input_dim = CONFIGURATIONS[configuration]

    model_fedavg = create_model(
        input_dim,
        SEED
    )

    model_fedprox = create_model(
        input_dim,
        SEED
    )

    passed, maximum_difference = compare_model_parameters(
        model_fedavg,
        model_fedprox
    )

    return {

        "test_id": "DET-03",
        "configuration": configuration,
        "test": "Matched initial model parameters",
        "passed": passed,
        "maximum_parameter_difference": maximum_difference,
        "tolerance": PARAMETER_TOLERANCE

    }


# ================================================================
# 14. DET-01 — FEDPROX MU=0 VERSUS FEDAVG
# ================================================================

def run_det01(configuration):

    fedavg_result = run_federated_experiment(

        configuration=configuration,
        seed=SEED,
        algorithm="FedAvg",
        rounds=SANITY_ROUNDS,
        local_epochs=SANITY_LOCAL_EPOCHS,
        mu=0.0

    )

    fedprox_zero_result = run_federated_experiment(

        configuration=configuration,
        seed=SEED,
        algorithm="FedProx",
        rounds=SANITY_ROUNDS,
        local_epochs=SANITY_LOCAL_EPOCHS,
        mu=0.0

    )

    passed, maximum_difference = compare_state_dicts(
        fedavg_result["final_state"],
        fedprox_zero_result["final_state"]
    )

    return {

        "test_id": "DET-01",
        "configuration": configuration,
        "test": "FedProx mu=0 versus FedAvg",
        "passed": passed,
        "maximum_parameter_difference": maximum_difference,
        "tolerance": PARAMETER_TOLERANCE

    }


# ================================================================
# 15. DET-02 — REPEATED SAME-SEED EXPERIMENT
# ================================================================

def run_det02(configuration):

    first_result = run_federated_experiment(

        configuration=configuration,
        seed=SEED,
        algorithm="FedAvg",
        rounds=SANITY_ROUNDS,
        local_epochs=SANITY_LOCAL_EPOCHS,
        mu=0.0

    )

    second_result = run_federated_experiment(

        configuration=configuration,
        seed=SEED,
        algorithm="FedAvg",
        rounds=SANITY_ROUNDS,
        local_epochs=SANITY_LOCAL_EPOCHS,
        mu=0.0

    )

    parameters_passed, maximum_parameter_difference = (
        compare_state_dicts(
            first_result["final_state"],
            second_result["final_state"]
        )
    )

    first_evaluation = pd.DataFrame(
        first_result["evaluation_records"]
    )

    second_evaluation = pd.DataFrame(
        second_result["evaluation_records"]
    )

    metric_columns = [
        "accuracy",
        "precision",
        "sensitivity_recall",
        "specificity",
        "f1",
        "macro_f1",
        "balanced_accuracy",
        "auroc",
        "auprc",
        "brier_score",
        "observed_positive_prevalence",
        "predicted_positive_prevalence",
        "calibration_ece_10_bins"
    ]

    first_values = first_evaluation[
        metric_columns
    ].to_numpy(dtype=float)

    second_values = second_evaluation[
        metric_columns
    ].to_numpy(dtype=float)

    metrics_passed = np.allclose(
        first_values,
        second_values,
        atol=PARAMETER_TOLERANCE,
        rtol=0,
        equal_nan=True
    )

    passed = (
        parameters_passed
        and metrics_passed
    )

    return {

        "test_id": "DET-02",
        "configuration": configuration,
        "test": "Repeated same-seed experiment",
        "passed": passed,
        "maximum_parameter_difference": (
            maximum_parameter_difference
        ),
        "metrics_identical_within_tolerance": (
            bool(metrics_passed)
        ),
        "tolerance": PARAMETER_TOLERANCE

    }


# ================================================================
# 16. ADDITIONAL FEDPROX MU=0.01 SMOKE CHECK
# ================================================================

def run_fedprox_smoke_test(configuration):

    result = run_federated_experiment(

        configuration=configuration,
        seed=SEED,
        algorithm="FedProx",
        rounds=SANITY_ROUNDS,
        local_epochs=SANITY_LOCAL_EPOCHS,
        mu=FEDPROX_MU

    )

    final_parameters = result["final_state"]

    finite_parameters = all(
        torch.isfinite(parameter).all().item()
        for parameter in final_parameters.values()
    )

    return {

        "configuration": configuration,
        "test": "FedProx mu=0.01 smoke check",
        "passed": bool(finite_parameters),
        "mu": FEDPROX_MU,
        "rounds": SANITY_ROUNDS,
        "local_epochs": SANITY_LOCAL_EPOCHS

    }


# ================================================================
# 17. RUN IMPLEMENTATION TESTS
# ================================================================

all_test_results = []
all_smoke_results = []

for configuration in CONFIGURATIONS:

    print("\n" + "=" * 75)
    print("CANONICAL IMPLEMENTATION VALIDATION:", configuration.upper())
    print("=" * 75)

    print("\nRunning DET-03...")
    det03 = run_det03(configuration)
    all_test_results.append(det03)
    print(det03)

    print("\nRunning DET-01...")
    det01 = run_det01(configuration)
    all_test_results.append(det01)
    print(det01)

    print("\nRunning DET-02...")
    det02 = run_det02(configuration)
    all_test_results.append(det02)
    print(det02)

    print("\nRunning FedProx smoke check...")
    smoke = run_fedprox_smoke_test(configuration)
    all_smoke_results.append(smoke)
    print(smoke)


# ================================================================
# 18. TEST REPORTS
# ================================================================

test_df = pd.DataFrame(
    all_test_results
)

smoke_df = pd.DataFrame(
    all_smoke_results
)

test_file = OUTPUT_DIR / "sanity_test_report.csv"

smoke_file = OUTPUT_DIR / "fedprox_smoke_test_report.csv"

test_df.to_csv(
    test_file,
    index=False
)

smoke_df.to_csv(
    smoke_file,
    index=False
)


# ================================================================
# 19. OVERALL IMPLEMENTATION STATUS
# ================================================================

all_sanity_tests_passed = bool(
    test_df["passed"].all()
)

all_smoke_tests_passed = bool(
    smoke_df["passed"].all()
)

implementation_tests_passed = (
    all_sanity_tests_passed
    and all_smoke_tests_passed
)

print("\n" + "=" * 75)
print("IMPLEMENTATION VALIDATION SUMMARY")
print("=" * 75)

print(
    test_df[
        [
            "test_id",
            "configuration",
            "passed"
        ]
    ].to_string(index=False)
)

print("\nFedProx smoke checks:")
print(
    smoke_df[
        [
            "configuration",
            "passed"
        ]
    ].to_string(index=False)
)

print("\nAll sanity tests passed:", all_sanity_tests_passed)

print("All FedProx smoke checks passed:", all_smoke_tests_passed)

print("Implementation tests passed:", implementation_tests_passed)


# ================================================================
# 20. SAVE IMPLEMENTATION MANIFEST
# ================================================================

manifest = {

    "implementation_name": (
        "Canonical Federated Learning Implementation"
    ),

    "version": VERSION,

    "purpose": (
        "Canonical implementation and sanity testing only"
    ),

    "algorithms": [
        "FedAvg",
        "FedProx"
    ],

    "feature_configurations": list(
        CONFIGURATIONS.keys()
    ),

    "model_architecture": (
        "input -> 16 -> 8 -> 1"
    ),

    "hidden_activation": "ReLU",

    "output": "Single logit",

    "loss": "BCEWithLogitsLoss",

    "optimizer": "Adam",

    "optimizer_lifecycle": (
        "New Adam optimizer for every client in every "
        "communication round; optimizer moments reset "
        "between client-rounds."
    ),

    "learning_rate": LEARNING_RATE,

    "batch_size": BATCH_SIZE,

    "fedprox_mu": FEDPROX_MU,

    "client_participation": (
        "All six geopolitical-zone clients every round"
    ),

    "client_order": (
        "Fixed region ID order 1 through 6"
    ),

    "aggregation": (
        "Sample-size-weighted model parameter aggregation"
    ),

    "client_seed_formula": (
        "experiment_seed + round_number * 1000 + region_id"
    ),

    "dataloader": {
        "shuffle": True,
        "explicit_generator": True,
        "num_workers": 0
    },

    "prediction_threshold": PREDICTION_THRESHOLD,

    "calibration_method": (
        "Expected calibration error with 10 equal-width bins"
    ),

    "sanity_test_rounds": SANITY_ROUNDS,

    "sanity_test_local_epochs": SANITY_LOCAL_EPOCHS,

    "parameter_tolerance": PARAMETER_TOLERANCE,

    "test_results": all_test_results,

    "smoke_test_results": all_smoke_results,

    "all_sanity_tests_passed": all_sanity_tests_passed,

    "all_smoke_tests_passed": all_smoke_tests_passed,

    "implementation_tests_passed": implementation_tests_passed,

    "test_partition_accessed": False,

    "confirmatory_experiment": False,

    "environment": get_environment(),

    "created_utc": datetime.now(
        timezone.utc
    ).isoformat()

}

manifest_file = (
    OUTPUT_DIR / "canonical_implementation_manifest.json"
)

with open(
    manifest_file,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        manifest,
        file,
        indent=4
    )


# ================================================================
# 21. FINAL STATUS
# ================================================================

print("\n" + "=" * 75)

if implementation_tests_passed:

    print("IMPLEMENTATION SANITY TESTS PASSED.")

    print(
        "The implementation may proceed to evaluation "
        "validation and independent review."
    )

else:

    print("IMPLEMENTATION SANITY TESTS FAILED.")

    print(
        "Do not approve the implementation for "
        "confirmatory experiments."
    )

print("=" * 75)

print("\nOutput directory:", OUTPUT_DIR)

print("\nSaved files:")

print("- sanity_test_report.csv")
print("- fedprox_smoke_test_report.csv")
print("- canonical_implementation_manifest.json")

print("\nNo confirmatory experiment was launched.")
print("The test partition was not loaded.")
