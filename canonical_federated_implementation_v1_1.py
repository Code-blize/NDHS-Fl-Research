"""
NDHS Federated Learning — Canonical Implementation v1.1
========================================================
Versioned integration of the approved FL training implementation with the
independently tested validation evaluator v1.1.

Preserved design:
- CPU execution; deterministic PyTorch algorithms.
- MLP: input -> 16 -> 8 -> 1, ReLU, one logit.
- BCEWithLogitsLoss; Adam, learning rate 0.001; batch size 32.
- Six clients in fixed region-ID order 1..6; all participate.
- Sample-size-weighted parameter aggregation.
- FedProx penalty: (mu/2) * sum((local_parameter-global_parameter)^2).
- New Adam optimizer for each client in each communication round.
- Explicit DataLoader generator; num_workers=0.
- Client seed: experiment_seed + round_number*1000 + region_id.
- FedProx primary mu = 0.01; FedAvg mu = 0.
- Validation-only evaluation; test partition is not loaded.

Important:
- This file is deliberately safe to import: it launches no experiment on import.
- Run this file directly to execute the small integration smoke run.
- The smoke run is NOT a confirmatory experiment and does not authorize
  confirmatory analysis.
- Keep this file beside evaluate_model_v1_1.py.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import platform
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sklearn
from torch.utils.data import DataLoader, TensorDataset

from evaluate_model_v1_1 import evaluate_model_v1_1


# ---------------------------------------------------------------------
# 1. Project paths and frozen configuration
# ---------------------------------------------------------------------

ROOT = Path(r"C:\Users\usher\Documents\NDHS-Fl-Research\NDHS_FL_Research")
DATA_DIR = ROOT / "05_preprocessed"
OUTPUT_ROOT = ROOT / "09_canonical_implementation_v1_1"

VERSION = "1.1"
DEVICE = torch.device("cpu")

CONFIGURATIONS = {"baseline": 15, "expanded": 51}
REGIONS = {
    1: "North Central",
    2: "North East",
    3: "North West",
    4: "South East",
    5: "South South",
    6: "South West",
}

SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 0.001
FEDPROX_MU = 0.01
PREDICTION_THRESHOLD = 0.5
CALIBRATION_BINS = 10
PARAMETER_TOLERANCE = 1e-7

# Integration-smoke settings only. These are not confirmatory settings.
SMOKE_ROUNDS = 1
SMOKE_LOCAL_EPOCHS = 1
SMOKE_ALGORITHMS = ("FedAvg", "FedProx")


# ---------------------------------------------------------------------
# 2. Reproducibility and environment
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def get_environment() -> dict:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "device": str(DEVICE),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------
# 3. Data loading — train and validation only; preserve source IDs
# ---------------------------------------------------------------------

def load_partition(configuration: str, partition: str):
    if configuration not in CONFIGURATIONS:
        raise ValueError(f"Unknown feature configuration: {configuration}")
    if partition not in ("train", "validation"):
        raise ValueError(
            "Only train and validation partitions are permitted; "
            "the test partition is deliberately inaccessible."
        )

    path = DATA_DIR / f"{configuration}_{partition}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Required partition file not found: {path}")

    with np.load(path, allow_pickle=True) as data:
        required = {"X", "y", "v024", "source_row_id"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(
                f"{path.name} is missing required arrays: {sorted(missing)}"
            )

        X = np.asarray(data["X"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.float32).reshape(-1)
        regions = np.asarray(data["v024"], dtype=np.int64).reshape(-1)
        source_ids = np.asarray(data["source_row_id"]).reshape(-1).astype(str)

    n = len(y)
    if X.ndim != 2 or not (len(X) == len(regions) == len(source_ids) == n):
        raise ValueError(f"Inconsistent array dimensions in {path.name}.")
    if n == 0:
        raise ValueError(f"Empty partition: {path.name}")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError(f"Non-finite values found in {path.name}.")
    if not set(np.unique(y)).issubset({0.0, 1.0}):
        raise ValueError(f"Outcome must be binary in {path.name}.")
    if not set(np.unique(regions)).issubset(set(REGIONS)):
        raise ValueError(f"Unexpected region ID in {path.name}.")
    if len(np.unique(source_ids)) != n:
        raise ValueError(f"Duplicate source_row_id values in {path.name}.")
    if X.shape[1] != CONFIGURATIONS[configuration]:
        raise ValueError(
            f"{configuration} expected {CONFIGURATIONS[configuration]} "
            f"features, found {X.shape[1]}."
        )

    return X, y, regions, source_ids


def validate_training_data(X, y, regions, expected_features):
    if X.shape[1] != expected_features:
        raise ValueError("Training feature dimension mismatch.")
    if set(np.unique(regions)) != set(REGIONS):
        raise ValueError("Training partition must contain all six clients.")
    if set(np.unique(y)) != {0.0, 1.0}:
        raise ValueError("Training partition must contain both outcome classes.")

    for region_id, region_name in REGIONS.items():
        mask = regions == region_id
        if not mask.any():
            raise ValueError(f"Training client is empty: {region_name}.")
        if set(np.unique(y[mask])) != {0.0, 1.0}:
            raise ValueError(
                f"Training client {region_name} does not contain both classes."
            )


# ---------------------------------------------------------------------
# 4. Canonical MLP and parameter comparisons
# ---------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


def create_model(input_dim: int, seed: int):
    set_seed(seed)
    return MLP(input_dim).to(DEVICE)


def compare_state_dicts(state_a, state_b):
    if state_a.keys() != state_b.keys():
        return False, float("inf")
    differences = []
    for key in state_a:
        if state_a[key].shape != state_b[key].shape:
            return False, float("inf")
        differences.append(
            torch.max(torch.abs(state_a[key] - state_b[key])).item()
        )
    maximum_difference = max(differences) if differences else 0.0
    return maximum_difference <= PARAMETER_TOLERANCE, maximum_difference


# ---------------------------------------------------------------------
# 5. Client training — retained approved optimizer lifecycle
# ---------------------------------------------------------------------

def train_client(
    global_model,
    X_client,
    y_client,
    mu: float,
    client_seed: int,
    local_epochs: int,
):
    if mu < 0:
        raise ValueError("FedProx mu cannot be negative.")
    if local_epochs < 1:
        raise ValueError("local_epochs must be at least 1.")
    if len(y_client) == 0:
        raise ValueError("Cannot train an empty client.")

    set_seed(client_seed)

    X_tensor = torch.tensor(X_client, dtype=torch.float32)
    y_tensor = torch.tensor(y_client, dtype=torch.float32)

    generator = torch.Generator()
    generator.manual_seed(client_seed)

    loader = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    local_model = MLP(X_client.shape[1]).to(DEVICE)
    local_model.load_state_dict(global_model.state_dict())

    global_parameters = [
        parameter.detach().clone()
        for parameter in global_model.parameters()
    ]

    # Confirmed PH4-DEC-01: reset Adam moments for every client-round.
    optimizer = torch.optim.Adam(
        local_model.parameters(),
        lr=LEARNING_RATE,
    )
    criterion = nn.BCEWithLogitsLoss()
    local_model.train()

    for _epoch in range(local_epochs):
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            optimizer.zero_grad()
            logits = local_model(batch_X)
            loss = criterion(logits, batch_y)

            if mu > 0:
                proximal_penalty = torch.zeros((), device=DEVICE)
                for local_parameter, global_parameter in zip(
                    local_model.parameters(), global_parameters
                ):
                    proximal_penalty += torch.sum(
                        (local_parameter - global_parameter) ** 2
                    )
                loss = loss + (mu / 2.0) * proximal_penalty

            loss.backward()
            optimizer.step()

    return copy.deepcopy(local_model.state_dict())


# ---------------------------------------------------------------------
# 6. Sample-size-weighted FedAvg/FedProx aggregation
# ---------------------------------------------------------------------

def aggregate_models(client_states, client_sizes):
    if len(client_states) != 6 or len(client_sizes) != 6:
        raise ValueError("Exactly six client updates and sizes are required.")
    if any(int(size) <= 0 for size in client_sizes):
        raise ValueError("Every client must have a positive sample size.")

    total_samples = int(sum(client_sizes))
    if total_samples <= 0:
        raise ValueError("Invalid total sample size.")

    reference_keys = client_states[0].keys()
    for state in client_states:
        if state.keys() != reference_keys:
            raise ValueError("Client parameter keys do not match.")

    global_state = {}
    for key in reference_keys:
        reference_tensor = client_states[0][key]
        aggregated_tensor = torch.zeros_like(reference_tensor)

        for state, size in zip(client_states, client_sizes):
            if state[key].shape != reference_tensor.shape:
                raise ValueError(f"Parameter shape mismatch: {key}")
            if not torch.isfinite(state[key]).all():
                raise ValueError(f"Non-finite client parameter: {key}")
            aggregated_tensor += state[key] * (int(size) / total_samples)

        if not torch.isfinite(aggregated_tensor).all():
            raise ValueError(f"Non-finite aggregated parameter: {key}")
        global_state[key] = aggregated_tensor

    return global_state


# ---------------------------------------------------------------------
# 7. Canonical federated experiment with v1.1 evaluator integration
# ---------------------------------------------------------------------

def run_federated_experiment(
    configuration: str,
    seed: int,
    algorithm: str,
    rounds: int,
    local_epochs: int,
    mu: float | None = None,
):
    if algorithm not in ("FedAvg", "FedProx"):
        raise ValueError(f"Unknown algorithm: {algorithm}")
    if rounds < 1 or local_epochs < 1:
        raise ValueError("rounds and local_epochs must both be >= 1.")

    if algorithm == "FedAvg":
        if mu not in (None, 0, 0.0):
            raise ValueError("FedAvg must use mu=0.")
        effective_mu = 0.0
    else:
        effective_mu = FEDPROX_MU if mu is None else float(mu)
        if effective_mu < 0:
            raise ValueError("FedProx mu cannot be negative.")

    X_train, y_train, train_regions, _train_ids = load_partition(
        configuration, "train"
    )
    X_val, y_val, val_regions, val_ids = load_partition(
        configuration, "validation"
    )

    expected_features = CONFIGURATIONS[configuration]
    validate_training_data(
        X_train, y_train, train_regions, expected_features
    )
    if X_val.shape[1] != expected_features:
        raise ValueError("Validation feature dimension mismatch.")
    if set(np.unique(val_regions)) != set(REGIONS):
        raise ValueError("Validation partition must contain all six regions.")

    global_model = create_model(X_train.shape[1], seed)
    initial_state = copy.deepcopy(global_model.state_dict())

    round_records = []
    metric_records = []
    confusion_records = []
    probability_frames = []
    participation_records = []

    for round_number in range(1, rounds + 1):
        client_states = []
        client_sizes = []

        # Fixed client order: region IDs 1 through 6.
        for region_id, region_name in REGIONS.items():
            mask = train_regions == region_id
            X_client = X_train[mask]
            y_client = y_train[mask]
            client_seed = seed + round_number * 1000 + region_id

            client_state = train_client(
                global_model=global_model,
                X_client=X_client,
                y_client=y_client,
                mu=effective_mu,
                client_seed=client_seed,
                local_epochs=local_epochs,
            )
            client_states.append(client_state)
            client_sizes.append(len(y_client))
            participation_records.append({
                "configuration": configuration,
                "algorithm": algorithm,
                "seed": seed,
                "round": round_number,
                "region_id": region_id,
                "region": region_name,
                "training_n": len(y_client),
                "client_seed": client_seed,
            })

        global_model.load_state_dict(
            aggregate_models(client_states, client_sizes)
        )

        # Evaluator is explicitly validation-only and rejects other labels.
        metrics_df, confusion_df, probabilities = evaluate_model_v1_1(
            global_model,
            X_val,
            y_val,
            val_regions,
            feature_configuration=configuration,
            algorithm=algorithm,
            seed=seed,
            communication_round=round_number,
            partition="validation",
            threshold=PREDICTION_THRESHOLD,
            ece_bins=CALIBRATION_BINS,
            region_names=REGIONS,
            device=DEVICE,
        )

        metrics_df = metrics_df.copy()
        confusion_df = confusion_df.copy()
        metrics_df["mu"] = effective_mu
        confusion_df["mu"] = effective_mu
        metric_records.extend(metrics_df.to_dict(orient="records"))
        confusion_records.extend(confusion_df.to_dict(orient="records"))

        if len(probabilities) != len(val_ids):
            raise RuntimeError("Evaluator probability count does not match validation IDs.")

        probability_frames.append(pd.DataFrame({
            "source_row_id": val_ids,
            "region_id": val_regions,
            "observed_outcome": y_val.astype(int),
            "predicted_probability": np.asarray(probabilities, dtype=float),
            "configuration": configuration,
            "algorithm": algorithm,
            "seed": seed,
            "round": round_number,
            "partition": "validation",
        }))

        pooled_rows = metrics_df.loc[metrics_df["region_id"] == 0]
        if len(pooled_rows) != 1:
            raise RuntimeError("Evaluator must return exactly one pooled metrics row.")
        pooled = pooled_rows.iloc[0]

        round_records.append({
            "configuration": configuration,
            "algorithm": algorithm,
            "seed": seed,
            "round": round_number,
            "mu": effective_mu,
            "local_epochs": local_epochs,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "validation_binary_cross_entropy_loss": pooled.get(
                "binary_cross_entropy_loss", np.nan
            ),
            "validation_accuracy": pooled.get("accuracy", np.nan),
            "validation_macro_f1": pooled.get("macro_f1", np.nan),
            "validation_auroc": pooled.get("auroc", np.nan),
            "validation_auprc": pooled.get("auprc", np.nan),
            "validation_brier_score": pooled.get("brier_score", np.nan),
            "validation_ece": pooled.get("ece", np.nan),
            "validation_calibration_intercept": pooled.get(
                "calibration_intercept", np.nan
            ),
            "validation_calibration_slope": pooled.get(
                "calibration_slope", np.nan
            ),
        })

    return {
        "model": global_model,
        "initial_state": initial_state,
        "final_state": copy.deepcopy(global_model.state_dict()),
        "round_records": round_records,
        "evaluation_records": metric_records,
        "confusion_records": confusion_records,
        "probability_records": probability_frames,
        "participation_records": participation_records,
    }


# ---------------------------------------------------------------------
# 8. Non-overwriting output writer and integration smoke runner
# ---------------------------------------------------------------------

def save_experiment_outputs(result, output_dir: Path, metadata: dict):
    output_dir.mkdir(parents=True, exist_ok=False)

    pd.DataFrame(result["round_records"]).to_csv(
        output_dir / "round_summary.csv", index=False
    )
    pd.DataFrame(result["evaluation_records"]).to_csv(
        output_dir / "validation_metrics_pooled_and_regional.csv", index=False
    )
    pd.DataFrame(result["confusion_records"]).to_csv(
        output_dir / "validation_confusion_counts.csv", index=False
    )
    pd.concat(result["probability_records"], ignore_index=True).to_csv(
        output_dir / "validation_probabilities_with_source_ids.csv", index=False
    )
    pd.DataFrame(result["participation_records"]).to_csv(
        output_dir / "client_participation.csv", index=False
    )

    torch.save(result["initial_state"], output_dir / "initial_model_state.pt")
    torch.save(result["final_state"], output_dir / "final_model_state.pt")

    manifest = {
        "implementation": "Canonical Federated Learning Implementation",
        "version": VERSION,
        "run_type": "integration_smoke_only",
        "confirmatory_experiment": False,
        "test_partition_accessed": False,
        "validation_partition_used": True,
        "metadata": metadata,
        "model_architecture": "input -> 16 -> 8 -> 1",
        "activation": "ReLU",
        "output": "single logit",
        "training_loss": "BCEWithLogitsLoss",
        "optimizer": "Adam",
        "optimizer_lifecycle": (
            "New Adam optimizer for each client in each communication round; "
            "optimizer moments reset between client-rounds."
        ),
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "fedprox_mu": FEDPROX_MU,
        "client_participation": "All six clients every round",
        "client_order": "Fixed region ID order 1 through 6",
        "aggregation": "Sample-size-weighted parameter aggregation",
        "client_seed_formula": "seed + round_number*1000 + region_id",
        "dataloader": {
            "shuffle": True,
            "explicit_generator": True,
            "num_workers": 0,
        },
        "evaluator": {
            "version": "1.1",
            "threshold": PREDICTION_THRESHOLD,
            "ece_bins": CALIBRATION_BINS,
            "undefined_metrics": "NaN with explicit status/reason",
        },
        "source_hashes": {
            "canonical_implementation": sha256_file(Path(__file__).resolve()),
            "evaluation_module": sha256_file(
                Path(__file__).with_name("evaluate_model_v1_1.py")
            ),
        },
        "environment": get_environment(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, default=str)


def run_integration_smoke():
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Preprocessed data directory not found: {DATA_DIR}")
    if not Path(__file__).with_name("evaluate_model_v1_1.py").is_file():
        raise FileNotFoundError(
            "Place evaluate_model_v1_1.py beside this implementation script."
        )

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = OUTPUT_ROOT / f"integration_smoke_{run_stamp}"
    run_root.mkdir(parents=True, exist_ok=False)

    summaries = []
    for configuration in CONFIGURATIONS:
        for algorithm in SMOKE_ALGORITHMS:
            effective_mu = 0.0 if algorithm == "FedAvg" else FEDPROX_MU
            print(
                f"Running smoke: {configuration} | {algorithm} | "
                f"rounds={SMOKE_ROUNDS} | local_epochs={SMOKE_LOCAL_EPOCHS}"
            )

            result = run_federated_experiment(
                configuration=configuration,
                seed=SEED,
                algorithm=algorithm,
                rounds=SMOKE_ROUNDS,
                local_epochs=SMOKE_LOCAL_EPOCHS,
                mu=effective_mu,
            )

            experiment_dir = run_root / f"{configuration}_{algorithm.lower()}"
            metadata = {
                "configuration": configuration,
                "input_features": CONFIGURATIONS[configuration],
                "algorithm": algorithm,
                "seed": SEED,
                "rounds": SMOKE_ROUNDS,
                "local_epochs": SMOKE_LOCAL_EPOCHS,
                "mu": effective_mu,
                "training_n": len(load_partition(configuration, "train")[1]),
                "validation_n": len(load_partition(configuration, "validation")[1]),
                "validation_ids_saved": True,
            }
            save_experiment_outputs(result, experiment_dir, metadata)

            final_round = result["round_records"][-1]
            summaries.append({
                **metadata,
                "final_validation_accuracy": final_round[
                    "validation_accuracy"
                ],
                "final_validation_macro_f1": final_round[
                    "validation_macro_f1"
                ],
                "outputs_saved": True,
            })

    pd.DataFrame(summaries).to_csv(
        run_root / "integration_smoke_summary.csv", index=False
    )
    print("\nIntegration smoke run completed.")
    print(f"Outputs: {run_root}")
    print("This is an integration smoke only, not a confirmatory experiment.")
    return run_root


if __name__ == "__main__":
    run_integration_smoke()
