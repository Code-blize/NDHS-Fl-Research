from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

import canonical_federated_implementation_v1_1 as canonical


SEED = 42
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
THRESHOLD = 0.5
CONFIGURATIONS = ("baseline", "expanded")

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "phase7_1_baseline_benchmarks_seed42"


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def metric_row(y_true, probabilities, *, configuration, model_name,
               seed, scope, region_id=None):
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "configuration": configuration,
        "model": model_name,
        "seed": seed,
        "scope": scope,
        "region_id": region_id,
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, predictions),
        "macro_f1": f1_score(
            y_true, predictions, average="macro", zero_division=0
        ),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": THRESHOLD,
    }


def evaluate_all_regions(model, X, y, regions, *, configuration,
                         model_name, seed, rows):
    probabilities = model.predict_proba(X)[:, 1]

    rows.append(metric_row(
        y, probabilities,
        configuration=configuration,
        model_name=model_name,
        seed=seed,
        scope="pooled",
    ))

    for region_id in canonical.REGIONS:
        mask = regions == region_id
        rows.append(metric_row(
            y[mask], probabilities[mask],
            configuration=configuration,
            model_name=model_name,
            seed=seed,
            scope="region",
            region_id=region_id,
        ))


def train_mlp(X, y, *, seed, epochs=EPOCHS):
    set_all_seeds(seed)

    model = canonical.MLP(X.shape[1]).to(canonical.DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )
    criterion = nn.BCEWithLogitsLoss()

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    model.train()
    for _ in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def mlp_predict_proba(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        return torch.sigmoid(logits).cpu().numpy()


def evaluate_mlp(model, X, y, regions, *, configuration,
                 model_name, seed, rows):
    probabilities = mlp_predict_proba(model, X)

    rows.append(metric_row(
        y, probabilities,
        configuration=configuration,
        model_name=model_name,
        seed=seed,
        scope="pooled",
    ))

    for region_id in canonical.REGIONS:
        mask = regions == region_id
        rows.append(metric_row(
            y[mask], probabilities[mask],
            configuration=configuration,
            model_name=model_name,
            seed=seed,
            scope="region",
            region_id=region_id,
        ))


def main():
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {OUTPUT_DIR}"
        )
    OUTPUT_DIR.mkdir(parents=True)

    all_rows = []

    for configuration in CONFIGURATIONS:
        X_train, y_train, train_regions, train_ids = canonical.load_partition(
            configuration, "train"
        )
        X_val, y_val, val_regions, val_ids = canonical.load_partition(
            configuration, "validation"
        )

        canonical.validate_training_data(
            X_train, y_train, train_regions,
            canonical.CONFIGURATIONS[configuration],
        )
        if set(np.unique(val_regions)) != set(canonical.REGIONS):
            raise ValueError(
                f"{configuration} validation data must contain all regions."
            )

        # Centralized logistic regression
        logistic = LogisticRegression(
            max_iter=2000,
            random_state=SEED,
            solver="lbfgs",
        )
        logistic.fit(X_train, y_train.astype(int))
        evaluate_all_regions(
            logistic, X_val, y_val, val_regions,
            configuration=configuration,
            model_name="centralized_logistic_regression",
            seed=SEED,
            rows=all_rows,
        )

        # Centralized MLP
        central_mlp = train_mlp(X_train, y_train, seed=SEED)
        evaluate_mlp(
            central_mlp, X_val, y_val, val_regions,
            configuration=configuration,
            model_name="centralized_mlp",
            seed=SEED,
            rows=all_rows,
        )

        # Six independent local-only MLPs, one per region.
        for region_id in canonical.REGIONS:
            train_mask = train_regions == region_id
            val_mask = val_regions == region_id

            local_model = train_mlp(
                X_train[train_mask],
                y_train[train_mask],
                seed=SEED + region_id,
            )
            local_probabilities = mlp_predict_proba(
                local_model, X_val[val_mask]
            )
            all_rows.append(metric_row(
                y_val[val_mask],
                local_probabilities,
                configuration=configuration,
                model_name="local_only_mlp",
                seed=SEED,
                scope="region",
                region_id=region_id,
            ))

        print(f"Completed {configuration}")

    results = pd.DataFrame(all_rows)
    results.to_csv(OUTPUT_DIR / "validation_metrics.csv", index=False)

    metadata = {
        "phase": "7.1",
        "seed": SEED,
        "configurations": list(CONFIGURATIONS),
        "centralized_and_local_mlp_epochs": EPOCHS,
        "federated_reference_budget": "10 rounds x 5 local epochs",
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "threshold": THRESHOLD,
        "evaluation_partition": "validation",
        "survey_weights_used": False,
        "local_only_protocol": (
            "One independently trained MLP per region; each evaluated "
            "on that region's validation subset."
        ),
    }
    with (OUTPUT_DIR / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"Metrics rows: {len(results)}")


if __name__ == "__main__":
    main()