from pathlib import Path
import csv
import hashlib
import math
import json
import platform
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
import torch
import flwr

from flwr.serverapp import ServerApp
from flwr.common import ArrayRecord, ConfigRecord, MetricRecord
from .model import NDHSMLP
from .evaluation import evaluate_pooled_and_regions
from .strategy import SixRegionFedAvg
from .reproducibility import set_global_seed

app = ServerApp()

def _sha256_arrays(arrays):
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()

def _write_json(path, payload):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    temp.replace(path)

def _resolve_proximal_mu(algorithm, configured_mu):
    """Enforce the prespecified FedProx setting and disable the penalty for FedAvg."""
    if algorithm == "fedavg":
        return 0.0

    mu = float(configured_mu)
    if not math.isclose(mu, 0.01, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("FedProx requires the prespecified proximal-mu = 0.01")
    return mu

@app.main()
def main(grid, context):
    cfg = context.run_config
    required = {
        "num-server-rounds", "local-epochs", "batch-size", "learning-rate",
        "proximal-mu", "algorithm", "feature-config", "seed",
        "train-data", "validation-data", "output-dir",
    }
    missing = required.difference(cfg.keys())
    if missing:
        raise KeyError(f"Flower run configuration missing keys: {sorted(missing)}")

    feature = str(cfg["feature-config"]).lower()
    dimensions = {"baseline": 15, "expanded": 51}
    if feature not in dimensions:
        raise ValueError(f"Unsupported feature-config: {feature}")
    input_dim = dimensions[feature]
    algorithm = str(cfg["algorithm"]).lower()
    if algorithm not in {"fedavg", "fedprox"}:
        raise ValueError("algorithm must be fedavg or fedprox")
    mu = _resolve_proximal_mu(algorithm, cfg["proximal-mu"])
    if int(cfg["num-server-rounds"]) < 1 or int(cfg["local-epochs"]) < 1:
        raise ValueError("num-server-rounds and local-epochs must be positive")
    if int(cfg["batch-size"]) < 1 or float(cfg["learning-rate"]) <= 0:
        raise ValueError("batch-size and learning-rate must be positive")

    train_path = Path(str(cfg["train-data"]))
    validation_path = Path(str(cfg["validation-data"]))
    if not train_path.is_file():
        raise FileNotFoundError(f"Training archive not found: {train_path}")
    if not validation_path.is_file():
        raise FileNotFoundError(f"Validation archive not found: {validation_path}")

    output_dir = Path(str(cfg["output-dir"]))
    # Never overwrite a prior run, even if it is incomplete.
    output_dir.mkdir(parents=True, exist_ok=False)

    seed = int(cfg["seed"])
    set_global_seed(seed)
    model = NDHSMLP(input_dim)
    initial_arrays = ArrayRecord(model.state_dict())
    initial_hash = _sha256_arrays(initial_arrays.to_numpy_ndarrays())

    train_config = ConfigRecord({
        "train-data": str(train_path.resolve()),
        "local-epochs": int(cfg["local-epochs"]),
        "batch-size": int(cfg["batch-size"]),
        "learning-rate": float(cfg["learning-rate"]),
        "proximal-mu": mu,
        "seed": seed,
    })

    strategy = SixRegionFedAvg(
        fraction_train=1.0,
        fraction_evaluate=0.0,
        min_train_nodes=6,
        min_available_nodes=6,
        weighted_by_key="num-examples",
    )
    validation_rows = []

    def server_evaluate(server_round, arrays):
        rows = evaluate_pooled_and_regions(arrays.to_numpy_ndarrays(), validation_path, input_dim)
        for row in rows:
            row.update({
                "server_round": int(server_round),
                "algorithm": algorithm,
                "feature_config": feature,
                "local_epochs": int(cfg["local-epochs"]),
                "seed": seed,
            })
        validation_rows.extend(rows)
        pooled = rows[0]
        return MetricRecord({
            "validation-accuracy": float(pooled["accuracy"]),
            "validation-macro-f1": float(pooled["macro_f1"]),
            "validation-auroc": float(pooled["auroc"]),
        })

    status = "failed"
    failure_reason = None
    result = None
    try:
        result = strategy.start(
            grid=grid,
            initial_arrays=initial_arrays,
            train_config=train_config,
            num_rounds=int(cfg["num-server-rounds"]),
            evaluate_fn=server_evaluate,
        )
        if len(strategy.round_audit) != int(cfg["num-server-rounds"]):
            raise RuntimeError(
                f"Expected {cfg['num-server-rounds']} completed rounds, got {len(strategy.round_audit)}"
            )
        if result.arrays is None:
            raise RuntimeError("Flower returned no final model arrays")
        final_hash = _sha256_arrays(result.arrays.to_numpy_ndarrays())
        torch.save(result.arrays.to_torch_state_dict(), output_dir / "final_model_state_dict.pt")
        if validation_rows:
            with (output_dir / "validation_metrics_pooled_and_regional.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(validation_rows[0].keys()))
                writer.writeheader()
                writer.writerows(validation_rows)
        status = "completed"
    except Exception as exc:
        failure_reason = f"{type(exc).__name__}: {exc}"
        final_hash = None
        raise
    finally:
        manifest = {
            "implementation_version": "1.2.1",
            "status": status,
            "failure_reason": failure_reason,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "software": {
                "python": sys.version,
                "platform": platform.platform(),
                "flower": flwr.__version__,
                "torch": torch.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "configuration": {
                "num_server_rounds": int(cfg["num-server-rounds"]),
                "local_epochs": int(cfg["local-epochs"]),
                "batch_size": int(cfg["batch-size"]),
                "learning_rate": float(cfg["learning-rate"]),
                "algorithm": algorithm,
                "proximal_mu": mu,
                "feature_config": feature,
                "input_dim": input_dim,
                "seed": seed,
                "train_data": str(train_path.resolve()),
                "validation_data": str(validation_path.resolve()),
                "output_dir": str(output_dir.resolve()),
                "weighted_by": "num-examples",
                "fraction_train": 1.0,
                "expected_regions": [1, 2, 3, 4, 5, 6],
            },
            "initial_parameters_sha256": initial_hash,
            "final_parameters_sha256": final_hash,
            "round_audit": strategy.round_audit,
            "validation_rows": len(validation_rows),
        }
        _write_json(output_dir / "run_manifest.json", manifest)
