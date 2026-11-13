# NDHS Flower Implementation v1.0 — source build

This is a separate Flower branch. It does not overwrite the historical PyTorch branch.

## Encoded design
- Six regional clients, `v024` 1–6 (NC, NE, NW, SE, SS, SW).
- All six required each round; sample-size-weighted FedAvg aggregation.
- MLP input→16→8→1; ReLU; BCEWithLogitsLoss; Adam lr=.001; batch=32.
- Fresh Adam optimizer each client-round; threshold .5 for reported class metrics.
- FedProx adds `(mu/2) * sum((local_parameter - received_global_parameter)^2)` to client BCE; primary μ=.01.
- Server-side pooled validation; test archive is not read.
- Per-client/per-round deterministic seed derivation and explicit DataLoader generator.

## Data
The four NPZ files must be in `05_preprocessed`. Each contains X, y, source_row_id, v024. `source_row_id` is object dtype; loader uses allow_pickle=True only for these trusted local files and never feeds IDs to the model.

Edit data paths in `pyproject.toml` if your Windows root differs.

## Install
```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

## Critical verification gate
This is a source build, not yet a verified Flower run. Before a pilot:
1. Confirm Flower 1.43.0 API imports and message/config behavior.
2. Confirm local simulation maps exactly six nodes to the six regions. The current client expects `context.node_config["partition-id"]` equal to v024 codes 1..6; do not assume this mapping until runtime evidence confirms it.
3. Add/execute tests for hand-calculated weighted aggregation, repeatability, client isolation, and complete metric collection.
4. Integrate calibration intercept/slope and ECE from the already-audited evaluator before calling the metric specification complete.
5. Verify the per-round config (`server-round`) is updated on each training message; adjust with a custom strategy/config hook if needed.

## First target after gates
10 rounds, E=1, seed 42, baseline and expanded, FedAvg and FedProx (μ=.01), validation only. Use distinct output directories for every run. E=5/E=10 only after reviewing the first runtime pilot.
