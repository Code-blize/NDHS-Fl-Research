# NDHS Flower implementation v1.1 — corrected candidate

This versioned candidate preserves v1.0. Do not overwrite the v1.0 source or outputs.

## Corrections and implementation
- Explicit Flower partition ID 0–5 to DHS v024 1–6 mapping.
- Exactly six nodes required; strategy checks one successful training reply per region every round.
- Flower FedAvg's built-in configure_train injects the current server round into the client ConfigRecord.
- Sample-size-weighted model and training-metric aggregation uses `num-examples`.
- FedProx adds `(mu/2) * sum((local_parameter - received_global_parameter)^2)` to BCE; aggregation remains sample-size-weighted FedAvg.
- New Adam optimizer per client-round; deterministic seed derives from base seed, region and server round.
- Server-side pooled and six-region validation metrics; test data never loaded.
- Metrics include confusion counts, accuracy, precision, sensitivity, specificity, F1, macro-F1, balanced accuracy, AUROC, average precision, Brier, binary cross-entropy, prevalence, calibration intercept/slope/status and 10-bin ECE.

## Install into the existing Windows environment
From this directory:
```powershell
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```
Dependencies are already installed; `--no-deps` avoids network package resolution.

## Run unit/contract tests
```powershell
.\.venv\Scripts\python.exe -m pytest .\tests -v
```

## Run six-client local simulation
```powershell
.\.venv\Scripts\python.exe run_simulation.py
```
This calls Flower's `run_simulation` API directly and avoids `flwr run`'s application dependency synchronization step. The run config in `pyproject.toml` is set to 10 rounds, E=1, baseline FedAvg, seed 42.

## Important status
This is a corrected implementation candidate, not a passed verification report. The local tests do not prove Ray can run six clients on this Windows machine. The simulation smoke test remains mandatory. Also, the one-to-one partition-id mapping assumption must be confirmed from runtime replies; the strategy enforces the expected six-region set and will stop rather than silently accept a wrong mapping.

## Pilot sequencing
Only after unit tests and the six-client smoke test pass:
- Baseline FedAvg E=1, 10 rounds, seed 42
- Baseline FedProx E=1, 10 rounds, seed 42
- Expanded FedAvg E=1, 10 rounds, seed 42
- Expanded FedProx E=1, 10 rounds, seed 42
Each run must use a unique output directory. E=5/E=10 assessment follows the first E=1 runtime assessment.
