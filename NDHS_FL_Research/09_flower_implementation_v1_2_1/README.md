# NDHS Flower implementation v1.2.0

This is a separate, versioned correction package. It does not overwrite v1.0 or v1.1.

## Purpose and status

v1.2 corrects the application execution/configuration path and strengthens auditability and six-region reply validation. It is a **candidate pending tests in the user's Windows environment**. No simulation result is claimed by this package.

## Fixed research settings

- Six regional clients: v024 codes 1–6, in the established NC, NE, NW, SE, SS, SW order.
- MLP: input → 16 → 8 → 1, ReLU hidden activations, one logit.
- BCEWithLogitsLoss, Adam, learning rate 0.001, batch size 32.
- New Adam optimizer per client per communication round.
- FedAvg and FedProx; FedProx μ=0.01.
- Sample-size-weighted aggregation using `num-examples`.
- Server-side validation; the test partition is not loaded.
- Baseline 15 transformed inputs; expanded 51 transformed inputs.
- Deterministic seed derivation by base seed, region, and round.

## Important v1.2 changes

1. The official execution path is the Flower CLI (`flwr run . local --stream`). The direct `run_simulation.py` runner is intentionally removed.
2. `pyproject.toml` is the authoritative run configuration.
3. The strategy validates six replies, checks error messages and record keys, verifies partition-to-region correspondence, and records participation without using Flower's private `_check_and_log_replies`.
4. Each run creates its output directory exclusively. Existing output directories cause a hard failure; prior results cannot be overwritten.
5. A `run_manifest.json` records software versions, configuration, parameter hashes, completion/failure status and per-round participation.
6. Validation output includes pooled and six regional rows at initialisation (round 0) and after each round.

## Environment and PATH

Use the existing venv. In each new PowerShell session, set the venv Scripts directory on PATH before invoking `flwr`:

```powershell
$env:PATH = "C:\Users\usher\Documents\NDHS-Fl-Research\NDHS_FL_Research\09_flower_implementation_v1\.venv\Scripts;$env:PATH"
```

Do not change system PATH.

## Install and test

From the extracted v1.2 folder:

```powershell
& "C:\Users\usher\Documents\NDHS-Fl-Research\NDHS_FL_Research\09_flower_implementation_v1\.venv\Scripts\python.exe" -m pip install -e . --no-deps
& "C:\Users\usher\Documents\NDHS-Fl-Research\NDHS_FL_Research\09_flower_implementation_v1\.venv\Scripts\python.exe" -m pytest .\tests -v
```

The package deliberately uses one communication round for its first integration smoke test.

## First integration smoke test

Before running, confirm the data files exist at the configured paths and ensure the output directory does not already exist:

`outputs/integration_smoke/baseline_fedavg_e1_s42`

Then, from this v1.2 directory:

```powershell
flwr run . local --stream
```

This runs only one round, one local epoch, baseline features, FedAvg, seed 42. It is an integration smoke test, not a pilot or confirmatory result.

## Switching settings later

Only after the smoke test is audited should the configuration be changed for a computational pilot. For a distinct run, use a new output directory and CLI overrides, for example:

```powershell
flwr run . local --stream --run-config "algorithm='fedprox' proximal-mu=0.01 output-dir='outputs/integration_smoke/baseline_fedprox_e1_s42'"
```

Do not reuse an output directory. Do not run confirmatory experiments until the smoke test, pilot sign-off and configuration freeze are complete.
