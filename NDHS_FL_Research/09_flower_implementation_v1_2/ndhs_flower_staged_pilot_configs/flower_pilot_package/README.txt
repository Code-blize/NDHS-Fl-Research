NDHS FLOWER v1.2 STAGED PILOT CONFIGURATION PACKAGE

Contains configuration overrides only. No source files, datasets, partitions,
existing smoke-test outputs, or prior experiment outputs are modified.
No training has been launched.

STAGE A (8 runs)
baseline/expanded x FedAvg/FedProx x seeds 42 and 43.
10 rounds, 1 local epoch, batch size 32, learning rate 0.001.
FedProx mu=0.01; FedAvg mu=0.0.

STAGE B (8 runs)
baseline/expanded x FedAvg/FedProx x local epochs 5 and 10.
10 rounds, seed 42, batch size 32, learning rate 0.001.
FedProx mu=0.01; FedAvg mu=0.0.

DATA PATHS
Training and validation files are absolute paths under:
C:/Users/usher/Documents/NDHS-Fl-Research/NDHS_FL_Research/05_preprocessed/
No test partition path is included.

RUNNING
From the Flower v1.2 project directory, run one config at a time:
flwr run . --stream --run-config ".\stage_A\<run_name>.toml"
or
flwr run . --stream --run-config ".\stage_B\<run_name>.toml"
Use the filename stem from pilot_run_matrix.csv. Do not batch-launch runs.
Inspect each run manifest and validation output before proceeding.

STAGE C
Not generated. The 25/50-round plan will be decided after reviewing Stage A/B.

PRE-FREEZE SAFEGUARD
The current server_app.py checks only that FedProx mu is positive, not exactly 0.01.
All configs in this package specify 0.01. An exact server-side guard remains recommended
before configuration freeze.

These are proposed pilot configs, not a frozen confirmatory configuration.
