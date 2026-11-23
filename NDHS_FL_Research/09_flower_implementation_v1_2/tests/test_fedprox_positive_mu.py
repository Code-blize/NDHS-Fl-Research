import copy

import numpy as np
import torch

from ndhs_flower.model import NDHSMLP
from ndhs_flower.training import train_local


def test_fedprox_positive_mu_changes_local_update():
    rng = np.random.default_rng(2026)
    X = rng.normal(size=(128, 4)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.float32)

    torch.manual_seed(2026)
    initial_model = NDHSMLP(input_dim=4)

    fedavg_model = copy.deepcopy(initial_model)
    fedprox_model = copy.deepcopy(initial_model)

    common = {
        "X": X,
        "y": y,
        "epochs": 10,
        "batch_size": 16,
        "lr": 0.001,
        "seed": 2026,
    }

    train_local(
        fedavg_model,
        proximal_mu=0.0,
        **common,
    )

    train_local(
        fedprox_model,
        proximal_mu=0.01,
        **common,
    )

    differences = [
        torch.max(torch.abs(a - b)).item()
        for a, b in zip(
            fedavg_model.parameters(),
            fedprox_model.parameters(),
        )
    ]

    max_difference = max(differences)

    assert max_difference > 1e-12, (
        "FedProx with μ=0.01 produced the same local parameters "
        "as FedAvg with μ=0."
    )

    print(f"Maximum parameter difference: {max_difference:.12g}")