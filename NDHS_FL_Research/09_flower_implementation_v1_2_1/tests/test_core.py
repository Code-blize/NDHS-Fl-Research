import numpy as np
import torch
import pytest

from ndhs_flower.model import NDHSMLP, get_parameters, set_parameters
from ndhs_flower.training import train_local
from ndhs_flower.data_loader import load_partition
from ndhs_flower.reproducibility import flower_partition_to_region
from ndhs_flower.evaluation import _metrics
from ndhs_flower.data_loader import make_loader
from ndhs_flower.reproducibility import set_global_seed

def test_parameter_roundtrip():
    source = NDHSMLP(15)
    target = NDHSMLP(15)
    set_parameters(target, get_parameters(source))
    assert all(torch.equal(a, b) for a, b in zip(source.state_dict().values(), target.state_dict().values()))

def test_mu_zero_repeatability():
    X = np.array([[.1,.2],[.2,.3],[.8,.7],[.9,.8]], dtype=np.float32)
    y = np.array([0,0,1,1], dtype=np.float32)
    initial = NDHSMLP(2).state_dict()
    models = []
    for _ in range(2):
        model = NDHSMLP(2)
        model.load_state_dict(initial)
        train_local(model, X, y, 1, 2, .001, 0.0, 777)
        models.append(model)
    assert all(torch.equal(a, b) for a, b in zip(models[0].state_dict().values(), models[1].state_dict().values()))

def test_partition_mapping():
    assert [flower_partition_to_region(i, 6) for i in range(6)] == [1,2,3,4,5,6]
    with pytest.raises(ValueError):
        flower_partition_to_region(6, 6)
    with pytest.raises(ValueError):
        flower_partition_to_region(0, 5)

def test_partition_loader_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "duplicate.npz"
    np.savez(path, X=np.ones((2,2),np.float32), y=np.array([0,1],np.float32),
             source_row_id=np.array(["x","x"],object), v024=np.array([1,1]))
    with pytest.raises(ValueError, match="Duplicate"):
        load_partition(path, 1)

def test_region_isolation(tmp_path):
    path = tmp_path / "toy.npz"
    np.savez(path, X=np.ones((6,2),np.float32), y=np.array([0,1,0,1,0,1],np.float32),
             source_row_id=np.array(["a","b","c","d","e","f"],object), v024=np.array([1,1,2,2,3,3]))
    X, y, ids = load_partition(path, 2)
    assert len(X) == 2 and set(ids) == {"c","d"}

def test_metrics_and_one_class_status():
    metrics = _metrics(np.array([0,0,1,1]), np.array([.1,.4,.6,.9]))
    assert metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"] == 4
    assert 0 <= metrics["ece_10"] <= 1
    one_class = _metrics(np.array([0,0]), np.array([.1,.2]))
    assert np.isnan(one_class["auroc"])
    assert one_class["calibration_status"] == "undefined_one_class"


def test_fedprox_mu_zero_matches_fedavg_reference():
    X = np.array(
        [
            [.1, .2], [.2, .3], [.3, .4], [.4, .5],
            [.6, .5], [.7, .6], [.8, .7], [.9, .8],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32)

    seed = 2026
    epochs = 2
    batch_size = 2
    lr = 0.001

    # Both algorithms begin with exactly the same parameters.
    initial_model = NDHSMLP(2)
    initial_parameters = {
        key: value.detach().clone()
        for key, value in initial_model.state_dict().items()
    }

    fedprox_mu_zero = NDHSMLP(2)
    fedprox_mu_zero.load_state_dict(initial_parameters)

    fedavg_reference = NDHSMLP(2)
    fedavg_reference.load_state_dict(initial_parameters)

    # Run the implementation with FedProx mu = 0.
    fedprox_loss = train_local(
        fedprox_mu_zero,
        X, y,
        epochs,
        batch_size,
        lr,
        0.0,
        seed,
    )

    # Independent ordinary FedAvg local-training reference:
    # same initialization, data order, loss, optimizer and settings,
    # but without a proximal penalty.
    set_global_seed(seed)
    loader = make_loader(X, y, batch_size, seed)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        fedavg_reference.parameters(),
        lr=lr,
    )

    total_loss = 0.0
    n_total = 0

    fedavg_reference.train()

    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(fedavg_reference(xb), yb)
            loss.backward()
            optimizer.step()

            n = len(yb)
            total_loss += float(loss.detach()) * n
            n_total += n

    fedavg_loss = total_loss / n_total

    # The local updates and reported losses must match.
    assert np.isclose(fedprox_loss, fedavg_loss, rtol=0, atol=1e-12)

    for key, fedprox_value in fedprox_mu_zero.state_dict().items():
        fedavg_value = fedavg_reference.state_dict()[key]
        assert torch.equal(fedprox_value, fedavg_value), (
            f"Parameter mismatch at {key}"
        )
