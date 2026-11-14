import pytest

from ndhs_flower.server_app import _resolve_proximal_mu


def test_fedprox_accepts_prespecified_mu():
    assert _resolve_proximal_mu("fedprox", 0.01) == pytest.approx(0.01)


@pytest.mark.parametrize("mu", [0.0, 0.005, 0.02, 0.1])
def test_fedprox_rejects_other_mu_values(mu):
    with pytest.raises(ValueError, match="prespecified proximal-mu = 0.01"):
        _resolve_proximal_mu("fedprox", mu)


def test_fedavg_always_uses_zero_proximal_mu():
    assert _resolve_proximal_mu("fedavg", 0.01) == 0.0