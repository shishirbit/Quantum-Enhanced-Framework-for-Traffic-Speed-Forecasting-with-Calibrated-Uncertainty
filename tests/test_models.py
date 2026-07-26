import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
try:
    import pennylane  # noqa: F401
except ImportError:
    PENNYLANE_AVAILABLE = False
else:
    PENNYLANE_AVAILABLE = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quarts.models import (
    AsymmetricRegimeIntervalCalibrator,
    FourierResidualCalibrator,
    FourierRegimeGate,
    FourierUncertaintyCalibrator,
    GatedResidualCalibrator,
    GraphTemporalBackbone,
    HybridForecaster,
    MLPRegimeGate,
    MLPUncertaintyCalibrator,
    QuantumRegimeGate,
    QuantumResidualCalibrator,
    QuantumUncertaintyCalibrator,
    RegimeIntervalCalibrator,
    TorchStatevectorQuantumLayer,
    masked_asymmetric_pinball_loss,
    masked_asymmetric_weighted_interval_score,
    masked_laplace_nll,
    masked_weighted_interval_score,
    parameter_count,
)


@pytest.mark.parametrize("kind", ["entangled", "separable", "fourier"])
def test_residual_forward_backward(kind):
    batch, steps, nodes, features, horizons = 2, 12, 3, 1, 4
    if kind == "fourier":
        calibrator = FourierResidualCalibrator(8, horizons, width=3, depth=2)
    else:
        calibrator = QuantumResidualCalibrator(
            8, horizons, qubits=3, depth=2, entangled=kind == "entangled"
        )
    model = HybridForecaster(GraphTemporalBackbone(features, 8, horizons), calibrator)
    x = torch.randn(batch, steps, nodes, features)
    y = torch.randn(batch, horizons, nodes)
    mask = torch.ones_like(y, dtype=torch.bool)
    output = model(x, torch.eye(nodes))
    assert output.location.shape == y.shape
    assert torch.all(output.scale > 0)
    loss = masked_laplace_nll(output, y, mask)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_noisy_quantum_forward_is_finite():
    if not PENNYLANE_AVAILABLE:
        pytest.skip("PennyLane runtime is blocked by the current Application Control policy")
    calibrator = QuantumResidualCalibrator(
        4, 2, qubits=2, depth=1, depolarizing=0.001, amplitude_damping=0.001
    )
    residual, scale = calibrator(torch.randn(1, 2, 4))
    assert torch.isfinite(residual).all()
    assert torch.isfinite(scale).all() and torch.all(scale > 0)


def test_torch_statevector_matches_single_qubit_ry_expectation():
    layer = TorchStatevectorQuantumLayer(qubits=1, depth=1, entangled=False)
    with torch.no_grad():
        layer.weights.zero_()
    angles = torch.tensor([[0.0], [torch.pi / 2], [torch.pi]])
    measured = layer(angles).squeeze(-1)
    assert torch.allclose(measured, torch.cos(angles.squeeze(-1)), atol=1e-5)


@pytest.mark.parametrize("kind", ["mlp", "fourier", "separable", "quantum"])
def test_uncertainty_head_is_positive_and_parameter_matched(kind):
    hidden, horizons = 20, 12
    if kind == "mlp":
        head = MLPUncertaintyCalibrator(hidden, horizons)
    elif kind == "fourier":
        head = FourierUncertaintyCalibrator(hidden, horizons)
    else:
        head = QuantumUncertaintyCalibrator(
            hidden,
            horizons,
            entangled=kind == "quantum",
        )
    scale = head(torch.randn(3, 7, hidden))
    assert scale.shape == (3, horizons, 7)
    assert torch.isfinite(scale).all() and torch.all(scale > 0)
    scale.mean().backward()
    assert any(parameter.grad is not None for parameter in head.parameters())
    assert 150 <= parameter_count(head) <= 180


@pytest.mark.parametrize("kind", ["mlp", "fourier", "separable", "quantum"])
def test_regime_interval_heads_are_ordered_and_parameter_matched(kind):
    hidden, horizons, experts = 44, 12, 3
    if kind == "mlp":
        gate = MLPRegimeGate(hidden, experts)
    elif kind == "fourier":
        gate = FourierRegimeGate(hidden, experts)
    else:
        gate = QuantumRegimeGate(
            hidden,
            experts,
            entangled=kind == "quantum",
        )
    head = RegimeIntervalCalibrator(gate, horizons, levels=3, experts=experts)
    radii, weights = head(torch.randn(3, 7, hidden))
    assert radii.shape == (3, horizons, 7, 3)
    assert weights.shape == (3, 7, experts)
    assert torch.allclose(weights.sum(-1), torch.ones_like(weights[..., 0]))
    assert torch.all(radii[..., 0] >= radii[..., 1])
    assert torch.all(radii[..., 1] >= radii[..., 2])
    loss = masked_weighted_interval_score(
        torch.randn(3, horizons, 7),
        torch.randn(3, horizons, 7),
        radii,
        torch.ones(3, horizons, 7, dtype=torch.bool),
    )
    loss.backward()
    assert any(parameter.grad is not None for parameter in head.parameters())
    assert 280 <= parameter_count(head) <= 360


def test_constant_interval_head_is_context_invariant():
    head = RegimeIntervalCalibrator(None, horizons=2, levels=3, experts=1)
    first, weights = head(torch.randn(4, 5, 8))
    second, _ = head(torch.randn(4, 5, 8))
    assert torch.equal(first, second)
    assert torch.equal(weights, torch.ones_like(weights))


def test_asymmetric_interval_head_is_ordered_and_backpropagates():
    hidden, horizons, experts = 20, 4, 3
    gate = FourierRegimeGate(hidden, experts=experts)
    head = AsymmetricRegimeIntervalCalibrator(
        gate,
        horizons=horizons,
        levels=3,
        experts=experts,
    )
    lower, upper, weights = head(torch.randn(2, 5, hidden))
    assert lower.shape == upper.shape == (2, horizons, 5, 3)
    assert weights.shape == (2, 5, experts)
    for radii in (lower, upper):
        assert torch.all(radii > 0)
        assert torch.all(radii[..., 0] >= radii[..., 1])
        assert torch.all(radii[..., 1] >= radii[..., 2])
    loss = masked_asymmetric_weighted_interval_score(
        torch.randn(2, horizons, 5),
        torch.randn(2, horizons, 5),
        lower,
        upper,
        torch.ones(2, horizons, 5, dtype=torch.bool),
    )
    loss.backward()
    assert any(parameter.grad is not None for parameter in head.parameters())


def test_asymmetric_wis_rejects_mismatched_radius_shape():
    target = torch.zeros(1, 2, 3)
    with pytest.raises(ValueError):
        masked_asymmetric_weighted_interval_score(
            target,
            target,
            torch.ones(1, 2, 3, 2),
            torch.ones(1, 2, 3, 3),
            torch.ones_like(target, dtype=torch.bool),
        )


def test_asymmetric_pinball_loss_is_finite_and_backpropagates():
    hidden, horizons = 10, 3
    head = AsymmetricRegimeIntervalCalibrator(
        FourierRegimeGate(hidden, experts=3),
        horizons=horizons,
        levels=3,
        experts=3,
    )
    lower, upper, _ = head(torch.randn(2, 4, hidden))
    target = torch.randn(2, horizons, 4)
    loss = masked_asymmetric_pinball_loss(
        target,
        torch.zeros_like(target),
        lower,
        upper,
        torch.ones_like(target, dtype=torch.bool),
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in head.parameters())


def test_asymmetric_pinball_rejects_mismatched_radius_shape():
    target = torch.zeros(1, 2, 3)
    with pytest.raises(ValueError):
        masked_asymmetric_pinball_loss(
            target,
            target,
            torch.ones(1, 2, 3, 2),
            torch.ones(1, 2, 3, 3),
            torch.ones_like(target, dtype=torch.bool),
        )


def test_gated_residual_starts_near_noop_and_backpropagates():
    base = FourierResidualCalibrator(4, horizons=3, width=2, depth=1)
    gated = GatedResidualCalibrator(base, horizons=3, residual_limit=2.0, initial_logit=-8.0)
    latent = torch.randn(2, 5, 4)
    residual, scale = gated(latent)
    assert residual.abs().max() < 0.001
    assert torch.all(scale > 0)
    (residual.sum() + scale.mean()).backward()
    assert gated.gate_logits.grad is not None
    assert any(parameter.grad is not None for parameter in base.parameters())


def test_gated_residual_can_disable_selected_horizons_exactly():
    base = FourierResidualCalibrator(4, horizons=3, width=2, depth=1)
    gated = GatedResidualCalibrator(base, horizons=3, initial_logit=0.0)
    gated.set_residual_mask(torch.tensor([1, 0, 1], dtype=torch.bool))
    residual, _ = gated(torch.randn(2, 5, 4))
    assert torch.equal(residual[:, 1], torch.zeros_like(residual[:, 1]))
    assert gated.gate[1] == 0
