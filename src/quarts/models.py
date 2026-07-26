from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ForecastOutput:
    location: torch.Tensor  # [batch, horizon, node]
    scale: torch.Tensor
    base: torch.Tensor
    residual: torch.Tensor


class GraphTemporalBackbone(nn.Module):
    """Compact reference backbone; SOTA models are external protocol baselines."""

    def __init__(self, features: int, hidden_dim: int, horizons: int):
        super().__init__()
        self.encoder = nn.GRU(2 * features, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, horizons)
        self.hidden_dim = hidden_dim
        self.horizons = horizons

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor):
        # x: [B,T,N,F], adjacency: row-normalized [N,N]
        propagated = torch.einsum("nm,btmf->btnf", adjacency, x)
        features = torch.cat([x, propagated], dim=-1)
        batch, steps, nodes, channels = features.shape
        sequence = features.permute(0, 2, 1, 3).reshape(batch * nodes, steps, channels)
        _, hidden = self.encoder(sequence)
        latent = hidden[-1].reshape(batch, nodes, self.hidden_dim)
        base = self.head(latent).permute(0, 2, 1)
        return base, latent


class TorchStatevectorQuantumLayer(nn.Module):
    """Small differentiable state-vector backend for noiseless circuits.

    It avoids optional native simulator dependencies for the four-qubit
    exploratory model while preserving the RY/Rot/re-upload/CNOT circuit used
    by the PennyLane implementation.
    """

    def __init__(self, qubits: int, depth: int, entangled: bool):
        super().__init__()
        self.qubits = qubits
        self.depth = depth
        self.entangled = entangled
        self.weights = nn.Parameter(torch.empty(depth, qubits, 3))
        nn.init.uniform_(self.weights, 0.0, 2.0 * math.pi)

    @staticmethod
    def _ry(angle: torch.Tensor) -> torch.Tensor:
        cosine, sine = torch.cos(angle / 2), torch.sin(angle / 2)
        return torch.stack(
            [
                torch.stack([cosine, -sine], dim=-1),
                torch.stack([sine, cosine], dim=-1),
            ],
            dim=-2,
        ).to(torch.complex64)

    @staticmethod
    def _rot(angles: torch.Tensor) -> torch.Tensor:
        phi, theta, omega = angles.unbind()
        cosine, sine = torch.cos(theta / 2), torch.sin(theta / 2)
        return torch.stack(
            [
                torch.stack(
                    [
                        torch.exp(-0.5j * (phi + omega)) * cosine,
                        -torch.exp(0.5j * (phi - omega)) * sine,
                    ]
                ),
                torch.stack(
                    [
                        torch.exp(-0.5j * (phi - omega)) * sine,
                        torch.exp(0.5j * (phi + omega)) * cosine,
                    ]
                ),
            ]
        ).to(torch.complex64)

    def _single_qubit(self, state: torch.Tensor, gate: torch.Tensor, wire: int) -> torch.Tensor:
        tensor = state.reshape(state.shape[0], *([2] * self.qubits))
        tensor = torch.movedim(tensor, wire + 1, -1)
        if gate.ndim == 3:
            tensor = torch.einsum("b...i,bji->b...j", tensor, gate)
        else:
            tensor = torch.einsum("b...i,ji->b...j", tensor, gate)
        tensor = torch.movedim(tensor, -1, wire + 1)
        return tensor.reshape(state.shape)

    def _cnot(self, state: torch.Tensor, control: int, target: int) -> torch.Tensor:
        tensor = state.reshape(state.shape[0], *([2] * self.qubits))
        tensor = torch.movedim(tensor, (control + 1, target + 1), (-2, -1))
        shape = tensor.shape
        tensor = tensor.reshape(*shape[:-2], 4)[..., [0, 1, 3, 2]].reshape(shape)
        tensor = torch.movedim(tensor, (-2, -1), (control + 1, target + 1))
        return tensor.reshape(state.shape)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        state = torch.zeros(
            inputs.shape[0],
            2**self.qubits,
            dtype=torch.complex64,
            device=inputs.device,
        )
        state[:, 0] = 1.0
        for layer in range(self.depth):
            for wire in range(self.qubits):
                state = self._single_qubit(state, self._ry(inputs[:, wire]), wire)
                state = self._single_qubit(state, self._rot(self.weights[layer, wire]), wire)
            if self.entangled and self.qubits > 1:
                for wire in range(self.qubits):
                    state = self._cnot(state, wire, (wire + 1) % self.qubits)
        probability = state.abs().square().reshape(inputs.shape[0], *([2] * self.qubits))
        expectations = []
        for wire in range(self.qubits):
            marginal = probability.movedim(wire + 1, -1).reshape(inputs.shape[0], -1, 2).sum(1)
            expectations.append(marginal[:, 0] - marginal[:, 1])
        return torch.stack(expectations, dim=-1)


def _make_quantum_layer(
    qubits: int,
    depth: int,
    entangled: bool,
    shots: int | None,
    depolarizing: float = 0.0,
    amplitude_damping: float = 0.0,
):
    try:
        import pennylane as qml
    except ImportError as exc:
        if shots is not None or depolarizing > 0 or amplitude_damping > 0:
            raise RuntimeError(
                "PennyLane is required for finite-shot or noisy quantum simulation"
            ) from exc
        return TorchStatevectorQuantumLayer(qubits, depth, entangled)

    noisy = depolarizing > 0 or amplitude_damping > 0
    device = qml.device("default.mixed" if noisy else "default.qubit", wires=qubits, shots=shots)

    @qml.qnode(device, interface="torch", diff_method="backprop" if shots is None else "parameter-shift")
    def circuit(inputs, weights):
        for layer in range(depth):
            # Re-upload the compressed local state at every circuit layer.
            for wire in range(qubits):
                qml.RY(inputs[..., wire], wires=wire)
                qml.Rot(*weights[layer, wire], wires=wire)
            if entangled and qubits > 1:
                for wire in range(qubits):
                    qml.CNOT(wires=[wire, (wire + 1) % qubits])
            if noisy:
                for wire in range(qubits):
                    if depolarizing > 0:
                        qml.DepolarizingChannel(depolarizing, wires=wire)
                    if amplitude_damping > 0:
                        qml.AmplitudeDamping(amplitude_damping, wires=wire)
        return [qml.expval(qml.PauliZ(wire)) for wire in range(qubits)]

    return qml.qnn.TorchLayer(circuit, {"weights": (depth, qubits, 3)})


class QuantumResidualCalibrator(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        horizons: int,
        qubits: int = 4,
        depth: int = 2,
        entangled: bool = True,
        shots: int | None = None,
        depolarizing: float = 0.0,
        amplitude_damping: float = 0.0,
    ):
        super().__init__()
        self.compress = nn.Linear(hidden_dim, qubits)
        self.quantum = _make_quantum_layer(
            qubits, depth, entangled, shots, depolarizing, amplitude_damping
        )
        self.readout = nn.Linear(qubits, 2 * horizons)
        self.horizons = horizons

    def forward(self, latent: torch.Tensor):
        batch, nodes, hidden = latent.shape
        angles = math.pi * torch.tanh(self.compress(latent.reshape(batch * nodes, hidden)))
        measured = self.quantum(angles)
        if isinstance(measured, (list, tuple)):
            measured = torch.stack(measured, dim=-1)
        outputs = self.readout(measured).reshape(batch, nodes, 2, self.horizons)
        residual = outputs[:, :, 0].permute(0, 2, 1)
        scale = F.softplus(outputs[:, :, 1].permute(0, 2, 1)) + 1e-4
        return residual, scale


class FourierResidualCalibrator(nn.Module):
    """Critical classical control for a data-reuploading Fourier-like bias."""

    def __init__(self, hidden_dim: int, horizons: int, width: int = 4, depth: int = 2):
        super().__init__()
        self.compress = nn.Linear(hidden_dim, width)
        self.phase = nn.Parameter(torch.zeros(depth, width))
        self.mix = nn.ModuleList([nn.Linear(2 * width, width) for _ in range(depth)])
        self.readout = nn.Linear(width, 2 * horizons)
        self.horizons = horizons

    def forward(self, latent: torch.Tensor):
        batch, nodes, hidden = latent.shape
        state = math.pi * torch.tanh(self.compress(latent.reshape(batch * nodes, hidden)))
        for index, layer in enumerate(self.mix, start=1):
            periodic = torch.cat(
                [torch.sin(index * state + self.phase[index - 1]),
                 torch.cos(index * state + self.phase[index - 1])], dim=-1
            )
            state = torch.tanh(layer(periodic))
        outputs = self.readout(state).reshape(batch, nodes, 2, self.horizons)
        residual = outputs[:, :, 0].permute(0, 2, 1)
        scale = F.softplus(outputs[:, :, 1].permute(0, 2, 1)) + 1e-4
        return residual, scale


class MLPResidualCalibrator(nn.Module):
    def __init__(self, hidden_dim: int, horizons: int, width: int = 8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim, width), nn.Tanh(), nn.Linear(width, 2 * horizons))
        self.horizons = horizons

    def forward(self, latent: torch.Tensor):
        outputs = self.net(latent)
        residual = outputs[..., : self.horizons].permute(0, 2, 1)
        scale = F.softplus(outputs[..., self.horizons :].permute(0, 2, 1)) + 1e-4
        return residual, scale


class MLPUncertaintyCalibrator(nn.Module):
    """Predict conditional Laplace scale without changing the point forecast."""

    def __init__(self, hidden_dim: int, horizons: int, width: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, width),
            nn.Tanh(),
            nn.Linear(width, horizons),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(latent)).permute(0, 2, 1) + 1e-4


class FourierUncertaintyCalibrator(nn.Module):
    """Parameter-matched periodic control for conditional uncertainty."""

    def __init__(self, hidden_dim: int, horizons: int, width: int = 3, depth: int = 2):
        super().__init__()
        self.compress = nn.Linear(hidden_dim, width)
        self.phase = nn.Parameter(torch.zeros(depth, width))
        self.mix = nn.ModuleList(
            [nn.Linear(2 * width, width) for _ in range(depth)]
        )
        self.readout = nn.Linear(width, horizons)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        batch, nodes, hidden = latent.shape
        state = math.pi * torch.tanh(
            self.compress(latent.reshape(batch * nodes, hidden))
        )
        for index, layer in enumerate(self.mix, start=1):
            periodic = torch.cat(
                [
                    torch.sin(index * state + self.phase[index - 1]),
                    torch.cos(index * state + self.phase[index - 1]),
                ],
                dim=-1,
            )
            state = torch.tanh(layer(periodic))
        output = self.readout(state).reshape(batch, nodes, -1)
        return F.softplus(output).permute(0, 2, 1) + 1e-4


class QuantumUncertaintyCalibrator(nn.Module):
    """Shared VQC that predicts scale only, preserving the frozen backbone median."""

    def __init__(
        self,
        hidden_dim: int,
        horizons: int,
        qubits: int = 4,
        depth: int = 2,
        entangled: bool = True,
    ):
        super().__init__()
        self.compress = nn.Linear(hidden_dim, qubits)
        self.quantum = _make_quantum_layer(
            qubits,
            depth,
            entangled,
            shots=None,
        )
        self.readout = nn.Linear(qubits, horizons)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        batch, nodes, hidden = latent.shape
        angles = math.pi * torch.tanh(
            self.compress(latent.reshape(batch * nodes, hidden))
        )
        measured = self.quantum(angles)
        if isinstance(measured, (list, tuple)):
            measured = torch.stack(measured, dim=-1)
        output = self.readout(measured).reshape(batch, nodes, -1)
        return F.softplus(output).permute(0, 2, 1) + 1e-4


class MLPRegimeGate(nn.Module):
    """Small classical gate for a shared bank of uncertainty experts."""

    def __init__(self, hidden_dim: int, experts: int = 3, width: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, width),
            nn.Tanh(),
            nn.Linear(width, experts),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


class FourierRegimeGate(nn.Module):
    """Periodic parameter-matched control for regime selection."""

    def __init__(
        self,
        hidden_dim: int,
        experts: int = 3,
        width: int = 3,
        depth: int = 2,
    ):
        super().__init__()
        self.compress = nn.Linear(hidden_dim, width)
        self.phase = nn.Parameter(torch.zeros(depth, width))
        self.mix = nn.ModuleList(
            [nn.Linear(2 * width, width) for _ in range(depth)]
        )
        self.readout = nn.Linear(width, experts)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        state = math.pi * torch.tanh(self.compress(latent))
        for index, layer in enumerate(self.mix, start=1):
            periodic = torch.cat(
                [
                    torch.sin(index * state + self.phase[index - 1]),
                    torch.cos(index * state + self.phase[index - 1]),
                ],
                dim=-1,
            )
            state = torch.tanh(layer(periodic))
        return self.readout(state)


class QuantumRegimeGate(nn.Module):
    """Four-qubit gate whose only role is selecting uncertainty regimes."""

    def __init__(
        self,
        hidden_dim: int,
        experts: int = 3,
        qubits: int = 4,
        depth: int = 2,
        entangled: bool = True,
    ):
        super().__init__()
        self.compress = nn.Linear(hidden_dim, qubits)
        self.quantum = _make_quantum_layer(
            qubits,
            depth,
            entangled,
            shots=None,
        )
        self.readout = nn.Linear(qubits, experts)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        angles = math.pi * torch.tanh(self.compress(latent))
        measured = self.quantum(angles)
        if isinstance(measured, (list, tuple)):
            measured = torch.stack(measured, dim=-1)
        return self.readout(measured)


class RegimeIntervalCalibrator(nn.Module):
    """Mix ordered interval-radius experts with a context-dependent gate.

    Radii are returned in the order supplied by the experiment, conventionally
    alpha=(0.1, 0.2, 0.5). Positive increments and a reverse cumulative sum
    guarantee that higher-coverage intervals contain lower-coverage intervals.
    """

    def __init__(
        self,
        gate: nn.Module | None,
        horizons: int,
        levels: int = 3,
        experts: int = 3,
    ):
        super().__init__()
        if levels < 1 or experts < 1:
            raise ValueError("levels and experts must both be positive")
        if gate is None and experts != 1:
            raise ValueError("A context-free calibrator must use one expert")
        self.gate = gate
        self.horizons = horizons
        self.levels = levels
        self.experts = experts
        offsets = torch.linspace(-0.5, 0.5, experts).view(experts, 1, 1)
        self.raw_increments = nn.Parameter(
            offsets.expand(experts, horizons, levels).clone()
        )

    def forward(
        self, latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, nodes, _ = latent.shape
        if self.gate is None:
            weights = latent.new_ones(batch, nodes, 1)
        else:
            weights = torch.softmax(self.gate(latent), dim=-1)
        increments = F.softplus(self.raw_increments) + 1e-4
        mixed = torch.einsum("bne,ehl->bnhl", weights, increments)
        radii = torch.flip(
            torch.cumsum(torch.flip(mixed, dims=(-1,)), dim=-1),
            dims=(-1,),
        )
        return radii.permute(0, 2, 1, 3), weights


class AsymmetricRegimeIntervalCalibrator(nn.Module):
    """Mix ordered lower/upper interval-radius experts independently."""

    def __init__(
        self,
        gate: nn.Module | None,
        horizons: int,
        levels: int = 3,
        experts: int = 3,
    ):
        super().__init__()
        if levels < 1 or experts < 1:
            raise ValueError("levels and experts must both be positive")
        if gate is None and experts != 1:
            raise ValueError("A context-free calibrator must use one expert")
        self.gate = gate
        self.horizons = horizons
        self.levels = levels
        self.experts = experts
        offsets = torch.linspace(-0.5, 0.5, experts).view(experts, 1, 1, 1)
        self.raw_increments = nn.Parameter(
            offsets.expand(experts, horizons, levels, 2).clone()
        )

    def forward(
        self, latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, nodes, _ = latent.shape
        if self.gate is None:
            weights = latent.new_ones(batch, nodes, 1)
        else:
            weights = torch.softmax(self.gate(latent), dim=-1)
        increments = F.softplus(self.raw_increments) + 1e-4
        mixed = torch.einsum("bne,ehls->bnhls", weights, increments)
        radii = torch.flip(
            torch.cumsum(torch.flip(mixed, dims=(-2,)), dim=-2),
            dims=(-2,),
        ).permute(0, 2, 1, 3, 4)
        return radii[..., 0], radii[..., 1], weights


def masked_weighted_interval_score(
    target: torch.Tensor,
    location: torch.Tensor,
    radii: torch.Tensor,
    mask: torch.Tensor,
    alphas: tuple[float, ...] = (0.1, 0.2, 0.5),
) -> torch.Tensor:
    """Differentiable WIS for symmetric intervals about a fixed median."""

    if radii.shape != (*target.shape, len(alphas)):
        raise ValueError(
            f"Expected radii shape {(*target.shape, len(alphas))}, "
            f"received {tuple(radii.shape)}"
        )
    error = target - location
    aggregate = 0.5 * error.abs()
    for level, alpha in enumerate(alphas):
        radius = radii[..., level].clamp_min(1e-6)
        lower = location - radius
        upper = location + radius
        score = (
            2 * radius
            + (2 / alpha) * (lower - target) * (target < lower)
            + (2 / alpha) * (target - upper) * (target > upper)
        )
        aggregate = aggregate + (alpha / 2) * score
    aggregate = aggregate / (len(alphas) + 0.5)
    weight = mask.to(aggregate.dtype)
    return (aggregate * weight).sum() / weight.sum().clamp_min(1)


def masked_asymmetric_weighted_interval_score(
    target: torch.Tensor,
    location: torch.Tensor,
    lower_radii: torch.Tensor,
    upper_radii: torch.Tensor,
    mask: torch.Tensor,
    alphas: tuple[float, ...] = (0.1, 0.2, 0.5),
) -> torch.Tensor:
    """Differentiable WIS for asymmetric intervals about a fixed median."""

    expected = (*target.shape, len(alphas))
    if lower_radii.shape != expected or upper_radii.shape != expected:
        raise ValueError(
            f"Expected radius shapes {expected}, received "
            f"{tuple(lower_radii.shape)} and {tuple(upper_radii.shape)}"
        )
    aggregate = 0.5 * (target - location).abs()
    for level, alpha in enumerate(alphas):
        lower_radius = lower_radii[..., level].clamp_min(1e-6)
        upper_radius = upper_radii[..., level].clamp_min(1e-6)
        lower = location - lower_radius
        upper = location + upper_radius
        score = (
            lower_radius
            + upper_radius
            + (2 / alpha) * (lower - target) * (target < lower)
            + (2 / alpha) * (target - upper) * (target > upper)
        )
        aggregate = aggregate + (alpha / 2) * score
    aggregate = aggregate / (len(alphas) + 0.5)
    weight = mask.to(aggregate.dtype)
    return (aggregate * weight).sum() / weight.sum().clamp_min(1)


def masked_asymmetric_pinball_loss(
    target: torch.Tensor,
    location: torch.Tensor,
    lower_radii: torch.Tensor,
    upper_radii: torch.Tensor,
    mask: torch.Tensor,
    alphas: tuple[float, ...] = (0.1, 0.2, 0.5),
) -> torch.Tensor:
    """Pinball loss for nested central intervals around a fixed location.

    The lower and upper residual quantiles are represented as ``-lower_radii``
    and ``upper_radii`` at probabilities alpha/2 and 1-alpha/2.
    """

    expected = (*target.shape, len(alphas))
    if lower_radii.shape != expected or upper_radii.shape != expected:
        raise ValueError(
            f"Expected radius shapes {expected}, received "
            f"{tuple(lower_radii.shape)} and {tuple(upper_radii.shape)}"
        )
    residual = target - location
    aggregate = torch.zeros_like(residual)
    for level, alpha in enumerate(alphas):
        lower_quantile = -lower_radii[..., level].clamp_min(1e-6)
        upper_quantile = upper_radii[..., level].clamp_min(1e-6)
        lower_error = residual - lower_quantile
        upper_error = residual - upper_quantile
        lower_tau = alpha / 2
        upper_tau = 1 - alpha / 2
        aggregate = aggregate + torch.maximum(
            lower_tau * lower_error,
            (lower_tau - 1) * lower_error,
        )
        aggregate = aggregate + torch.maximum(
            upper_tau * upper_error,
            (upper_tau - 1) * upper_error,
        )
    aggregate = aggregate / (2 * len(alphas))
    weight = mask.to(aggregate.dtype)
    return (aggregate * weight).sum() / weight.sum().clamp_min(1)


class GatedResidualCalibrator(nn.Module):
    """Bound a residual head and initialize it close to a no-op.

    The v1 residual heads could immediately perturb an already useful base
    forecast. This wrapper constrains the correction in normalized target units
    and learns a separate per-horizon gate. A negative initial logit preserves
    the backbone while still allowing gradients to reach the wrapped head.
    """

    def __init__(
        self,
        calibrator: nn.Module,
        horizons: int,
        residual_limit: float = 2.0,
        initial_logit: float = -6.0,
    ):
        super().__init__()
        self.calibrator = calibrator
        self.gate_logits = nn.Parameter(torch.full((horizons,), float(initial_logit)))
        self.register_buffer("residual_enabled", torch.ones(horizons))
        self.residual_limit = float(residual_limit)

    def forward(self, latent: torch.Tensor):
        raw_residual, scale = self.calibrator(latent)
        gain = (torch.sigmoid(self.gate_logits) * self.residual_enabled)[None, :, None]
        residual = self.residual_limit * gain * torch.tanh(raw_residual)
        return residual, scale

    def set_residual_mask(self, enabled: torch.Tensor) -> None:
        """Enable only residual horizons that improve validation MAE."""
        enabled = torch.as_tensor(enabled, device=self.residual_enabled.device)
        if enabled.shape != self.residual_enabled.shape:
            raise ValueError(
                f"Expected residual mask {tuple(self.residual_enabled.shape)}, "
                f"received {tuple(enabled.shape)}"
            )
        self.residual_enabled.copy_(enabled.to(self.residual_enabled.dtype))

    @property
    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logits) * self.residual_enabled


class HybridForecaster(nn.Module):
    def __init__(self, backbone: GraphTemporalBackbone, calibrator: nn.Module | None):
        super().__init__()
        self.backbone = backbone
        self.calibrator = calibrator
        self.default_log_scale = nn.Parameter(torch.zeros(backbone.horizons))

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> ForecastOutput:
        base, latent = self.backbone(x, adjacency)
        if self.calibrator is None:
            residual = torch.zeros_like(base)
            scale = F.softplus(self.default_log_scale)[None, :, None].expand_as(base) + 1e-4
        else:
            residual, scale = self.calibrator(latent)
        return ForecastOutput(base + residual, scale, base, residual)


def masked_laplace_nll(output: ForecastOutput, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    scale = output.scale.clamp_min(1e-6)
    loss = torch.log(2 * scale) + torch.abs(target - output.location) / scale
    weight = mask.to(loss.dtype)
    return (loss * weight).sum() / weight.sum().clamp_min(1)


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
