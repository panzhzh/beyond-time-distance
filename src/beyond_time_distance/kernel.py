"""Forecast-shape kernels, matched controls, and change-CRPS calibration."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import torch
from torch import Tensor

from .base import ChangePairs, DependenceConfig, ForecastInputs, change_crps, profile_features


@dataclass(frozen=True)
class FitConfig:
    horizon: int = 48
    lags: tuple[int, ...] = (1, 2, 4, 8, 12)
    gamma_fit: tuple[float, ...] = (0.125, 1.0)
    penalties: tuple[float, ...] = (1e-6, 1e-4, 1e-2)
    gammas: tuple[float, ...] = (0.0, 0.03125, 0.0625, 0.125, 0.25, 0.5, 0.75, 1.0)
    initial_strength: float = 0.01
    max_iter: int = 120

    def __post_init__(self):
        DependenceConfig(horizon=self.horizon, lags=tuple(self.lags))
        if not self.penalties or any(not math.isfinite(x) or x <= 0 for x in self.penalties):
            raise ValueError("Penalties must be finite and positive")
        if not self.gamma_fit or any(not 0 < x <= 1 for x in self.gamma_fit):
            raise ValueError("Fitting interpolation must lie in (0, 1]")
        if not self.gammas or any(not 0 <= x <= 1 for x in self.gammas):
            raise ValueError("Deployment interpolation must lie in [0, 1]")
        if self.max_iter < 1 or not math.isfinite(self.initial_strength) or self.initial_strength <= 0:
            raise ValueError("Invalid optimizer settings")

    def pairs(self, device):
        return ChangePairs.create(DependenceConfig(horizon=self.horizon, lags=tuple(self.lags)), device)


@dataclass
class CalibrationBatch:
    """A training or validation batch; references are full matrices or scored pairs."""
    quantiles: Tensor
    reference: Tensor
    target: Tensor
    scale: Tensor

    def validate(self, config):
        q = self.quantiles
        if q.ndim != 3 or q.shape[1:] != (config.horizon, 9) or not len(q):
            raise ValueError("Expected nonempty N x H x 9 quantiles")
        if self.target.shape != q.shape[:2] or self.scale.shape != (len(q),):
            raise ValueError("Targets must be N x H and history scales must be N")
        for value in (q, self.reference, self.target, self.scale):
            if value.device != q.device or value.dtype != torch.float64:
                raise ValueError("All calibration tensors must share a device and use float64")
            if not bool(torch.isfinite(value).all()):
                raise ValueError("Calibration inputs must be finite")
        if not bool((self.scale > 0).all()):
            raise ValueError("History scales must be positive")
        pairs = config.pairs(q.device)
        select_reference(self.reference, pairs, len(q), config.horizon)


def raw_coordinates(quantiles):
    """Within-query centered spread, location, adjacent change, and tail ratio."""
    proxy = ForecastInputs(quantiles.new_empty((len(quantiles), 0)), quantiles,
                           quantiles.new_ones(len(quantiles)))
    return profile_features(proxy)[:, :, [0, 1, 2, 4]]


def time_coordinates(quantiles):
    h = quantiles.shape[1]
    t = torch.linspace(-1, 1, h, device=quantiles.device, dtype=quantiles.dtype)
    q, r = torch.linalg.qr(torch.stack([t**k for k in range(5)], -1))
    q = q * torch.where(r.diag() >= 0, 1., -1.)
    return q[:, 1:5][None].expand(len(quantiles), -1, -1)


def select_reference(reference, pairs, batch, horizon):
    if reference.shape == (batch, horizon, horizon):
        selected = reference if pairs is None else reference[:, pairs.left, pairs.right]
    elif pairs is not None and reference.shape == (batch, len(pairs.left)):
        selected = reference
    else:
        raise ValueError("Reference shape must match full horizons or the requested pairs")
    if not bool(torch.isfinite(selected).all()) or bool((selected.abs() > 1 + 1e-10).any()):
        raise ValueError("Reference correlations must be finite and lie in [-1, 1]")
    return selected


@dataclass
class GeometryRefinement:
    """A portable four-strength refinement on a supplied correlation reference."""
    horizon: int
    coefficient: Tensor
    gamma: float
    center: Tensor
    sd: Tensor
    distance_factor: Tensor
    feature: str = "geometry"
    response: str = "rational"
    representation: str = "shape"
    shift: int = 0
    operator: str = "schur"

    def __post_init__(self):
        if self.feature not in ("geometry", "time") or self.response not in ("rational", "gaussian", "laplace"):
            raise ValueError("Unknown feature or kernel response")
        if self.representation != "shape" or self.operator not in ("schur", "mixture"):
            raise ValueError("Unknown representation or correlation operator")
        if not 0 <= self.gamma <= 1 or not 0 <= self.shift < self.horizon:
            raise ValueError("Invalid interpolation or circular shift")
        if self.coefficient.shape != (4,) or self.distance_factor.shape != (4,):
            raise ValueError("Exactly four strengths and distance factors are required")
        if self.center.shape not in ((1, 1, 4), (self.horizon, 4)) or self.sd.shape != (4,):
            raise ValueError("Invalid coordinate normalization shapes")
        for value in (self.coefficient, self.center, self.sd, self.distance_factor):
            if value.device != self.coefficient.device or value.dtype != torch.float64 or not bool(torch.isfinite(value).all()):
                raise ValueError("Model state must be finite float64 on one device")
        if not bool((self.sd > 0).all()) or not bool((self.distance_factor > 0).all()):
            raise ValueError("Normalization scales and distance factors must be positive")

    @property
    def strengths(self):
        return torch.nn.functional.softplus(self.coefficient)

    def coordinates(self, quantiles):
        if quantiles.ndim != 3 or quantiles.shape[1:] != (self.horizon, 9):
            raise ValueError("Expected N x H x 9 raw quantiles")
        if quantiles.device != self.coefficient.device or quantiles.dtype != torch.float64:
            raise ValueError("Quantiles must match the model device and float64 precision")
        if not bool(torch.isfinite(quantiles).all()):
            raise ValueError("Quantiles must be finite")
        raw = time_coordinates(quantiles) if self.feature == "time" else raw_coordinates(quantiles)
        z = ((raw - self.center) / self.sd).clamp(-8, 8)
        return torch.roll(z, self.shift, 1) if self.shift else z

    def kernel(self, quantiles, pairs=None):
        z = self.coordinates(quantiles)
        delta = z[:, :, None] - z[:, None, :] if pairs is None else z[:, pairs.left] - z[:, pairs.right]
        distance = delta.abs() if self.response == "laplace" else delta.square() * self.distance_factor
        exponent = (distance * self.strengths).sum(-1)
        return 1 / (1 + exponent) if self.response == "rational" else torch.exp(-exponent)

    @torch.no_grad()
    def predict(self, quantiles, reference, pairs=None):
        reference = select_reference(reference, pairs, len(quantiles), self.horizon)
        if reference.device != self.coefficient.device or reference.dtype != torch.float64:
            raise ValueError("Reference must match the model device and float64 precision")
        kernel = self.kernel(quantiles, pairs)
        proposal = reference * kernel if self.operator == "schur" else kernel
        return (1 - self.gamma) * reference + self.gamma * proposal

    def to(self, device):
        fields = dict(vars(self))
        for key, value in fields.items():
            if torch.is_tensor(value): fields[key] = value.to(device)
        return type(self)(**fields)

    def save(self, path):
        payload = {k: v.detach().cpu().tolist() if torch.is_tensor(v) else v for k, v in vars(self).items()}
        with Path(path).open("x") as stream:
            json.dump({"schema": 1, **payload}, stream, indent=2, allow_nan=False)
            stream.write("\n")

    @classmethod
    def load(cls, path, device="cpu"):
        payload = json.loads(Path(path).read_text())
        if payload.pop("schema") != 1:
            raise ValueError("Unsupported refinement schema")
        for key in ("coefficient", "center", "sd", "distance_factor"):
            payload[key] = torch.tensor(payload[key], device=device, dtype=torch.float64)
        return cls(**payload)


def fit_initializer(train, validation, *, temperature=1.0, config=None, feature="geometry",
                   response="rational", representation="shape", shift=0, operator="schur"):
    """Fit six strength vectors and select interpolation by validation change CRPS."""
    config = config or FitConfig()
    train.validate(config); validation.validate(config)
    if train.quantiles.device != validation.quantiles.device:
        raise ValueError("Training and validation must use the same device")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    quantiles = torch.cat([train.quantiles, validation.quantiles])
    device = quantiles.device
    pairs = config.pairs(device)
    reference = torch.cat([select_reference(b.reference, pairs, len(b.quantiles), config.horizon)
                           for b in (train, validation)]).detach()
    training = torch.arange(len(train.quantiles), device=device)
    calibration = torch.arange(len(train.quantiles), len(quantiles), device=device)
    scale = torch.cat([train.scale, validation.scale]).detach()
    mean = quantiles[:, :, 4].detach()
    sigma = ((quantiles[:, :, 8] - quantiles[:, :, 0]) / (2 * 1.2815515655446004)).clamp_min(1e-8) * temperature
    sigma = sigma.detach()
    raw = time_coordinates(quantiles) if feature == "time" else raw_coordinates(quantiles)
    if feature == "time":
        center = raw.new_zeros((1, 1, 4)); residual = raw
    else:
        center = raw[training].mean((0, 1), keepdim=True); residual = raw - center
    sd = raw[training].std((0, 1), correction=0).clamp_min(1e-6)
    z = (residual / sd).clamp(-8, 8).detach()
    if shift: z = torch.roll(z, shift, 1)
    delta = z[:, pairs.left] - z[:, pairs.right]
    absolute = (delta[training].abs() * pairs.weights[None, :, None]).sum(1).mean(0)
    square = (delta[training].square() * pairs.weights[None, :, None]).sum(1).mean(0)
    factor = torch.where(square > 1e-24, absolute / square.clamp_min(1e-24), torch.ones_like(square))
    if response == "laplace": factor = torch.ones_like(factor)
    design = delta.abs() if response == "laplace" else delta.square() * factor
    # Validate configuration before optimization, including unsupported response names.
    GeometryRefinement(config.horizon, z.new_zeros(4), 0., center, sd, factor,
                       feature, response, representation, shift, operator)

    def loss(correlation, index, target):
        return (change_crps(mean[index], sigma[index], correlation, target, scale[index], pairs) @ pairs.weights).mean()

    def proposed(theta, index):
        exponent = (design[index] * torch.nn.functional.softplus(theta)).sum(-1)
        kernel = 1 / (1 + exponent) if response == "rational" else torch.exp(-exponent)
        return reference[index] * kernel if operator == "schur" else kernel

    norm = loss(reference[training], training, train.target).detach()
    if not bool(torch.isfinite(norm)) or float(norm) <= 0:
        raise ValueError("Reference training risk must be finite and positive")
    menu, candidates, optimization = [], [], []
    for gamma_fit in config.gamma_fit:
        for penalty in config.penalties:
            theta = quantiles.new_full((4,), math.log(math.expm1(config.initial_strength)), requires_grad=True)
            optimizer = torch.optim.LBFGS([theta], max_iter=config.max_iter, history_size=20,
                                         line_search_fn="strong_wolfe", tolerance_grad=1e-7, tolerance_change=1e-10)
            calls = 0

            def closure():
                nonlocal calls
                optimizer.zero_grad()
                corr = (1 - gamma_fit) * reference[training] + gamma_fit * proposed(theta, training)
                value = loss(corr, training, train.target) / norm + penalty * torch.nn.functional.softplus(theta).square().sum() / 2
                value.backward(); calls += 1
                return value

            optimizer.step(closure)
            objective = float(closure().detach())
            if not math.isfinite(objective) or not bool(torch.isfinite(theta).all()):
                raise ValueError("Nonfinite fitted strength vector")
            candidates.append(theta.detach().clone())
            optimization.append({"gamma_fit": gamma_fit, "penalty": penalty, "objective": objective,
                                 "closure_calls": calls, "max_gradient": float(theta.grad.abs().max())})
            with torch.no_grad():
                proposal = proposed(theta, calibration)
                for gamma in config.gammas:
                    score = float(loss((1 - gamma) * reference[calibration] + gamma * proposal,
                                       calibration, validation.target))
                    menu.append({"fit_index": len(candidates) - 1, "gamma_fit": gamma_fit, "penalty": penalty,
                                 "gamma": gamma, "calibration_risk": score})
    selected = min(menu, key=lambda r: (r["calibration_risk"], r["gamma"], -r["penalty"], r["gamma_fit"]))
    head = GeometryRefinement(config.horizon, candidates[selected["fit_index"]], selected["gamma"],
                              center.detach(), sd.detach(), factor.detach(), feature, response,
                              representation, shift, operator)
    trace = {"config": asdict(config), "selection": selected, "menu": menu, "optimization": optimization,
             "reference_calibration_risk": float(loss(reference[calibration], calibration, validation.target))}
    return head, trace
