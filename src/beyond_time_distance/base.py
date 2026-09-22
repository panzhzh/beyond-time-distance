"""Forecast-conditioned temporal dependence with shared predictive marginals.

The fitted object needs only a history and nine fine forecast quantiles at
inference. All normalizers use training inputs; targets enter ``fit`` only as
separate training and calibration tensors. Local gates add ten coefficients to
the global context gate, and preserve positive semidefiniteness and unit diagonal.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

Z90 = 1.2815515655446004
INITIAL_LOGIT = math.log(.01 / .98)
PROFILE_COLUMNS = {"full_profile": (0, 1, 2, 3, 4)}


@dataclass(frozen=True)
class DependenceConfig:
    horizon: int = 48
    lags: tuple[int, ...] = (1, 2, 4, 8, 12)
    rho_grid: tuple[float, ...] = (-.8, -.4, 0., .2, .4, .6, .7, .8, .85, .9, .92, .94,
                                  .95, .96, .97, .98, .99, .995, .999, 1.)
    identity_weights: tuple[float, ...] = (0., .001, .01, .05, .1, .25, .5, 1.)
    penalties: tuple[float, ...] = (1e-5, 1e-4, .001)
    local_penalties: tuple[float, ...] | None = None
    gammas: tuple[float, ...] = (0., .03125, .0625, .125, .25, .5, .75, 1.)
    global_iterations: int = 80
    local_iterations: int = 120
    arms: tuple[str, ...] = tuple(PROFILE_COLUMNS)

    def __post_init__(self):
        if self.horizon <= 12 or not self.lags or any(k <= 0 or k >= self.horizon for k in self.lags):
            raise ValueError("Every change lag must lie strictly inside the forecast horizon")
        if len(set(self.lags)) != len(self.lags) or len(set(self.arms)) != len(self.arms):
            raise ValueError("Duplicate lags or arms")
        if not self.arms or set(self.arms) - PROFILE_COLUMNS.keys():
            raise ValueError("Unknown or empty profile arm menu")
        for values, low, high in ((self.rho_grid, -1, 1), (self.identity_weights, 0, 1),
                                  (self.gammas, 0, 1)):
            if not values or any(not low <= v <= high for v in values):
                raise ValueError("Invalid correlation or mixture menu")
        if not self.penalties or any(not math.isfinite(v) or v <= 0 for v in self.penalties):
            raise ValueError("Penalties must be finite and positive")
        if self.local_penalties is not None and (not self.local_penalties or any(
                not math.isfinite(v) or v <= 0 for v in self.local_penalties)):
            raise ValueError("Local penalties must be finite and positive")


@dataclass
class ForecastInputs:
    history: Tensor
    quantiles: Tensor
    scale: Tensor

    def validate(self, horizon: int):
        n = len(self.scale)
        if self.history.ndim != 2 or self.history.shape[0] != n or self.history.shape[1] % 12:
            raise ValueError("Histories must be a batch of lengths divisible by 12")
        if self.history.shape[1] <= 12 or self.quantiles.shape != (n, horizon, 9) or self.scale.shape != (n,):
            raise ValueError("Expected history, N x H x 9 quantiles, and N input-history IQRs")
        for value in (self.history, self.quantiles, self.scale):
            if value.dtype != torch.float64 or value.device != self.quantiles.device:
                raise ValueError("Inputs must share a device and use float64")
            if not bool(torch.isfinite(value).all()):
                raise ValueError("Nonfinite input")
        if not bool((self.scale > 0).all()):
            raise ValueError("History scales must be positive")

    @property
    def location(self):
        return self.quantiles[:, :, 4]

    @property
    def raw_sigma(self):
        return ((self.quantiles[:, :, 8] - self.quantiles[:, :, 0]) / (2 * Z90)).clamp_min(1e-8)


@dataclass
class ChangePairs:
    left: Tensor
    right: Tensor
    groups: Tensor
    weights: Tensor

    @classmethod
    def create(cls, config: DependenceConfig, device):
        h, lags = config.horizon, config.lags
        return cls(
            torch.cat([torch.arange(h - k, device=device) for k in lags]),
            torch.cat([torch.arange(k, h, device=device) for k in lags]),
            torch.cat([torch.full((h - k,), i, device=device) for i, k in enumerate(lags)]),
            torch.cat([torch.full((h - k,), 1 / len(lags) / (h - k), device=device,
                                 dtype=torch.float64) for k in lags]),
        )


def gaussian_crps(location: Tensor, sigma: Tensor, target: Tensor) -> Tensor:
    z = (target - location) / sigma
    return sigma * (z * torch.erf(z / math.sqrt(2)) + math.sqrt(2 / math.pi)
                    * torch.exp(-z.square() / 2) - 1 / math.sqrt(math.pi))


def change_crps(location, sigma, correlation, target, scale, pairs: ChangePairs):
    a, b = sigma[:, pairs.left], sigma[:, pairs.right]
    variance = (a.square() + b.square() - 2 * a * b * correlation).clamp_min(1e-16)
    return gaussian_crps(location[:, pairs.right] - location[:, pairs.left], variance.sqrt(),
                         target[:, pairs.right] - target[:, pairs.left]) / scale[:, None]


def shared_temperature(location, sigma, target, scale):
    """Convex fine-marginal CRPS calibration, shared by every dependence arm."""
    sd, error = sigma / scale[:, None], (target - location) / scale[:, None]
    low, high = .05, 20.
    for _ in range(48):
        mid = (low + high) / 2
        derivative = (sd * (math.sqrt(2) * torch.exp(-.5 * (error / (mid * sd)).square()) - 1)).mean()
        if float(derivative) < 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def residual_correlation(error, center):
    x = error - error.mean(0) if center else error
    moment = x.T @ x / len(x)
    sd = moment.diag().sqrt().clamp_min(1e-12)
    r = moment / sd[:, None] / sd[None, :]
    return r - torch.diag_embed(r.diag()) + torch.eye(r.shape[0], device=x.device, dtype=x.dtype)


def history_correlation(history):
    a = history[:, :-1] - history[:, :-1].mean(1, keepdim=True)
    b = history[:, 1:] - history[:, 1:].mean(1, keepdim=True)
    denominator = (a.square().sum(1) * b.square().sum(1)).sqrt()
    return torch.where(denominator > 1e-20, (a * b).sum(1) / denominator.clamp_min(1e-20),
                       torch.zeros_like(denominator)).clamp(-.999999, .999999)


def context_features(inputs: ForecastInputs):
    """23 pooled features; coarse statistics use input history, not extra forecasts."""
    q, h, scale, mean, sigma = (inputs.quantiles, inputs.history, inputs.scale,
                                inputs.location, inputs.raw_sigma)
    middle = mean.shape[1] // 2
    features = [
        (sigma / scale[:, None]).log().mean(1),
        sigma[:, middle:].log().mean(1) - sigma[:, :middle].log().mean(1),
        ((q[:, :, 8] - q[:, :, 0]).clamp_min(1e-8)
         / (q[:, :, 5] - q[:, :, 3]).clamp_min(1e-8)).log().mean(1),
        (h.std(1, correction=0) / scale).clamp_min(1e-8).log(),
    ]
    for lag in (1, 2, 4, 8, 12):
        a, b = h[:, :-lag], h[:, lag:]
        aa, bb = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
        features.extend([
            (aa * bb).sum(1) / (aa.square().sum(1) * bb.square().sum(1)).sqrt().clamp_min(1e-12),
            ((b - a).square().mean(1).sqrt() / scale).clamp_min(1e-8).log(),
            ((mean[:, lag:] - mean[:, :-lag]).square().mean(1).sqrt()
             / sigma.square().mean(1).sqrt()).clamp_min(1e-8).log(),
        ])
    for factor in (2, 3, 4, 6):
        coarse = h.reshape(len(h), -1, factor).mean(2)
        iqr = torch.quantile(coarse, .75, dim=1) - torch.quantile(coarse, .25, dim=1)
        features.append((iqr / scale).clamp_min(1e-8).log())
    return torch.stack(features, 1)


def profile_features(inputs: ForecastInputs):
    q, mean, sigma = inputs.quantiles, inputs.location, inputs.raw_sigma
    denominator = sigma.square().mean(1, keepdim=True).sqrt()
    delta = torch.cat([mean.new_zeros((len(mean), 1)), mean[:, 1:] - mean[:, :-1]], 1) / denominator
    time = torch.linspace(-1, 1, mean.shape[1], device=mean.device, dtype=mean.dtype)[None].expand_as(mean)
    features = torch.stack([
        sigma.log(), (mean - mean.mean(1, keepdim=True)) / denominator, delta, time,
        ((q[:, :, 8] - q[:, :, 0]).clamp_min(1e-8) / (q[:, :, 5] - q[:, :, 3]).clamp_min(1e-8)).log(),
    ], 2)
    return features - features.mean(1, keepdim=True)


def arm_features(features, arm, permutation=None):
    if arm != "full_profile":
        raise ValueError("Expected full_profile coordinates")
    return features


def soft_weights(logits):
    return torch.softmax(torch.stack([logits[..., 0], torch.zeros_like(logits[..., 0]),
                                      logits[..., 1]], -1), -1)


def log_weight_ratios(weights):
    # A zero shrinkage choice has exact zero outer weights. Keep local logits
    # finite at that boundary without changing any interior fitted weight.
    tiny = torch.finfo(weights.dtype).tiny
    return torch.stack([(weights[:, 0] / weights[:, 1].clamp_min(tiny)).clamp_min(tiny).log(),
                        (weights[:, 2] / weights[:, 1].clamp_min(tiny)).clamp_min(tiny).log()], 1)[:, None]


def global_pairs(weights, baseline):
    return weights[:, 1, None] * baseline + weights[:, 2, None]


def local_pairs(weights, baseline, pairs):
    a, b = weights[:, :, 1].sqrt(), weights[:, :, 2].sqrt()
    return a[:, pairs.left] * a[:, pairs.right] * baseline + b[:, pairs.left] * b[:, pairs.right]


@dataclass
class Gate:
    coefficient: Tensor
    center: Tensor
    sd: Tensor
    gamma: float
    penalty: float

    def transform(self, features, local=False):
        if local:
            x = (features / self.sd).clamp(-8, 8)
            return x - x.mean(1, keepdim=True)
        x = ((features - self.center) / self.sd).clamp(-8, 8)
        return torch.cat([x.new_ones((len(x), 1)), x], 1)


@dataclass
class FittedDependence:
    config: DependenceConfig
    temperature: float
    base_family: str
    base_matrix: Tensor
    global_gate: Gate
    local_gates: dict[str, Gate]
    trace: dict = field(default_factory=dict)

    def baseline(self, inputs, pairs=None):
        if self.base_family == "history_ar":
            rho = history_correlation(inputs.history)
            if pairs is not None:
                return rho[:, None].pow((pairs.right - pairs.left)[None])
            lag = torch.arange(self.config.horizon, device=rho.device)
            return rho[:, None, None].pow((lag[:, None] - lag[None, :]).abs()[None])
        if pairs is not None:
            return self.base_matrix[pairs.left, pairs.right][None].expand(len(inputs.scale), -1)
        return self.base_matrix[None].expand(len(inputs.scale), -1, -1)

    def weights(self, inputs, arm, permutation=None):
        features = context_features(inputs)
        gate = self.global_gate
        global_weight = gate.gamma * soft_weights(gate.transform(features) @ gate.coefficient + INITIAL_LOGIT)
        global_weight[:, 1] += 1 - gate.gamma
        if arm in ("global_context", "selected_simple"):
            return global_weight, None
        offset = log_weight_ratios(global_weight)
        local_gate = self.local_gates[arm]
        x = local_gate.transform(arm_features(profile_features(inputs), arm, permutation), local=True)
        return global_weight, soft_weights(offset + x @ local_gate.coefficient)

    @torch.no_grad()
    def predict(self, inputs: ForecastInputs, arm="full_profile", *, permutation=None, full_matrix=False):
        """Return a unit-diagonal correlation field. No target is accepted."""
        inputs.validate(self.config.horizon)
        if arm not in ("selected_simple", "global_context") and arm not in self.local_gates:
            raise ValueError(f"Arm was not fitted: {arm}")
        pairs = ChangePairs.create(self.config, inputs.quantiles.device)
        baseline = self.baseline(inputs, None if full_matrix else pairs)
        if arm == "selected_simple":
            return baseline
        global_weight, local_weight = self.weights(inputs, arm, permutation)
        if full_matrix:
            identity = torch.eye(self.config.horizon, device=baseline.device, dtype=baseline.dtype)
            pooled = (global_weight[:, 1, None, None] * baseline + global_weight[:, 2, None, None]
                      + global_weight[:, 0, None, None] * identity)
        else:
            pooled = global_pairs(global_weight, baseline)
        if arm == "global_context":
            return pooled
        if full_matrix:
            a, b = local_weight[:, :, 1].sqrt(), local_weight[:, :, 2].sqrt()
            local = a[:, :, None] * baseline * a[:, None, :] + b[:, :, None] * b[:, None, :]
            local += torch.diag_embed(local_weight[:, :, 0])
        else:
            local = local_pairs(local_weight, baseline, pairs)
        gamma = self.local_gates[arm].gamma
        return (1 - gamma) * pooled + gamma * local

    def marginal_parameters(self, inputs):
        """Gaussian working location/spread; exactly the same for every arm."""
        return inputs.location, inputs.raw_sigma * self.temperature

    def native_knots(self, inputs):
        """Sorted nine knots with the common temperature about the original median."""
        q = inputs.quantiles
        return q[:, :, 4, None] + self.temperature * (q.sort(-1).values - q[:, :, 4, None])

    def save(self, path: str | Path):
        """Portable arrays and JSON, without executable pickle or experiment imports."""
        arrays = {"base_matrix": self.base_matrix.detach().cpu().numpy()}
        gates = {"global_context": self.global_gate, **self.local_gates}
        scalars = {}
        for name, gate in gates.items():
            scalars[name] = {"gamma": gate.gamma, "penalty": gate.penalty}
            for key in ("coefficient", "center", "sd"):
                arrays[f"{name}__{key}"] = getattr(gate, key).detach().cpu().numpy()
        metadata = {"schema": 1, "config": asdict(self.config), "temperature": self.temperature,
                    "base_family": self.base_family, "gates": scalars, "trace": self.trace}
        arrays["metadata"] = np.asarray(json.dumps(metadata, allow_nan=False))
        with Path(path).open("xb") as stream:
            np.savez_compressed(stream, **arrays)

    @classmethod
    def load(cls, path: str | Path, device):
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata"]))
            if meta["schema"] != 1:
                raise ValueError("Unsupported fitted dependence schema")
            def tensor(key):
                return torch.as_tensor(archive[key], device=device, dtype=torch.float64)
            gates = {name: Gate(**{key: tensor(f"{name}__{key}") for key in ("coefficient", "center", "sd")},
                                **values) for name, values in meta["gates"].items()}
            return cls(DependenceConfig(**meta["config"]), meta["temperature"], meta["base_family"],
                       tensor("base_matrix"), gates.pop("global_context"), gates, meta["trace"])


def _fit_gate(x, training, calibration, ytrain, ycal, loss, base, proposal, config, *, local):
    normalizer = loss(base[training], training, ytrain).detach()
    trace, menu, candidates = {}, [], []
    penalties = config.local_penalties if local and config.local_penalties is not None else config.penalties
    for penalty in penalties:
        coef = x.new_zeros((x.shape[-1], 2), requires_grad=True)
        optimizer = torch.optim.LBFGS([coef], max_iter=config.local_iterations if local else config.global_iterations,
                                     history_size=20, line_search_fn="strong_wolfe", tolerance_grad=1e-7,
                                     tolerance_change=1e-10)
        calls = 0
        def closure():
            nonlocal calls
            optimizer.zero_grad()
            objective = loss(proposal(x[training] @ coef, training), training, ytrain) / normalizer
            objective = objective + penalty * coef.square().sum() / 2
            objective.backward()
            calls += 1
            return objective
        optimizer.step(closure)
        objective = closure()
        trace[str(penalty)] = {"objective": float(objective.detach()),
                               "max_gradient": float(coef.grad.abs().max()), "closure_calls": calls,
                               "coefficients": coef.detach().cpu().tolist()}
        with torch.no_grad():
            pair = proposal(x @ coef, None)
            candidates.append((coef.detach().clone(), pair))
            for gamma in config.gammas:
                score = float(loss((1 - gamma) * base[calibration] + gamma * pair[calibration], calibration, ycal))
                menu.append({"calibration_score": score, "gamma": gamma, "penalty": penalty,
                             "fit_index": len(candidates) - 1})
    choice = min(menu, key=lambda row: (row["calibration_score"], row["gamma"], -row["penalty"]))
    coef, pair = candidates[choice["fit_index"]]
    return coef, pair, {"selection": choice, "menu": menu, "optimization": trace}


def fit(inputs: ForecastInputs, training: Tensor, calibration: Tensor,
        ytrain: Tensor, ycal: Tensor, config=DependenceConfig(), *, permutation=None, progress=None):
    """Fit shared marginals and base dependence on disjoint training/validation rows."""
    inputs.validate(config.horizon)
    if config.horizon <= 12:
        raise ValueError("The fixed context features require a horizon longer than 12")
    device = inputs.quantiles.device
    for index, target in ((training, ytrain), (calibration, ycal)):
        if index.dtype != torch.int64 or index.ndim != 1 or not len(index) or index.device != device:
            raise ValueError("Partition indices must be nonempty int64 tensors on the input device")
        if len(index.unique()) != len(index) or int(index.min()) < 0 or int(index.max()) >= len(inputs.scale):
            raise ValueError("Invalid or duplicate partition indices")
        if target.shape != (len(index), config.horizon) or target.dtype != torch.float64 or target.device != device:
            raise ValueError("Targets must match their partition, input dtype and device")
        if not bool(torch.isfinite(target).all()):
            raise ValueError("Nonfinite target")
    if bool(torch.isin(training, calibration).any()):
        raise ValueError("Training and calibration partitions overlap")
    pairs = ChangePairs.create(config, device)
    mean, raw_sigma, scale = inputs.location, inputs.raw_sigma, inputs.scale
    temperature = shared_temperature(mean[calibration], raw_sigma[calibration], ycal, scale[calibration])
    sigma = raw_sigma * temperature
    def loss(pair, index, target):
        return (change_crps(mean[index], sigma[index], pair, target, scale[index], pairs) @ pairs.weights).mean()
    ones = mean.new_ones(len(mean))
    def ar(rho):
        return rho[:, None].pow((pairs.right - pairs.left)[None])
    candidates = {"independent": ar(ones * 0), "comonotonic": ar(ones),
                  "history_ar": ar(history_correlation(inputs.history))}
    trace = {"temperature": temperature, "base": {}, "global_context": {}, "local": {}}
    def cal_score(pair):
        return float(loss(pair[calibration], calibration, ycal))
    menu = [(cal_score(ar(ones * rho)), rho) for rho in config.rho_grid]
    score, rho = min(menu)
    candidates["global_ar"] = ar(ones * rho)
    trace["base"]["global_ar"] = {"rho": rho, "calibration_score": score, "menu": menu}
    lag = torch.arange(config.horizon, device=device)
    lag = (lag[:, None] - lag[None, :]).abs()
    identity = torch.eye(config.horizon, device=device, dtype=mean.dtype)
    matrices = {"independent": identity, "comonotonic": torch.ones_like(identity),
                "global_ar": mean.new_tensor(rho).pow(lag), "history_ar": identity}
    error = (ytrain - mean[training]) / sigma[training]
    for name, center in (("residual_corr", True), ("residual_second", False)):
        r = residual_correlation(error, center)
        menu = [(cal_score(((1 - a) * r[pairs.left, pairs.right])[None].expand(len(mean), -1)), a)
                for a in config.identity_weights]
        score, weight = min(menu)
        matrices[name] = (1 - weight) * r + weight * identity
        candidates[name] = ((1 - weight) * r[pairs.left, pairs.right])[None].expand(len(mean), -1)
        trace["base"][name] = {"identity_weight": weight, "calibration_score": score, "menu": menu}
    family = min(candidates, key=lambda name: cal_score(candidates[name]))
    base = candidates[family]
    trace["base"]["selected_simple"] = family
    trace["base"]["calibration_scores"] = {name: cal_score(pair) for name, pair in candidates.items()}
    features = context_features(inputs)
    center = features[training].mean(0)
    sd = features[training].std(0, correction=0).clamp_min(1e-6)
    gate = Gate(mean.new_empty(0), center, sd, 0., 0.)
    x = gate.transform(features)
    def global_proposal(logits, index):
        return global_pairs(soft_weights(logits + INITIAL_LOGIT), base if index is None else base[index])
    coef, _, trace["global_context"] = _fit_gate(x, training, calibration, ytrain, ycal, loss, base,
                                                global_proposal, config, local=False)
    choice = trace["global_context"]["selection"]
    gate.coefficient, gate.gamma, gate.penalty = coef, choice["gamma"], choice["penalty"]
    global_weight = gate.gamma * soft_weights(x @ coef + INITIAL_LOGIT)
    global_weight[:, 1] += 1 - gate.gamma
    offset = log_weight_ratios(global_weight)
    pooled = global_pairs(global_weight, base)
    if progress:
        progress("global_context", choice)
    profile = profile_features(inputs)
    local_gates = {}
    for arm in config.arms:
        features = arm_features(profile, arm, permutation)
        sd = features[training].std((0, 1), correction=0).clamp_min(1e-6)
        local_gate = Gate(mean.new_empty(0), mean.new_empty(0), sd, 0., 0.)
        x = local_gate.transform(features, local=True)
        def local_proposal(logits, index):
            off, r = (offset, base) if index is None else (offset[index], base[index])
            return local_pairs(soft_weights(off + logits), r, pairs)
        coef, _, trace["local"][arm] = _fit_gate(x, training, calibration, ytrain, ycal, loss, pooled,
                                               local_proposal, config, local=True)
        choice = trace["local"][arm]["selection"]
        local_gate.coefficient, local_gate.gamma, local_gate.penalty = coef, choice["gamma"], choice["penalty"]
        local_gates[arm] = local_gate
        if progress:
            progress(arm, choice)
    return FittedDependence(config, temperature, family, matrices[family], gate, local_gates, trace)
