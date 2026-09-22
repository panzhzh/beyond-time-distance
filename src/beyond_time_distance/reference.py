"""Fit and reload the shared temporal reference used by all geometry controls."""
from dataclasses import dataclass
from pathlib import Path

import torch

from .base import DependenceConfig, FittedDependence, ForecastInputs, fit as fit_base
from .gates import ControlGate, fit_gates
from .kernel import CalibrationBatch, FitConfig, GeometryRefinement, fit_initializer


DEFAULT_GATE_CONFIG = {
    "penalties": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
    "gammas": [0., .03125, .0625, .125, .25, .5, .75, 1.],
    "global_iterations": 80,
    "local_iterations": 120,
    "joint": {"penalties": [1e-4, 1e-3, 1e-2],
              "step_sizes": [0., .03125, .0625, .125, .25, .5, .75, 1.], "max_iter": 120},
}


def concatenate(train, validation):
    return ForecastInputs(torch.cat([train.history, validation.history]),
                          torch.cat([train.quantiles, validation.quantiles]),
                          torch.cat([train.scale, validation.scale]))


@dataclass
class TemporalReference:
    marginal: FittedDependence
    gate: ControlGate
    correction: GeometryRefinement

    @property
    def temperature(self):
        return self.marginal.temperature

    def native_knots(self, inputs):
        return self.marginal.native_knots(inputs)

    @torch.no_grad()
    def predict(self, inputs, pairs=None):
        inputs.validate(self.marginal.config.horizon)
        return self.correction.predict(inputs.quantiles, self.gate.predict(self.marginal, inputs, pairs), pairs)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.marginal.save(directory / "marginals.npz")
        self.gate.save(directory / "time_gate.npz")
        self.correction.save(directory / "time_kernel.json")

    @classmethod
    def load(cls, directory, device="cpu"):
        directory = Path(directory)
        return cls(FittedDependence.load(directory / "marginals.npz", device),
                   ControlGate.load(directory / "time_gate.npz", device),
                   GeometryRefinement.load(directory / "time_kernel.json", device))


def fit_temporal_reference(train, validation, ytrain, yvalidation, *, config=None, gate_config=None):
    """Fit common marginals, a joint time gate, and a selected time-distance kernel."""
    config = config or FitConfig()
    train.validate(config.horizon); validation.validate(config.horizon)
    inputs = concatenate(train, validation)
    tr = torch.arange(len(train.scale), device=inputs.scale.device)
    va = torch.arange(len(train.scale), len(inputs.scale), device=inputs.scale.device)
    base_config = DependenceConfig(horizon=config.horizon, lags=tuple(config.lags))
    marginal = fit_base(inputs, tr, va, ytrain, yvalidation, base_config)
    gate = fit_gates(marginal, inputs, tr, va, ytrain, yvalidation, inputs.scale,
                     gate_config or DEFAULT_GATE_CONFIG, ["time5"], {})["time5"]
    pairs = config.pairs(inputs.scale.device)
    with torch.no_grad():
        ref = gate.predict(marginal, inputs, pairs)
    batches = [CalibrationBatch(part.quantiles, ref[ix], target, part.scale)
               for part, ix, target in ((train, tr, ytrain), (validation, va, yvalidation))]
    options = []
    for response, operator in (("gaussian", "schur"), ("laplace", "mixture")):
        head, trace = fit_initializer(*batches, temperature=marginal.temperature, config=config,
                                     feature="time", response=response, operator=operator)
        options.append((trace["selection"]["calibration_risk"], operator != "schur", head, trace))
    chosen = min(options, key=lambda row: row[:2])
    trace = {"temperature": marginal.temperature, "selected_response": chosen[2].response,
             "selected_operator": chosen[2].operator,
             "time_kernels": {head.response: trace for _, _, head, trace in options}}
    return TemporalReference(marginal, gate, chosen[2]), trace
