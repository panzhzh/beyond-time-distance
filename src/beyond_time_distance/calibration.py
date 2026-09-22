"""Native change-CRPS calibration with common bounded scales and temperature."""
from dataclasses import asdict, dataclass, field, replace
import json
import math
from pathlib import Path

import torch

from .base import shared_temperature
from .interpolation import ContractionRisk
from .kernel import FitConfig, GeometryRefinement, fit_initializer, select_reference
from .scoring import calibrated_knots, native_change_scores, reconstructed_evidence


@dataclass(frozen=True)
class NativeConfig:
    fit: FitConfig = field(default_factory=FitConfig)
    contraction_grid: tuple = tuple((i / 16) ** 2 for i in range(17))
    eta_grid: tuple = (0., .00390625, .0078125, .015625, .03125, .0625, .125, .25, .5, 1.)
    train_count: int = 512
    validation_count: int = 1024
    confirmation_count: int = 4096
    seed: int = 2026091517
    batch_size: int = 4
    shortlist: int = 3

    def __post_init__(self):
        for count in (self.train_count, self.validation_count, self.confirmation_count):
            if count < 2 or count & (count - 1):
                raise ValueError("Quadrature sizes must be powers of two")
        for grid in (self.contraction_grid, self.eta_grid):
            if len(grid) < 3 or grid[0] != 0 or grid[-1] != 1 or any(a >= b for a, b in zip(grid, grid[1:])):
                raise ValueError("Grids must increase strictly from zero to one")
        if self.batch_size < 1 or self.shortlist < 1 or 0 not in self.fit.gammas:
            raise ValueError("Positive batch/shortlist sizes and the null gamma are required")

    def quadrature(self, count=None):
        return {"count": count or self.confirmation_count, "seed": self.seed, "batch_size": self.batch_size}

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["fit"] = FitConfig(**data.get("fit", {}))
        return cls(**data)


def bounded_floor(training_scale):
    positive = training_scale[training_scale > 1e-6]
    return max(1e-6, float(torch.quantile(positive, .05))) if len(positive) else 1e-6


def design(head, quantiles, pairs):
    z = head.coordinates(quantiles).detach()
    return {"left": z[:, pairs.left], "right": z[:, pairs.right], "factor": head.distance_factor}


def attenuation(theta, coordinates, gamma):
    distance = ((coordinates["left"] - coordinates["right"]).square()
                * coordinates["factor"] * torch.nn.functional.softplus(theta)).sum(-1)
    return gamma * distance / (1 + distance)


@torch.no_grad()
def make_surface(batch, pairs, temperature, floor, count, config, *, pilot=False):
    knots = calibrated_knots(batch.quantiles, temperature)
    reference = select_reference(batch.reference, pairs, len(knots), config.fit.horizon)
    grid = knots.new_tensor(config.contraction_grid)
    scale = batch.scale if pilot else batch.scale.clamp_min(floor)
    values = torch.stack([native_change_scores(knots, {"r": reference * (1 - t)},
                batch.target, scale, pairs, **config.quadrature(count))["r"] for t in grid], -1)
    if pilot:
        values = values * (batch.scale / batch.scale.clamp_min(floor))[:, None, None]
    return ContractionRisk.create(grid, values)


def fit_candidates(coordinates, surface, pairs, config):
    """Fresh six-fit menu; fitting labels are restricted to the training surface."""
    norm = (surface.values[..., 0] @ pairs.weights).mean().detach()
    if not bool(torch.isfinite(norm)) or float(norm) <= 0:
        raise ValueError("Reference training risk must be finite and positive")
    candidates, traces = [], []
    for gamma_fit in config.gamma_fit:
        for penalty in config.penalties:
            theta = coordinates["left"].new_full((4,), math.log(math.expm1(config.initial_strength)), requires_grad=True)
            optimizer = torch.optim.LBFGS([theta], max_iter=config.max_iter, history_size=20,
                line_search_fn="strong_wolfe", tolerance_grad=1e-7, tolerance_change=1e-10)
            calls = 0

            def closure():
                nonlocal calls
                optimizer.zero_grad()
                risk = (surface(attenuation(theta, coordinates, gamma_fit)) @ pairs.weights).mean()
                loss = risk / norm + penalty * torch.nn.functional.softplus(theta).square().sum() / 2
                loss.backward()
                calls += 1
                return loss

            optimizer.step(closure)
            objective = float(closure().detach())
            if not math.isfinite(objective) or not bool(torch.isfinite(theta).all()):
                raise ValueError("Nonfinite native fit")
            candidates.append(theta.detach().clone())
            traces.append({"gamma_fit": gamma_fit, "penalty": penalty, "calls": calls,
                "objective": objective, "max_gradient": float(theta.grad.abs().max()),
                "strengths": torch.nn.functional.softplus(theta).detach().cpu().tolist()})
    return candidates, traces


@torch.no_grad()
def select_candidate(candidates, metadata, coordinates, surface, validation, temperature,
                     floor, pairs, config, incumbent):
    menu = []
    for index, theta in enumerate(candidates):
        for gamma in config.fit.gammas:
            risk = float((surface(attenuation(theta, coordinates, gamma)) @ pairs.weights).mean())
            menu.append({"fit_index": index, "gamma": gamma, "screening_risk": risk,
                "penalty": metadata[index]["penalty"], "gamma_fit": metadata[index]["gamma_fit"], "origin": "refit"})
    key = lambda row: (row["screening_risk"], row["gamma"], -row["penalty"], row["gamma_fit"])
    chosen = [dict(r) for r in sorted(menu, key=key) if r["gamma"] > 0][:config.shortlist]
    chosen.append(dict(min((r for r in menu if r["gamma"] == 0), key=key)))
    chosen.append({"fit_index": len(candidates), "gamma": incumbent.gamma, "origin": "initializer",
        "penalty": 0., "gamma_fit": 0., "screening_risk": float((surface(
            attenuation(incumbent.coefficient, coordinates, incumbent.gamma)) @ pairs.weights).mean())})
    options = candidates + [incumbent.coefficient]
    reference = select_reference(validation.reference, pairs, len(validation.quantiles), config.fit.horizon)
    correlations = {str(i): reference * (1 - attenuation(options[r["fit_index"]], coordinates, r["gamma"]))
                    for i, r in enumerate(chosen)}
    scores, intervals = reconstructed_evidence(calibrated_knots(validation.quantiles, temperature),
        correlations, validation.target, validation.scale.clamp_min(floor), pairs, config.quadrature(), tuple(correlations))
    for i, row in enumerate(chosen):
        crps = float((scores[str(i)] @ pairs.weights).mean())
        row.update(validation_crps=crps, validation_risk=crps,
                   validation_is90=float((intervals[str(i) + "__interval_score90"] @ pairs.weights).mean()))
    winner = min(chosen, key=lambda r: (r["validation_risk"], r["origin"] != "initializer", r["gamma"], -r["penalty"]))
    return replace(incumbent, coefficient=options[winner["fit_index"]], gamma=winner["gamma"]), {
        "selection": dict(winner), "menu": menu, "confirmed": chosen}


@torch.no_grad()
def select_eta(head, batches, surfaces, temperature, floor, pairs, config):
    paths, contraction = {}, {}
    for role, batch in zip(("train", "validation"), batches):
        contraction[role] = attenuation(head.coefficient, design(head, batch.quantiles, pairs), head.gamma)
        paths[role] = [{"eta": eta, "risk": float((surfaces[role](
            eta + (1 - eta) * contraction[role]) @ pairs.weights).mean())} for eta in config.eta_grid]
    best = min(paths["train"], key=lambda r: (r["risk"], r["eta"]))
    eligible = {r["eta"] for r in paths["train"] if r["eta"] <= best["eta"] and r["risk"] <= paths["train"][0]["risk"]}
    chosen = [{"eta": 0.}] + sorted((r.copy() for r in paths["validation"] if r["eta"] in eligible and r["eta"] > 0),
        key=lambda r: (r["risk"], r["eta"]))[:config.shortlist]
    validation = batches[1]
    reference = select_reference(validation.reference, pairs, len(validation.quantiles), config.fit.horizon)
    correlations = {str(i): (1 - r["eta"]) * reference * (1 - contraction["validation"]) for i, r in enumerate(chosen)}
    scores = native_change_scores(calibrated_knots(validation.quantiles, temperature), correlations,
        validation.target, validation.scale.clamp_min(floor), pairs, **config.quadrature())
    for i, row in enumerate(chosen):
        row["validation_crps"] = float((scores[str(i)] @ pairs.weights).mean())
    winner = min(chosen, key=lambda r: (r["validation_crps"], r["eta"]))
    return winner["eta"], {"selection": winner, "training_choice": best, "eligible": sorted(eligible), "paths": paths, "confirmed": chosen}


def pair_distance(head, quantiles, pairs):
    d = design(head, quantiles, pairs)
    return ((d["left"] - d["right"]).square() * d["factor"] * head.strengths).sum(-1)


@dataclass
class CalibratedHeads:
    heads: dict[str, GeometryRefinement]
    temperature: float
    scale_floor: float
    eta: float
    config: NativeConfig
    grouping_head: GeometryRefinement
    group_cutoffs: torch.Tensor

    def __post_init__(self):
        if not math.isfinite(self.temperature) or self.temperature <= 0 or not 0 <= self.eta <= 1:
            raise ValueError("Invalid common temperature or shrinkage")
        if not math.isfinite(self.scale_floor) or self.scale_floor < 1e-6:
            raise ValueError("Invalid training scale floor")
        if self.group_cutoffs.shape != (len(self.config.fit.lags), 2):
            raise ValueError("Expected two training cutoffs per lag")

    def native_knots(self, quantiles):
        return calibrated_knots(quantiles, self.temperature)

    @torch.no_grad()
    def predict(self, quantiles, reference, pairs=None):
        """All heads use the same reference, diagonal shrinkage and marginal knots."""
        result = {name: (1 - self.eta) * head.predict(quantiles, reference, pairs)
                  for name, head in self.heads.items()}
        result["reference"] = (1 - self.eta) * select_reference(reference, pairs, len(quantiles), self.config.fit.horizon)
        if pairs is None:
            identity = torch.eye(self.config.fit.horizon, dtype=quantiles.dtype, device=quantiles.device)
            result = {name: value + self.eta * identity for name, value in result.items()}
        return result

    @torch.no_grad()
    def strata(self, quantiles, pairs):
        distance = pair_distance(self.grouping_head, quantiles, pairs)
        levels = torch.zeros_like(distance, dtype=torch.int8)
        for group, cutoffs in enumerate(self.group_cutoffs):
            use = pairs.groups == group
            levels[:, use] = (distance[:, use] > cutoffs[0]).to(torch.int8) + (distance[:, use] > cutoffs[1]).to(torch.int8)
        return levels

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        for name, head in self.heads.items():
            head.save(directory / f"{name}.json")
        self.grouping_head.save(directory / "grouping.json")
        metadata = {"schema": 2, "heads": list(self.heads), "temperature": self.temperature,
            "scale_floor": self.scale_floor, "eta": self.eta, "config": asdict(self.config),
            "group_cutoffs": self.group_cutoffs.detach().cpu().tolist()}
        (directory / "model.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")

    @classmethod
    def load(cls, directory, device="cpu"):
        directory = Path(directory)
        meta = json.loads((directory / "model.json").read_text())
        if meta["schema"] != 2:
            raise ValueError("Unsupported calibrated-model schema")
        return cls({name: GeometryRefinement.load(directory / f"{name}.json", device) for name in meta["heads"]},
            meta["temperature"], meta["scale_floor"], meta["eta"], NativeConfig.from_dict(meta["config"]),
            GeometryRefinement.load(directory / "grouping.json", device),
            torch.tensor(meta["group_cutoffs"], dtype=torch.float64, device=device))


def fit_calibration(train, validation, *, pilot_temperature, config=None, shifts=(), progress=None):
    """Fit from training and validation only; no test payload enters this API.

    Each alignment is refitted independently with identical budgets. Marginal
    temperature and isotropic shrinkage are common to every alignment.
    """
    config = config or NativeConfig()
    train.validate(config.fit); validation.validate(config.fit)
    if train.quantiles.device != validation.quantiles.device:
        raise ValueError("Training and validation must share a device")
    shifts = tuple(shifts)
    if len(set(shifts)) != len(shifts) or any(s < 1 or s >= config.fit.horizon for s in shifts):
        raise ValueError("Supply distinct nonzero shifts below the horizon")
    pairs = config.fit.pairs(train.quantiles.device)
    floor = bounded_floor(train.scale)
    q = validation.quantiles
    temperature = shared_temperature(q[:, :, 4], ((q[:, :, 8] - q[:, :, 0]) / (2 * 1.2815515655446004)).clamp_min(1e-8),
                                     validation.target, validation.scale.clamp_min(floor))
    surfaces = {}
    for stage, temp in (("pilot", pilot_temperature), ("native", temperature)):
        surfaces[stage] = {role: make_surface(batch, pairs, temp, floor, count, config, pilot=stage == "pilot")
            for role, batch, count in (("train", train, config.train_count), ("validation", validation, config.validation_count))}
        if progress: progress({"stage": stage, "event": "surfaces_ready"})
    specifications = [("geometry", "geometry", 0), ("time", "time", 0)] + [(f"shift_{s:02d}", "geometry", s) for s in shifts]
    heads, traces = {}, {}
    for name, feature, shift in specifications:
        head, initializer_trace = fit_initializer(train, validation, temperature=pilot_temperature,
            config=config.fit, feature=feature, shift=shift)
        if name == "geometry":
            grouping = head
            distance = pair_distance(head, train.quantiles, pairs)
            cutoffs = torch.stack([torch.quantile(distance[:, pairs.groups == g].flatten(), distance.new_tensor([1/3, 2/3]))
                                  for g in range(len(config.fit.lags))])
        coordinates = {role: design(head, batch.quantiles, pairs) for role, batch in (("train", train), ("validation", validation))}
        traces[name] = {"initializer": initializer_trace}
        for stage, temp in (("pilot", pilot_temperature), ("native", temperature)):
            candidates, optimization = fit_candidates(coordinates["train"], surfaces[stage]["train"], pairs, config.fit)
            head, trace = select_candidate(candidates, optimization, coordinates["validation"], surfaces[stage]["validation"],
                validation, temp, floor, pairs, config, head)
            traces[name][stage] = {"optimization": optimization, **trace}
        heads[name] = head
        if progress: progress({"head": name, **traces[name]["native"]["selection"]})
    eta, eta_trace = select_eta(heads["time"], (train, validation), surfaces["native"], temperature, floor, pairs, config)
    model = CalibratedHeads(heads, temperature, floor, eta, config, grouping, cutoffs)
    return model, {"pilot_temperature": pilot_temperature, "temperature": temperature,
                   "scale_floor": floor, "eta": eta_trace, "heads": traces}
