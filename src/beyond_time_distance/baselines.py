"""Native-objective full-profile and five-seed conditional-AR comparisons."""
from copy import deepcopy

import torch

from .base import ForecastInputs, INITIAL_LOGIT, context_features, local_pairs, log_weight_ratios, soft_weights
from .gates import fit_conditional_ar, fit_gates, local_features
from .interpolation import ContractionRisk
from .reference import DEFAULT_GATE_CONFIG, concatenate
from .scoring import native_change_scores


CORRELATION_GRID = (-1., -.999, -.99, -.95, -.9, -.75, -.5, -.25, 0., .1, .2, .3,
                    .4, .5, .6, .7, .8, .85, .9, .93, .95, .97, .98, .99, .995, .999, 1.)
AR_SEEDS = (2026091903, 2026091904, 2026091905, 2026091906, 2026091907)
AR_INITIALIZER = {"epochs": 1000, "checkpoint_epochs": [10, 30, 100, 200, 300, 500, 1000],
                  "learning_rate": .001, "batch_size": 256, "initial_rho": .9, "initial_beta": .01}


def subset(inputs, index):
    return ForecastInputs(inputs.history[index], inputs.quantiles[index], inputs.scale[index])


def fit_baselines(train, validation, ytrain, yvalidation, reference, calibration, *,
                  seeds=AR_SEEDS, ar_initializer=None, epochs=200, checkpoints=(10, 30, 100, 200),
                  learning_rate=.0001, batch_size=256, gate_config=None,
                  gate_penalties=(1e-7, 1e-6, 1e-5, 1e-4, 1e-3), gate_steps=(.25, .5, 1.), max_iter=120):
    """Adapt each comparator independently on the common final marginal knots.

    Each AR seed selects its own validation checkpoint. Evaluation averages
    seed scores, retaining all five fitted models.
    """
    inputs = concatenate(train, validation)
    device = inputs.scale.device
    tr = torch.arange(len(train.scale), device=device)
    va = torch.arange(len(train.scale), len(inputs.scale), device=device)
    pairs = calibration.config.fit.pairs(device)
    bounded = inputs.scale.clamp_min(calibration.scale_floor)
    knots = calibration.native_knots(inputs.quantiles)
    grid = inputs.scale.new_tensor(CORRELATION_GRID)
    values = torch.stack([native_change_scores(knots[tr], {"r": rho.expand(len(tr), len(pairs.left))},
        ytrain, bounded[tr], pairs, **calibration.config.quadrature(calibration.config.train_count))["r"] for rho in grid], -1)
    surface = ContractionRisk.create(grid, values)
    norm = (surface(torch.zeros_like(values[..., 0])) @ pairs.weights).mean().detach()

    def training_risk(corr, index=None):
        risk = surface if index is None else ContractionRisk(surface.grid, surface.values[index], surface.slopes[index])
        return (risk(corr) @ pairs.weights).mean() / norm

    def exact(correlations):
        scores = native_change_scores(knots[va], correlations, yvalidation, bounded[va], pairs, **calibration.config.quadrature())
        return {key: float((value @ pairs.weights).mean()) for key, value in scores.items()}

    models, traces = {}, {}
    anchor = reference.marginal
    for seed in seeds:
        cfg = dict(ar_initializer or AR_INITIALIZER, seeds=[seed])
        model, initial_trace, _ = fit_conditional_ar(anchor, inputs, tr, va, ytrain, yvalidation,
                                                    inputs.scale, cfg, pairs)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        generator = torch.Generator(device=device).manual_seed(seed)
        states, correlations, epoch_numbers = [], {}, []

        def checkpoint(epoch):
            with torch.no_grad():
                states.append(deepcopy(model.state_dict()))
                correlations[str(len(states) - 1)] = model.predict(validation, pairs)
                epoch_numbers.append(epoch)

        checkpoint(0)
        for epoch in range(1, epochs + 1):
            for ix in torch.randperm(len(tr), device=device, generator=generator).split(batch_size):
                optimizer.zero_grad()
                training_risk(model.predict(subset(train, ix), pairs), ix).backward()
                optimizer.step()
            if epoch in checkpoints: checkpoint(epoch)
        scores = exact(correlations)
        best = min(range(len(states)), key=lambda i: (scores[str(i)], epoch_numbers[i]))
        model.load_state_dict(states[best])
        models[f"ar_{seed}"] = model
        traces[f"ar_{seed}"] = {"initializer": initial_trace, "selected_epoch": epoch_numbers[best],
            "validation_scores": [{"epoch": epoch, "crps": scores[str(i)]} for i, epoch in enumerate(epoch_numbers)]}

    head = fit_gates(anchor, inputs, tr, va, ytrain, yvalidation, inputs.scale,
                     gate_config or DEFAULT_GATE_CONFIG, ["full_profile"], {})["full_profile"]
    with torch.no_grad():
        xg = head.global_gate.transform(context_features(inputs))
        xl = head.local_gate.transform(local_features(inputs, "full_profile"), local=True)
        base = anchor.baseline(inputs, pairs)
        initial = head.predict(anchor, validation, pairs)
    oldg, oldl = head.global_gate.coefficient, head.local_gate.coefficient

    def predict(index, dg, dl, step=1.):
        weights = soft_weights(xg[index] @ (oldg + step * dg) + INITIAL_LOGIT)
        local = soft_weights(log_weight_ratios(weights) + xl[index] @ (oldl + step * dl))
        return local_pairs(local, base[index], pairs)

    options, correlations = [], {"initial": initial}
    menu = {"initial": {"penalty": None, "step": 0.}}
    for penalty in gate_penalties:
        dg, dl = torch.zeros_like(oldg, requires_grad=True), torch.zeros_like(oldl, requires_grad=True)
        optimizer = torch.optim.LBFGS([dg, dl], max_iter=max_iter, history_size=20,
            line_search_fn="strong_wolfe", tolerance_grad=1e-7, tolerance_change=1e-10)

        def closure():
            optimizer.zero_grad()
            loss = training_risk(predict(tr, dg, dl)) + penalty * (dg.square().sum() + dl.square().sum()) / 2
            loss.backward()
            return loss

        optimizer.step(closure)
        for step in gate_steps:
            key = str(len(options))
            options.append((dg.detach().clone(), dl.detach().clone(), step))
            with torch.no_grad(): correlations[key] = predict(va, dg, dl, step)
            menu[key] = {"penalty": penalty, "step": step}
    scores = exact(correlations)
    best = min(scores, key=lambda key: (scores[key], key != "initial", menu[key]["step"]))
    if best != "initial":
        dg, dl, step = options[int(best)]
        head.global_gate.coefficient = oldg + step * dg
        head.local_gate.coefficient = oldl + step * dl
        head.global_gate.gamma = 1.; head.local_gate.gamma = 1.
    models["full_profile"] = head
    traces["full_profile"] = {"selection": best, "menu": {key: {**menu[key], "validation_crps": value} for key, value in scores.items()}}
    return models, traces


def save_baselines(models, traces, directory):
    import json
    from pathlib import Path
    import numpy as np
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for name, model in models.items():
        if name == 'full_profile':
            model.save(directory / f'{name}.npz')
        else:
            with (directory / f'{name}.npz').open('xb') as stream:
                np.savez_compressed(stream, **{k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})
    (directory / 'models.json').write_text(json.dumps({'schema': 1, 'models': list(models), 'traces': traces}, indent=2, allow_nan=False) + '\n')


def load_baselines(directory, device):
    import json
    from pathlib import Path
    import numpy as np
    from .gates import ConditionalAR, ControlGate
    directory = Path(directory)
    metadata = json.loads((directory / 'models.json').read_text())
    if metadata['schema'] != 1: raise ValueError('Unsupported baseline schema')
    models = {}
    for name in metadata['models']:
        if name == 'full_profile':
            models[name] = ControlGate.load(directory / f'{name}.npz', device)
        else:
            model = ConditionalAR(device)
            with np.load(directory / f'{name}.npz', allow_pickle=False) as arrays:
                model.load_state_dict({k: torch.as_tensor(arrays[k], device=device) for k in arrays.files})
            models[name] = model
    return models
