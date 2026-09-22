"""Temporal gates and conditional-AR baselines with shared marginal forecasts."""
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .base import (
    ChangePairs, DependenceConfig, Gate, INITIAL_LOGIT, _fit_gate, arm_features,
    change_crps, context_features, global_pairs, local_pairs, log_weight_ratios,
    profile_features, soft_weights,
)


def _synchronize(inputs):
    if inputs.quantiles.is_cuda:
        torch.cuda.synchronize(inputs.quantiles.device)


def time_basis(horizon, reference):
    t = torch.linspace(-1, 1, horizon, device=reference.device, dtype=reference.dtype)
    q, r = torch.linalg.qr(torch.stack([t**k for k in range(6)], -1))
    q = q * torch.where(r.diag() >= 0, 1., -1.)
    return q[:, 1:]


def local_features(inputs, arm, permutation=None):
    if arm == 'time5':
        return time_basis(inputs.quantiles.shape[1], inputs.scale)[None].expand(len(inputs.scale), -1, -1)
    return arm_features(profile_features(inputs), arm, permutation)


def ar_correlation(rho, beta, pairs=None, horizon=48):
    if pairs is not None:
        return (1 - beta[:, None]) * rho[:, None].pow((pairs.right - pairs.left)[None])
    distance = torch.arange(horizon, device=rho.device)
    distance = (distance[:, None] - distance[None]).abs()
    return ((1 - beta[:, None, None]) * rho[:, None, None].pow(distance[None])
            + beta[:, None, None] * torch.eye(horizon, device=rho.device, dtype=rho.dtype))


class ConditionalAR(torch.nn.Module):
    """30-64-32-2 conditional AR copula, task-adapted from https://arxiv.org/abs/2510.02224."""

    def __init__(self, device):
        super().__init__()
        self.network = torch.nn.Sequential(torch.nn.Linear(30, 64), torch.nn.ReLU(),
                                           torch.nn.Linear(64, 32), torch.nn.ReLU(),
                                           torch.nn.Linear(32, 2)).to(device=device, dtype=torch.float64)

    def forward(self, history):
        x = history[:, -30:]
        x = x / x.abs().amax(1, keepdim=True).clamp_min(1e-6)
        output = self.network(x)
        return output[:, 0].tanh(), output[:, 1].sigmoid()

    def predict(self, inputs, pairs=None):
        return ar_correlation(*self(inputs.history), pairs, inputs.quantiles.shape[1])


@dataclass
class ControlGate:
    arm: str
    global_gate: Gate
    local_gate: Gate | None
    trace: dict

    def weights(self, inputs, permutation=None):
        g = self.global_gate
        weights = g.gamma * soft_weights(g.transform(context_features(inputs)) @ g.coefficient + INITIAL_LOGIT)
        weights[:, 1] += 1 - g.gamma
        if self.local_gate is None:
            return weights, None
        x = self.local_gate.transform(local_features(inputs, self.arm, permutation), local=True)
        return weights, soft_weights(log_weight_ratios(weights) + x @ self.local_gate.coefficient)

    def predict(self, anchor, inputs, pairs=None, permutation=None):
        w, local = self.weights(inputs, permutation)
        base = anchor.baseline(inputs, pairs)
        if pairs is None:
            eye = torch.eye(anchor.config.horizon, device=base.device, dtype=base.dtype)
            pooled = w[:, 1, None, None] * base + w[:, 2, None, None] + w[:, 0, None, None] * eye
            if local is None:
                return pooled
            a, b = local[:, :, 1].sqrt(), local[:, :, 2].sqrt()
            proposal = a[:, :, None] * base * a[:, None, :] + b[:, :, None] * b[:, None, :]
            proposal = proposal + torch.diag_embed(local[:, :, 0])
        else:
            pooled = global_pairs(w, base)
            if local is None:
                return pooled
            proposal = local_pairs(local, base, pairs)
        return (1 - self.local_gate.gamma) * pooled + self.local_gate.gamma * proposal

    def save(self, path):
        arrays, metadata = {}, {'arm': self.arm, 'trace': self.trace, 'gates': {}}
        for name, gate in [('global', self.global_gate), ('local', self.local_gate)]:
            if gate is None:
                continue
            metadata['gates'][name] = {'gamma': gate.gamma, 'penalty': gate.penalty}
            for key in ('coefficient', 'center', 'sd'):
                arrays[f'{name}__{key}'] = getattr(gate, key).detach().cpu().numpy()
        with Path(path).open('xb') as stream:
            np.savez_compressed(stream, metadata=np.asarray(json.dumps(metadata, allow_nan=False)), **arrays)

    @classmethod
    def load(cls, path, device):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data['metadata']))
            gates = {name: Gate(**{k: torch.as_tensor(data[f'{name}__{k}'], device=device)
                                   for k in ('coefficient', 'center', 'sd')}, **values)
                     for name, values in meta['gates'].items()}
        return cls(meta['arm'], gates['global'], gates.get('local'), meta['trace'])


def joint_refine(head, xglobal, xlocal, base, pairs, training, calibration, ytrain, ycal, loss, config):
    """Refine global and local coefficients jointly, then select a validation step."""
    oldg = head.global_gate.coefficient
    oldl = head.local_gate.coefficient if head.local_gate else oldg.new_empty((0, 2))
    def predict(index, dg, dl, step=1.):
        w = head.global_gate.gamma * soft_weights(xglobal[index] @ (oldg + step * dg) + INITIAL_LOGIT)
        w[:, 1] += 1 - head.global_gate.gamma
        pooled = global_pairs(w, base[index])
        if head.local_gate is None:
            return pooled
        local = soft_weights(log_weight_ratios(w) + xlocal[index] @ (oldl + step * dl))
        return (1 - head.local_gate.gamma) * pooled + head.local_gate.gamma * local_pairs(local, base[index], pairs)
    zg, zl = torch.zeros_like(oldg), torch.zeros_like(oldl)
    norm = loss(predict(training, zg, zl), training, ytrain).detach()
    menu, candidates, optimization = [], [], []
    for penalty in config['penalties']:
        dg, dl = torch.zeros_like(oldg, requires_grad=True), torch.zeros_like(oldl, requires_grad=True)
        params = [dg, dl] if head.local_gate else [dg]
        opt = torch.optim.LBFGS(params, max_iter=config['max_iter'], history_size=20,
                                line_search_fn='strong_wolfe', tolerance_grad=1e-7, tolerance_change=1e-10)
        calls = 0
        def closure():
            nonlocal calls
            opt.zero_grad()
            value = loss(predict(training, dg, dl), training, ytrain) / norm
            value = value + penalty * (dg.square().sum() + dl.square().sum()) / 2
            value.backward()
            calls += 1
            return value
        opt.step(closure)
        value = closure()
        optimization.append({'penalty': penalty, 'calls': calls, 'objective': float(value.detach()),
                             'max_gradient': max(float(p.grad.abs().max()) for p in params),
                             'global_delta': dg.detach().cpu().tolist(), 'local_delta': dl.detach().cpu().tolist()})
        with torch.no_grad():
            candidates.append((dg.detach().clone(), dl.detach().clone()))
            for step in config['step_sizes']:
                menu.append({'penalty': penalty, 'step': step, 'fit_index': len(candidates)-1,
                             'calibration_score': float(loss(predict(calibration, dg, dl, step), calibration, ycal))})
    chosen = min(menu, key=lambda r: (r['calibration_score'], r['step'], -r['penalty']))
    dg, dl = candidates[chosen['fit_index']]
    head.global_gate.coefficient = oldg + chosen['step'] * dg
    if head.local_gate:
        head.local_gate.coefficient = oldl + chosen['step'] * dl
    head.trace['joint'] = {'selection': chosen, 'menu': menu, 'optimization': optimization}
    return head


def fit_gates(anchor, inputs, training, calibration, ytrain, ycal, denominator, protocol, arms, permutations,
              progress=None):
    if bool(torch.isin(training, calibration).any()):
        raise ValueError('Training and calibration overlap')
    cfg = DependenceConfig(horizon=anchor.config.horizon, lags=anchor.config.lags,
                           penalties=tuple(protocol['penalties']), local_penalties=tuple(protocol['penalties']),
                           gammas=tuple(protocol['gammas']), global_iterations=protocol['global_iterations'],
                           local_iterations=protocol['local_iterations'])
    pairs = ChangePairs.create(cfg, inputs.scale.device)
    mean, sigma = anchor.marginal_parameters(inputs)
    base = anchor.baseline(inputs, pairs)
    def loss(corr, index, target):
        return (change_crps(mean[index], sigma[index], corr, target, denominator[index], pairs) @ pairs.weights).mean()
    features = context_features(inputs)
    global_gate = Gate(mean.new_empty(0), features[training].mean(0),
                       features[training].std(0, correction=0).clamp_min(1e-6), 0., 0.)
    xglobal = global_gate.transform(features)
    def proposal(logits, index):
        return global_pairs(soft_weights(logits + INITIAL_LOGIT), base if index is None else base[index])
    _synchronize(inputs)
    start = time.perf_counter()
    coef, _, trace = _fit_gate(xglobal, training, calibration, ytrain, ycal, loss, base, proposal, cfg, local=False)
    choice = trace['selection']
    global_gate.coefficient, global_gate.gamma, global_gate.penalty = coef, choice['gamma'], choice['penalty']
    w = global_gate.gamma * soft_weights(xglobal @ coef + INITIAL_LOGIT)
    w[:, 1] += 1 - global_gate.gamma
    offset, pooled = log_weight_ratios(w), global_pairs(w, base)
    _synchronize(inputs)
    initial_seconds = time.perf_counter() - start
    heads = {}
    for arm in arms:
        _synchronize(inputs)
        start = time.perf_counter()
        local_gate, xlocal, local_trace = None, None, None
        if arm != 'pooled':
            features = local_features(inputs, arm, permutations.get(arm))
            local_gate = Gate(mean.new_empty(0), mean.new_empty(0),
                              features[training].std((0, 1), correction=0).clamp_min(1e-6), 0., 0.)
            xlocal = local_gate.transform(features, local=True)
            def proposal(logits, index):
                off, r = (offset, base) if index is None else (offset[index], base[index])
                return local_pairs(soft_weights(off + logits), r, pairs)
            coef, _, local_trace = _fit_gate(xlocal, training, calibration, ytrain, ycal, loss, pooled, proposal, cfg, local=True)
            choice = local_trace['selection']
            local_gate.coefficient, local_gate.gamma, local_gate.penalty = coef, choice['gamma'], choice['penalty']
        head = ControlGate(arm, deepcopy(global_gate), local_gate,
                           {'global': trace, 'local': local_trace, 'initial_pooled_fit_seconds': initial_seconds})
        head = joint_refine(head, xglobal, xlocal, base, pairs, training, calibration, ytrain, ycal, loss, protocol['joint'])
        _synchronize(inputs)
        head.trace['additional_fit_seconds'] = time.perf_counter() - start
        head.trace['total_fit_seconds_including_shared_pooled'] = initial_seconds + head.trace['additional_fit_seconds']
        head.trace['parameter_count'] = 48 + (local_gate.coefficient.numel() if local_gate else 0)
        heads[arm] = head
        if progress:
            progress(arm, head.trace)
    return heads


def fit_conditional_ar(anchor, inputs, training, calibration, ytrain, ycal, denominator, config, pairs):
    mean, sigma = anchor.marginal_parameters(inputs)
    normalizer = (change_crps(mean[training], sigma[training], anchor.baseline(inputs, pairs)[training],
                              ytrain, denominator[training], pairs) @ pairs.weights).mean().detach()
    def loss(model, idx, y):
        corr = ar_correlation(*model(inputs.history[idx]), pairs)
        return (change_crps(mean[idx], sigma[idx], corr, y, denominator[idx], pairs) @ pairs.weights).mean()
    menu, candidates, training_trace = [], [], []
    _synchronize(inputs)
    start = time.perf_counter()
    for seed in config['seeds']:
        torch.manual_seed(seed)
        model = ConditionalAR(inputs.scale.device)
        with torch.no_grad():
            model.network[-1].weight.mul_(.01)
            model.network[-1].bias.copy_(mean.new_tensor([math.atanh(config['initial_rho']),
                                       math.log(config['initial_beta'] / (1-config['initial_beta']))]))
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        rng = torch.Generator(device=inputs.scale.device).manual_seed(seed)
        calls = 0
        for epoch in range(1, config['epochs'] + 1):
            order = torch.randperm(len(training), generator=rng, device=training.device)
            for sub in order.split(config['batch_size']):
                optimizer.zero_grad()
                objective = loss(model, training[sub], ytrain[sub]) / normalizer
                objective.backward()
                optimizer.step()
                calls += 1
            if epoch in config['checkpoint_epochs']:
                with torch.no_grad():
                    score = float(loss(model, calibration, ycal))
                    train_score = float(loss(model, training, ytrain))
                menu.append({'seed': seed, 'epoch': epoch, 'calibration_score': score,
                             'training_score': train_score, 'fit_index': len(candidates), 'calls': calls})
                candidates.append(deepcopy(model.state_dict()))
        training_trace.append({'seed': seed, 'objective_calls': calls, 'epochs': config['epochs']})
    chosen = min(menu, key=lambda row: (row['calibration_score'], row['epoch'], row['seed']))
    model.load_state_dict(candidates[chosen['fit_index']])
    _synchronize(inputs)
    trace = {'selection': chosen, 'menu': menu, 'optimization': training_trace,
             'total_fit_seconds': time.perf_counter() - start,
             'parameter_count': sum(p.numel() for p in model.parameters())}
    return model, trace, candidates
