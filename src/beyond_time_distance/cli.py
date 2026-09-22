"""Fit and evaluate the paper protocol from cached marginal forecasts."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .calibration import CalibratedHeads, NativeConfig, fit_calibration
from .io import load_forecasts, write_json
from .kernel import CalibrationBatch
from .reference import TemporalReference, fit_temporal_reference
from .scoring import reconstructed_evidence


def device_arg(value):
    device = torch.device(value)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('Fitting and quadrature require an available CUDA device')
    torch.cuda.set_device(device)
    return device


def fit_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', required=True)
    parser.add_argument('--validation', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--reference', help='Reuse a fitted temporal-reference directory')
    parser.add_argument('--config', help='Native fitting configuration JSON')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--shifts', choices=['all', 'none'], default='all')
    args = parser.parse_args()
    if Path(args.train).resolve() == Path(args.validation).resolve():
        parser.error('Training and validation must be separate partitions')
    out = Path(args.output)
    if out.exists(): parser.error('Choose a new output directory')
    config = NativeConfig.from_dict(json.loads(Path(args.config).read_text())) if args.config else NativeConfig()
    device = device_arg(args.device)
    train, yt = load_forecasts(args.train, device)
    validation, yv = load_forecasts(args.validation, device)
    if args.reference:
        reference = TemporalReference.load(args.reference, device); reference_trace = None
    else:
        reference, reference_trace = fit_temporal_reference(train, validation, yt, yv, config=config.fit)
    pairs = config.fit.pairs(device)
    batches = [CalibrationBatch(x.quantiles, reference.predict(x, pairs), y, x.scale)
               for x, y in ((train, yt), (validation, yv))]
    model, trace = fit_calibration(*batches, pilot_temperature=reference.temperature, config=config,
        shifts=range(1, config.fit.horizon) if args.shifts == 'all' else (),
        progress=lambda event: print(json.dumps(event), flush=True))
    model.save(out)
    reference.save(out / 'reference')
    write_json(out / 'fit.json', {'reference': reference_trace, **trace})
    print(json.dumps({'event': 'saved', 'temperature': model.temperature, 'eta': model.eta, 'heads': len(model.heads)}))


def summarize_scores(scores, intervals, scale, floor, strata, pairs):
    """Equal lag means; selected pairs are pooled within each lag and stratum."""
    rows = []
    groups = int(pairs.groups.max()) + 1
    for name, crps in scores.items():
        for stratum, level in (('all', None), ('low', 0), ('middle', 1), ('high', 2)):
            def average(value):
                pieces = []
                for lag in range(groups):
                    use = pairs.groups == lag
                    mask = torch.ones_like(strata[:, use], dtype=torch.bool) if level is None else strata[:, use] == level
                    if not bool(mask.any()):
                        return None
                    pieces.append(value[:, use][mask].mean())
                return float(torch.stack(pieces).mean())
            row = {'method': name, 'stratum': stratum, 'bounded_crps': average(crps / scale.clamp_min(floor)[:, None]),
                   'physical_crps': average(crps), 'original_crps': average(crps / scale[:, None])}
            if name + '__interval_score90' in intervals:
                interval = intervals[name + '__interval_score90']
                row.update(bounded_is90=average(interval / scale.clamp_min(floor)[:, None]), physical_is90=average(interval),
                           coverage90=average(intervals[name + '__coverage90']), width90=average(intervals[name + '__width90']))
            rows.append(row)
    return rows


def evaluate_main():
    parser = argparse.ArgumentParser(description='Evaluate frozen heads with change CRPS and 90% interval diagnostics.')
    parser.add_argument('--model', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--baselines', help='Directory from examples/fit_baselines.py')
    parser.add_argument('--pair-scores', help='Optional NPZ output of physical-unit scores, strata and scales')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if Path(args.output).exists() or (args.pair_scores and Path(args.pair_scores).exists()):
        parser.error('Choose new output files')
    device = device_arg(args.device)
    model = CalibratedHeads.load(args.model, device)
    reference = TemporalReference.load(Path(args.model) / 'reference', device)
    inputs, targets = load_forecasts(args.test, device)
    pairs = model.config.fit.pairs(device)
    correlations = model.predict(inputs.quantiles, reference.predict(inputs, pairs), pairs)
    baseline_names = []
    if args.baselines:
        from .baselines import load_baselines
        for name, baseline in load_baselines(args.baselines, device).items():
            with torch.no_grad():
                correlations[name] = baseline.predict(reference.marginal, inputs, pairs) if name == 'full_profile' else baseline.predict(inputs, pairs)
            if name.startswith('ar_'): baseline_names.append(name)
    diagnostics = ('geometry', 'time', 'reference')
    scores, intervals = reconstructed_evidence(model.native_knots(inputs.quantiles), correlations, targets,
        torch.ones_like(inputs.scale), pairs, model.config.quadrature(), diagnostics)
    shift_names = [f'shift_{s:02d}' for s in range(1, model.config.fit.horizon)]
    if all(name in scores for name in shift_names):
        scores['mean_shifts'] = torch.stack([scores[name] for name in shift_names]).mean(0)
    if baseline_names:
        scores['conditional_ar'] = torch.stack([scores[name] for name in baseline_names]).mean(0)
    strata = model.strata(inputs.quantiles, pairs)
    result = {'windows': len(inputs.scale), 'temperature': model.temperature, 'eta': model.eta,
              'scale_floor': model.scale_floor,
              'rows': summarize_scores(scores, intervals, inputs.scale, model.scale_floor, strata, pairs)}
    write_json(args.output, result)
    if args.pair_scores:
        with Path(args.pair_scores).open('xb') as stream:
            np.savez_compressed(stream, scale=inputs.scale.cpu().numpy(), bounded_scale=inputs.scale.clamp_min(model.scale_floor).cpu().numpy(),
                strata=strata.cpu().numpy(), lag_groups=pairs.groups.cpu().numpy(),
                **{'crps__'+k: v.cpu().numpy() for k, v in scores.items()},
                **{k: v.cpu().numpy() for k, v in intervals.items()})
    print(json.dumps(result, indent=2, allow_nan=False))
