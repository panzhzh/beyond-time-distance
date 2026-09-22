"""Fit native full-profile and per-seed conditional-AR baselines."""
import argparse
from pathlib import Path
from beyond_time_distance import CalibratedHeads, TemporalReference
from beyond_time_distance.baselines import fit_baselines, save_baselines
from beyond_time_distance.cli import device_arg
from beyond_time_distance.io import load_forecasts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('train', 'validation', 'model', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if Path(args.output).exists(): parser.error('Choose a new output directory')
    if Path(args.train).resolve() == Path(args.validation).resolve(): parser.error('Use separate training and validation partitions')
    device = device_arg(args.device)
    train, yt = load_forecasts(args.train, device)
    validation, yv = load_forecasts(args.validation, device)
    model = CalibratedHeads.load(args.model, device)
    reference = TemporalReference.load(Path(args.model) / 'reference', device)
    models, traces = fit_baselines(train, validation, yt, yv, reference, model)
    save_baselines(models, traces, args.output)
    print({'saved_models': list(models)})


if __name__ == '__main__':
    main()
