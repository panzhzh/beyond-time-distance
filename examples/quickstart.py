"""Synthetic native calibration with a supplied valid temporal reference."""
import argparse
from pathlib import Path
import torch
from beyond_time_distance import CalibrationBatch, CalibratedHeads, FitConfig, NativeConfig, fit_calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', required=True, help='New model directory')
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != 'cuda': parser.error('Use a CUDA device')
    if Path(args.output).exists(): parser.error('Choose a new output directory')
    torch.cuda.set_device(device); torch.manual_seed(7)
    h = torch.arange(48, dtype=torch.float64, device=device)
    location = torch.sin(h / 8)[None] + .05 * torch.randn(12, 48, dtype=torch.float64, device=device)
    quantiles = location[:, :, None] + .4 * torch.special.ndtri(torch.arange(1, 10, dtype=torch.float64, device=device) / 10)
    reference = (.9 ** (h[:, None] - h[None]).abs())[None].expand(len(location), -1, -1)
    target = location + .4 * torch.randn_like(location)
    scale = torch.ones(len(location), dtype=torch.float64, device=device)
    train = CalibrationBatch(quantiles[:8], reference[:8], target[:8], scale[:8])
    validation = CalibrationBatch(quantiles[8:], reference[8:], target[8:], scale[8:])
    # A small numerical demonstration; configs/paper.json contains the paper budget.
    config = NativeConfig(fit=FitConfig(max_iter=5, penalties=(.001,)), train_count=64,
                          validation_count=128, confirmation_count=256, batch_size=2)
    model, _ = fit_calibration(train, validation, pilot_temperature=1., config=config,
                              progress=lambda event: print(event, flush=True))
    model.save(args.output)
    restored = CalibratedHeads.load(args.output, device)
    expected = model.predict(quantiles, reference)
    for name, value in restored.predict(quantiles, reference).items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        assert float(torch.linalg.eigvalsh(value).min()) > -1e-12
    print({'temperature': model.temperature, 'eta': model.eta, 'gamma': model.heads['geometry'].gamma,
           'strengths': model.heads['geometry'].strengths.tolist(), 'reload': 'passed'})


if __name__ == '__main__':
    main()
