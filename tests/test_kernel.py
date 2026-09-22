"""Numerical contracts for the public dependence-calibration interface."""
import json
import os

import pytest
import torch

from beyond_time_distance import CalibrationBatch, FitConfig, GeometryRefinement
from beyond_time_distance.kernel import raw_coordinates, fit_initializer

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not os.environ.get("BTD_TEST_DEVICE"),
                                                  reason="Set BTD_TEST_DEVICE to an available CUDA device")]


@pytest.fixture
def example():
    device = torch.device(os.environ["BTD_TEST_DEVICE"])
    assert device.type == "cuda"
    torch.cuda.set_device(device)
    torch.manual_seed(23)
    h = torch.arange(48, device=device, dtype=torch.float64)
    location = torch.sin(h / 7)[None] + torch.randn(12, 48, device=device, dtype=torch.float64) * .06
    spread = .3 + .1 * torch.sigmoid(h / 12)[None]
    q = location[:, :, None] + spread[:, :, None] * torch.special.ndtri(
        torch.arange(1, 10, device=device, dtype=torch.float64) / 10)
    reference = .8 ** (h[:, None] - h[None]).abs()
    reference = reference[None].expand(len(q), -1, -1)
    target = location + torch.randn_like(location) * .4
    return q, reference, target, torch.ones(len(q), device=device, dtype=torch.float64)


def fitted(example):
    q, r, y, s = example
    cfg = FitConfig(gamma_fit=(.125, 1.), penalties=(.001,), gammas=(0., .5, 1.), max_iter=4)
    train = CalibrationBatch(q[:8], r[:8], y[:8], s[:8])
    val = CalibrationBatch(q[8:], r[8:], y[8:], s[8:])
    return fit_initializer(train, val, config=cfg), cfg


def test_fit_psd_pairs_null_and_marginals(example):
    (head, trace), cfg = fitted(example)
    q, reference, _, _ = example
    untouched = q.clone()
    assert len(trace["menu"]) == 6
    head.gamma = .5
    full = head.predict(q, reference)
    assert torch.linalg.eigvalsh(full).min() >= -1e-12
    torch.testing.assert_close(full.diagonal(dim1=1, dim2=2), torch.ones_like(q[:, :, 0]), rtol=0, atol=1e-14)
    pairs = cfg.pairs(q.device)
    torch.testing.assert_close(head.predict(q, reference, pairs), full[:, pairs.left, pairs.right], rtol=0, atol=1e-14)
    head.gamma = 0.
    torch.testing.assert_close(head.predict(q, reference), reference, rtol=0, atol=0)
    torch.testing.assert_close(q, untouched, rtol=0, atol=0)


def test_portable_reload(example, tmp_path):
    (head, _), _ = fitted(example)
    path = tmp_path / "head.json"
    head.save(path)
    restored = GeometryRefinement.load(path, example[0].device)
    torch.testing.assert_close(restored.predict(example[0], example[1]), head.predict(example[0], example[1]), rtol=0, atol=0)
    payload = json.loads(path.read_text()); payload["gamma"] = 2.
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="interpolation"):
        GeometryRefinement.load(path, example[0].device)


def test_normalization_and_fits_use_training_only(example):
    (head, trace), cfg = fitted(example)
    q, r, y, s = example
    altered, other = fit_initializer(CalibrationBatch(q[:8], r[:8], y[:8], s[:8]),
                                    CalibrationBatch(q[8:] * 20, r[8:], y[8:] + 100, s[8:]), config=cfg)
    for name in ["center", "sd", "distance_factor"]:
        torch.testing.assert_close(getattr(head, name), getattr(altered, name), rtol=0, atol=0)
    assert trace["optimization"] == other["optimization"]


def test_degenerate_quantile_coordinates(example):
    q, _, _, _ = example
    for value in [torch.zeros_like(q), torch.ones_like(q) * 1e-20]:
        z = raw_coordinates(value)
        assert bool(torch.isfinite(z).all())
        torch.testing.assert_close(z, torch.zeros_like(z), rtol=0, atol=1e-12)


def test_shifts_keep_coordinate_vectors(example):
    (head, _), _ = fitted(example)
    aligned = head.coordinates(example[0])
    head.shift = 17
    torch.testing.assert_close(head.coordinates(example[0]), torch.roll(aligned, 17, 1), rtol=0, atol=0)


def test_incompatible_batch_rejected(example):
    q, r, y, s = example
    with pytest.raises(ValueError, match="positive"):
        CalibrationBatch(q, r, y, -s).validate(FitConfig())
    with pytest.raises(ValueError, match="float64"):
        CalibrationBatch(q.float(), r, y, s).validate(FitConfig())
