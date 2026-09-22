"""Native calibration: common marginals, PSD, fixed grouping and portable state."""
import os
from dataclasses import replace
import pytest
import torch
from beyond_time_distance import CalibrationBatch, CalibratedHeads, FitConfig, NativeConfig, fit_calibration
from beyond_time_distance.calibration import bounded_floor
from beyond_time_distance.interpolation import ContractionRisk
from beyond_time_distance.scoring import native_change_scores, reconstructed_evidence

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not os.environ.get('BTD_TEST_DEVICE'), reason='Set BTD_TEST_DEVICE to an available CUDA device')]


@pytest.fixture(scope='module')
def native_model():
    device = torch.device(os.environ['BTD_TEST_DEVICE'])
    torch.cuda.set_device(device); torch.manual_seed(18)
    h = torch.arange(16, dtype=torch.float64, device=device)
    loc = h.sin()[None] + torch.randn(10, 16, dtype=torch.float64, device=device) * .1
    q = loc[:, :, None] + .5 * torch.special.ndtri(torch.arange(1, 10, dtype=torch.float64, device=device) / 10)
    ref = (.8 ** (h[:, None] - h[None]).abs())[None].expand(10, -1, -1)
    y = loc + torch.randn_like(loc) * .6
    scale = loc.new_tensor([1e-6, .5, .6, .7, .8, .9, 1., .8, 1., .5])
    train = CalibrationBatch(q[:6], ref[:6], y[:6], scale[:6])
    val = CalibrationBatch(q[6:], ref[6:], y[6:], scale[6:])
    cfg = NativeConfig(fit=FitConfig(horizon=16, lags=(1, 2), gamma_fit=(.125,), penalties=(.001,),
        gammas=(0., .5, 1.), max_iter=3), train_count=32, validation_count=64, confirmation_count=128,
        batch_size=2, contraction_grid=(0., .0625, .25, .5625, 1.))
    model, trace = fit_calibration(train, val, pilot_temperature=1., config=cfg, shifts=(1, 7))
    return model, trace, q, ref, y, scale


def test_native_fit_uses_exact_validation(native_model):
    model, trace, q, ref, y, scale = native_model
    assert set(model.heads) == {'geometry', 'time', 'shift_01', 'shift_07'}
    assert model.scale_floor == bounded_floor(scale[:6])
    for head in trace['heads'].values():
        for stage in ('pilot', 'native'):
            selected = head[stage]['selection']
            assert selected['validation_risk'] == min(r['validation_crps'] for r in head[stage]['confirmed'])
            assert any(r['origin'] == 'initializer' for r in head[stage]['confirmed'])


def test_psd_pair_null_and_reload(native_model, tmp_path):
    model, _, q, ref, _, _ = native_model
    pairs = model.config.fit.pairs(q.device)
    matrix = model.predict(q, ref)
    selected = model.predict(q, ref, pairs)
    for name, value in matrix.items():
        assert float(torch.linalg.eigvalsh(value).min()) > -1e-12
        torch.testing.assert_close(value.diagonal(dim1=1, dim2=2), torch.ones_like(q[:, :, 0]), rtol=0, atol=1e-12)
        torch.testing.assert_close(value[:, pairs.left, pairs.right], selected[name], rtol=0, atol=1e-12)
    null = replace(model, heads={'geometry': replace(model.heads['geometry'], gamma=0.)})
    torch.testing.assert_close(null.predict(q, ref)['geometry'], matrix['reference'], rtol=0, atol=0)
    model.save(tmp_path / 'model')
    restored = CalibratedHeads.load(tmp_path / 'model', q.device)
    torch.testing.assert_close(restored.native_knots(q), model.native_knots(q), rtol=0, atol=0)
    for name, value in matrix.items():
        torch.testing.assert_close(restored.predict(q, ref)[name], value, rtol=0, atol=0)
    torch.testing.assert_close(restored.strata(q, pairs), model.strata(q, pairs), rtol=0, atol=0)


def test_interval_scoring_is_consistent(native_model):
    model, _, q, ref, y, scale = native_model
    pairs = model.config.fit.pairs(q.device)
    corr = {'geometry': model.predict(q, ref, pairs)['geometry']}
    knots = model.native_knots(q)
    scores, intervals = reconstructed_evidence(knots, corr, y, scale, pairs, model.config.quadrature(), ('geometry',))
    old = native_change_scores(knots, corr, y, scale, pairs, **model.config.quadrature())
    torch.testing.assert_close(scores['geometry'], old['geometry'], rtol=0, atol=0)
    assert bool((intervals['geometry__interval_score90'] >= intervals['geometry__width90'] / scale[:, None]).all())
    assert set(intervals['geometry__coverage90'].unique().tolist()) <= {0., 1.}


def test_pchip_is_exact_at_knots_and_differentiable(native_model):
    q = native_model[2]
    grid = q.new_tensor([0., .1, .4, 1.])
    values = q.new_tensor([[1., .8, 1.2, 2.]])
    risk = ContractionRisk.create(grid, values)
    for i, t in enumerate(grid):
        torch.testing.assert_close(risk(t.reshape(1)), values[:, i], rtol=0, atol=1e-14)
    x = q.new_tensor([.2], requires_grad=True)
    risk(x).sum().backward()
    assert bool(torch.isfinite(x.grad).all())
