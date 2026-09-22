"""Verify current paper exports, exhaustive offsets and aggregation definitions."""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

RESULTS = Path(__file__).resolve().parents[1] / 'results'


def rows(name):
    with (RESULTS / name).open() as stream:
        return list(csv.DictReader(stream))


def test_checksums_and_panel():
    manifest = json.loads((RESULTS / 'manifest.json').read_text())
    assert manifest['release'] == '0.1.0'
    for name, digest in manifest['files'].items():
        assert hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() == digest
    cells = rows('cellwise_scores.csv')
    assert len({r['cell'] for r in cells}) == 12
    assert {r['group'] for r in cells} == {'all', 'low', 'middle', 'high'}
    for row in cells:
        assert math.isclose(100*(1-float(row['geometry'])/float(row['time'])), float(row['gain_percent']), abs_tol=1e-10)
    datasets = rows('datasets.csv')
    assert [sum(int(r[k]) for r in datasets) for k in ('train', 'validation', 'test')] == [12339, 4720, 4870]


def test_main_aggregation():
    cells, shifts = rows('cellwise_scores.csv'), rows('alignment_cellwise.csv')
    for row in rows('main.csv'):
        if row['comparator'] == 'time':
            metric = 'bounded_crps' if row['scoring'] == 'normalized' else 'physical_crps'
            gains = [float(r['gain_percent']) for r in cells if r['group'] == 'all' and r['metric'] == metric]
        else:
            gains = [float(r['gain_vs_mean47']) for r in shifts if r['weight'] == row['scoring']]
        assert len(gains) == 12
        for name, value in [('gain_percent', mean(gains)), ('median_percent', median(gains)),
                            ('without_top_two_percent', mean(sorted(gains)[:-2]))]:
            assert math.isclose(value, float(row[name]), abs_tol=1e-10)
        assert sum(x > 0 for x in gains) == int(row['positive_cells'])


def test_all_alignment_scores():
    scores, summaries = rows('alignment_scores.csv'), rows('alignment_cellwise.csv')
    assert len(scores) == 12*2*48
    for summary in summaries:
        group = {int(r['shift']): float(r['crps']) for r in scores if r['cell'] == summary['cell'] and r['scoring'] == summary['weight']}
        assert set(group) == set(range(48))
        gain = 100*(1-group[0]/mean(group[s] for s in range(1, 48)))
        assert math.isclose(gain, float(summary['gain_vs_mean47']), abs_tol=1e-10)
    curves = rows('alignment.csv')
    for weight in ('normalized', 'physical'):
        group = [r for r in curves if r['weight'] == weight]
        assert len(group) == 48
        assert int(min(group, key=lambda r: float(r['common_reference_macro']))['shift']) == 0


def test_generic_scores_and_aggregation():
    scores = rows('absolute_baseline_scores.csv')
    cells = sorted({r['cell'] for r in scores})
    for summary in rows('generic_comparisons.csv'):
        geometry, baseline, gains, scaled_g, scaled_b = [], [], [], [], []
        for cell in cells:
            group = {r['method']: float(r['normalized']) for r in scores if r['cell'] == cell}
            b = mean(v for k, v in group.items() if k.startswith('ar_')) if summary['method'] == 'conditional_ar' else group[summary['method']]
            a = group['geometry']
            geometry.append(a); baseline.append(b); gains.append(100*(1-a/b))
            scaled_g.append(a/group['time']); scaled_b.append(b/group['time'])
        assert math.isclose(mean(gains), float(summary['normalized_mean']), abs_tol=1e-10)
        assert math.isclose(median(gains), float(summary['normalized_median']), abs_tol=1e-10)
        assert math.isclose(mean(baseline), float(summary['mean_normalized_score']), abs_tol=1e-12)
        assert math.isclose(100*(1-sum(geometry)/sum(baseline)), float(summary['pooled_normalized_gain']), abs_tol=1e-10)
        assert math.isclose(100*(1-sum(scaled_g)/sum(scaled_b)), float(summary['reference_standardized_pool_gain']), abs_tol=1e-10)
