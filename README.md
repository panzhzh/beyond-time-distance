<div align="center">

# Beyond Time Distance
### Calibrating Future Changes with Forecast Geometry

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-accelerated-76B900?logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-0077BB)
![Version](https://img.shields.io/badge/Release-v0.1.0-009988)

**Turn marginal forecasts into calibrated distributions of future changes.**

[Quick start](#quick-start) · [Method](#method) · [Results](#results) · [Reproduction](#reproduction) · [Repository guide](#repository-guide)

</div>

Multi-step forecasters provide a marginal distribution at each horizon. Uncertainty in a future change, such as $Y_j-Y_h$, also depends on how those horizons move together. This package uses the **shape of the current forecast** to calibrate that dependence while preserving the common marginal distributions.

- **Valid by construction.** A rational geometry kernel and a Schur product preserve positive semidefiniteness and unit diagonal.
- **Native change-distribution calibration.** Fit and select dependence heads with reconstructed change CRPS, common temperature, and training-derived bounded IQR weights.
- **Matched information controls.** A four-parameter time kernel and all 47 nonzero circular shifts use the same fitting and selection budgets.
- **Portable implementation.** Fit from cached quantiles, save JSON/NPZ state, and apply the correction without extra backbone calls.

## Quick start

Use Python 3.12 and a CUDA-enabled PyTorch installation compatible with your system. From this directory:

```bash
pip install -e '.[test]'
python examples/quickstart.py --device cuda:0 --output /path/to/outputs/example
```

The example fits a small synthetic problem and checks the saved model after reloading. For the paper protocol, use [`configs/paper.json`](configs/paper.json).

The Python interface accepts raw marginal quantiles and a shared correlation reference:

```python
from beyond_time_distance import CalibrationBatch, NativeConfig, fit_calibration

# q: [queries, horizons, 9], ordered probabilities 0.1, ..., 0.9
# B0: [queries, horizons, horizons], a valid correlation matrix
# y: [queries, horizons]; u: [queries], history IQR with a 1e-6 floor
# Use float64 tensors on the same CUDA device.
train = CalibrationBatch(q_train, B0_train, y_train, u_train)
validation = CalibrationBatch(q_val, B0_val, y_val, u_val)

model, trace = fit_calibration(
    train, validation,
    pilot_temperature=T0,
    config=NativeConfig(),
    shifts=range(1, 48),
)
model.save('/path/to/outputs/calibrated_heads')

correlations = model.predict(q_test, B0_test)
quantile_knots = model.native_knots(q_test)
```

`T0` is the shared pilot marginal temperature. The full command-line workflow below fits both the temporal reference and this pilot temperature. The final temperature is selected once on validation data using bounded marginal CRPS and is shared by every head. `model.predict` returns full correlation matrices; supply a `ChangePairs` object to compute only the scored pairs.

## Method

Four within-query centered coordinates describe log spread, location, adjacent location change, and tail-width ratio. Training statistics standardize the coordinates, with clipping at $\pm8$. The learned squared distance is

$$
D_z(h,j)=\sum_{k=1}^{4} a_k c_k\bigl(z_{hk}-z_{jk}\bigr)^2,
\qquad a_k=\operatorname{softplus}(\theta_k),
$$

where $c_k$ is a training-derived distance scale. A rational kernel contracts the shared temporal reference:

$$
K_z(h,j)=\frac{1}{1+D_z(h,j)},\qquad
R_z=B\odot\bigl[(1-\gamma)\mathbf{1}\mathbf{1}^{\mathsf T}+\gamma K_z\bigr],
\qquad B=(1-\eta)B_0+\eta I.
$$

The kernel has unit diagonal and is positive semidefinite. Hence $R_z$ is a valid correlation matrix. Setting $\gamma=0$ recovers the common reference $B$. The scalar $\eta$ is selected using the time arm and shared across all arms.

### Fitting and selection

1. **Common reference.** Fit the marginal pilot temperature, the temporal gate, and the time-distance reference on training/validation partitions.
2. **Gaussian initialization.** Independently initialize geometry, matched time, and each requested circular alignment.
3. **Native CRPS fitting.** Run a native bounded-CRPS pass at the pilot temperature, then a second pass at the final common temperature. Each pass fits six strength vectors from the same initialization: two fitting interpolations × three penalties.
4. **Exact validation confirmation.** A differentiable PCHIP risk surface screens the deployment interpolation menu. Reconstructed 4,096-point validation CRPS selects among the three shortlisted nonzero candidates, the null, and the preceding fitted head.
5. **Common shrinkage.** Select $\eta$ from the training-supported time-arm menu, confirm it on validation, and apply it to every head.

The native training/screening surfaces use 512/1,024 balanced Sobol points and 17 contraction-grid values. Final scoring uses the same nine marginal knots, linear interpolation between knots, and logarithmic tails for every method. [`configs/paper.json`](configs/paper.json) records the numerical settings; [`configs/reference.json`](configs/reference.json) records the shared reference settings.

## Results

Six sources × two backbones (**Chronos-Bolt** and **FlowState**) give twelve dataset–backbone cells. Gains below are equal-cell means of within-cell relative reductions, expressed in percent. Higher is better.

### Matched comparisons

| Comparator | Normalized CRPS reduction | Physical CRPS reduction |
|:--|--:|--:|
| Matched time kernel | **0.2005%** [0.1041, 0.3196] | **0.2483%** [0.1131, 0.3776] |
| Mean of all 47 nonzero shifts | **0.1801%** [0.1269, 0.2365] | **0.2034%** [0.1413, 0.2674] |

Correct alignment ranks **1/48** under both aggregate scoring conventions. The normalized time and mean-shift comparisons are positive in **12/12 cells**; their physical-unit counterparts are positive in **12/12** and **11/12**, respectively. Each shift is independently refitted. The mean-shift comparator averages scores across offsets, rather than mixing predictive distributions.

“Normalized” means division by the history IQR bounded below by the fifth percentile of positive training IQRs and a numerical floor of $10^{-6}$. Physical-unit scores retain each source's units when computing the within-cell ratio. The cross-source summary averages these relative effects.

### Where the correction helps

Distance strata use fixed within-lag training tertiles of the initialized geometry distance.

| Population | Normalized CRPS reduction | Physical CRPS reduction | Normalized IS90 reduction | Physical IS90 reduction |
|:--|--:|--:|--:|--:|
| All pairs | 0.2005% | 0.2483% | 1.2124% | 1.5322% |
| High forecast distance | **0.6570%** | **0.6898%** | **3.5900%** | **3.8684%** |

In the high-distance group, 90% coverage moves from **82.70% to 85.68%**, and mean absolute coverage error falls from **7.30 to 4.32 percentage points**. Interval width increases by **9.85%**, accompanied by improvements in both CRPS and the proper 90% interval score. Absolute coverage error improves in all twelve cells, and each cell's high-distance CRPS gain exceeds its overall gain. Low-distance width changes by only **0.24%**: the adjustment is concentrated where the temporal control under-covers most. The full low/middle/high breakdown is included in [`interval_diagnostics.csv`](results/interval_diagnostics.csv).

### Generic conditional baselines

All entries use the same final marginals. Conditional AR is the expected score across five independently trained, validation-selected seeds.

| Method | Mean absolute normalized CRPS ↓ | Geometry's relative reduction | Positive cells |
|:--|--:|--:|--:|
| Geometry | **0.15493** | — | — |
| Time kernel | 0.15519 | 0.2005% | 12/12 |
| Common reference | 0.15524 | 0.2424% | 12/12 |
| Full-profile gate | 0.15498 | 0.0515% [−0.1636, 0.2387] | 7/12 |
| Conditional AR | 0.15633 | **0.8285%** [0.5317, 1.0836] | 11/12 |

The full-profile interval includes zero. The conditional-AR comparison favors geometry under the reported cell-relative and pooled normalized summaries. Absolute scores, individual AR seeds, physical-unit results, medians, and aggregation checks are provided in the result files.

Intervals use 100,000 paired cluster/time-block bootstrap draws. Bonferroni family sizes are 144 for the time/group analysis, 2 for the two mean-shift comparisons, and 4 for the generic-baseline comparisons. File checksums and metric definitions are recorded in [`results/manifest.json`](results/manifest.json).

## Reproduction

### 1. Prepare cached forecasts

Create separate `train.npz`, `validation.npz`, and `test.npz` files for each dataset–backbone cell:

| Array | Shape | Meaning |
|:--|:--|:--|
| `history` | $N\times L$ | Observed input window |
| `quantiles` | $N\times48\times9$ | Raw forecast quantiles at probabilities 0.1–0.9 |
| `target` | $N\times48$ | Observed future trajectory |
| `scale` (optional) | $N$ | History IQR, floored at $10^{-6}$; computed from `history` when omitted |

The paper uses lags $\{1,2,4,8,12\}$, giving 213 horizon pairs. Scores assign equal weight to each lag and average within a lag. The source registry is [GIFT-Eval](https://huggingface.co/datasets/Salesforce/GiftEval); dataset IDs, resolutions, context lengths, and split counts are in [`datasets.csv`](results/datasets.csv). The [GIFT-Eval repository](https://github.com/SalesforceAIResearch/gift-eval) documents the source data and backbone forecast APIs. This package starts from cached forecasts, so the dependence workflow is independent of the backbone implementation.

### 2. Fit the reference and matched controls

```bash
btd-fit \
  --train /path/to/data/train.npz \
  --validation /path/to/data/validation.npz \
  --config configs/paper.json \
  --shifts all \
  --device cuda:0 \
  --output /path/to/outputs/model
```

Use `--shifts none` to fit just geometry and matched time. Use `--reference /path/to/existing/reference` to reuse a fitted temporal reference. All fits consume training and validation partitions; the test file is supplied only to evaluation. Choose a new output directory for each fit.

Independent dataset–backbone cells can run concurrently on separate GPUs. Set `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to match the CPU allocation. The `batch_size` in the configuration controls quadrature memory without changing the scoring definition.

### 3. Evaluate frozen models

```bash
btd-evaluate \
  --model /path/to/outputs/model \
  --test /path/to/data/test.npz \
  --device cuda:0 \
  --output /path/to/outputs/scores.json \
  --pair-scores /path/to/outputs/pair_scores.npz
```

The JSON includes all/low/middle/high scores, absolute 90% coverage, and interval width. The optional NPZ retains pairwise scores and fixed strata for further paired analysis. When all 47 shifts are present, evaluation also reports their mean-score comparison. An empty lag–stratum returns `null` rather than changing the lag weights.

### 4. Fit generic comparisons

```bash
python examples/fit_baselines.py \
  --train /path/to/data/train.npz \
  --validation /path/to/data/validation.npz \
  --model /path/to/outputs/model \
  --device cuda:0 \
  --output /path/to/outputs/baselines
```

This runs the five-seed conditional-AR baseline and native full-profile refinement under the model's common final temperature and bounded weights. Every AR seed keeps its own validation-selected checkpoint.

```bash
btd-evaluate \
  --model /path/to/outputs/model \
  --baselines /path/to/outputs/baselines \
  --test /path/to/data/test.npz \
  --device cuda:0 \
  --output /path/to/outputs/with_baselines.json
```

### 5. Run checks

```bash
pytest -q
BTD_TEST_DEVICE=cuda:0 pytest -q -m cuda
```

Numerical tests exercise PSD and unit diagonal, full/pair consistency, the exact null, shared marginals, native selection, interval scoring, and portable reload. Result checks verify source checksums, twelve-cell summaries, all 48 alignments, and baseline aggregation.

## Repository guide

```text
src/beyond_time_distance/
  calibration.py     Native fitting, common temperature/shrinkage and saved heads
  kernel.py          Forecast coordinates, PSD kernels and Gaussian initialization
  interpolation.py   Differentiable native-risk interpolation
  reference.py       Shared temporal-reference fitting and persistence
  base.py            Common marginal and correlation components
  gates.py           Temporal/full-profile gates and conditional AR
  baselines.py       Native adaptation of generic comparisons
  scoring.py         Balanced copula quadrature, CRPS and interval diagnostics
  cli.py             Fit/evaluate commands and fixed-stratum summaries
  io.py              Cached forecast input
configs/             Paper and reference settings
examples/            Synthetic quick start and baseline workflow
results/             Current manuscript tables, alignment sweep and diagnostics
tests/              Numerical and result-integrity checks
```

The release includes the core implementation, matched controls, generic comparisons, and manuscript-consistent aggregate results. Backbone checkpoints and raw source datasets remain with their upstream providers.

## License

[MIT](LICENSE).
