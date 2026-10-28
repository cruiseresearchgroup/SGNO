# Reproduction notes

## Implementation map

`sgno.model.build_model(ndim, seed)` constructs the paper predictor. The public
configuration supports dimensions 1–3 and the five reported seeds, 0–4.
`sgno.ablations.build_predictor` selects the controlled variants. Internal modules
separate the reference Fourier layers, initialization, fitted map, spectral output
filter, mean handling, and zero-initialized correction head. Their composition is
the released model; the internal reference class by itself is not the final SGNO.

The fitted state map operates on the full input grid. Fitting uses only adjacent
training states, with the original frequency constraints and regularization in
`_fit.py`. It initializes learnable parameters, rather than replacing training
with access to an analytical solution. The learned correction is filtered and
its zero-input high-frequency response is subtracted. The output is recentered
before restoring the input mean. See `_increment.py`, `_mean.py`, and
`_consistency.py` for these operations.

During release preparation, the reorganized implementation was compared to the
paper implementation in all three dimensions at seeds 0 and 4. State dictionaries,
forward outputs, input gradients, and parameter gradients agreed exactly on CPU.
The output head was perturbed for this check so that the correction path was
active. This validates code reorganization, not benchmark accuracy.

## Training and precision

Training flattens adjacent pairs in trajectory-major, then time-major order and
shuffles each epoch with a seeded CPU generator. Adam uses betas `(0.9, 0.999)`,
epsilon `1e-8`, and no weight decay. Update indices start at zero, so the first
optimizer update uses learning rate zero. No multi-step loss is used for SGNO.

The original training path allows cuDNN TF32 for the fitted joint model while
disabling matmul TF32. The direct-map controls disable both. Reported replays and
`evaluate.py` disable both and use float32/complex64. Hardware and PyTorch/CUDA
versions can affect training trajectories even when the seeds match; the checked
release environment is stated in the README. Numerical data-generation versions
are pinned independently. Regeneration uses CPU JAX and the original task-specific
precision, including float64 physical-task reference checks.

`--smoke-updates N` allows a short software check and smaller trajectory counts.
Its checkpoints and summaries are explicitly marked `smoke_only` and excluded
from aggregation. It is not a reduced-cost reproduction of the paper's scores.

## Metrics

`metrics.npz` stores the NRMSE of every test trajectory at every predicted step.
At each step, NRMSE is averaged over trajectories. GMean100 is the geometric mean
of that mean-error curve over steps 1–100, adding `1e-12` before taking its logarithm.
The late-window geometric mean uses steps
101–200. The evaluator also reports final-step error and least-squares slopes of
the trajectory-averaged error and its logarithm in the late window. The 3D metric
preserves the original float32 subtraction before float64 norm accumulation;
the 1D/2D metric uses float64 subtraction. See `metrics.py` for exact definitions.

Five-run summaries contain the median, mean, sample standard deviation (`ddof=1`),
and each run's value. The main and compact ablation tables use the median;
Figure 4 uses the mean and sample SD. The aggregation script retains all test
trajectories and training runs. It requires exactly one evaluation for each of
the five seeds and a common prediction horizon. A group containing a non-finite
value receives `NF` summaries. Recorded comparisons and newly trained controls
have distinct provenance and must not be pooled.

## Spectral and field diagnostics

For Fourier coefficients `rho * exp(i phi)` and `r * exp(i psi)`, the squared
difference is `(rho-r)^2 + 2*rho*r*(1-cos(phi-psi))`. The first term is the
amplitude contribution; the second is the amplitude-weighted phase contribution.
`sgno.metrics.field_diagnostics` sums them with real-FFT Parseval weights and
normalizes each field by its reference energy. Their sum equals normalized
squared field error. These quantities are **not square-rooted** for Figure 4.

The squared-norm ratio is `sum(prediction**2) / sum(reference**2)`. Field
correlation is the inner product divided by the product of field norms, without
an additional mean subtraction. Each is computed per test field before averaging
within a training run. Across runs, Figure 4 reports means and sample SDs;
the KS field-norm/correlation discussion reports medians. Correlation is undefined
for a zero-norm field and is retained as non-finite in that case.

`evaluate.py` saves all per-field endpoint diagnostics in `metrics.npz` under
`endpoint_*` keys and their within-run means in `summary.json`. With `--steps 200`,
these are the step-200 diagnostics. The aggregation script produces corresponding
five-run summaries. The slope called `late_absolute_slope` is the OLS slope of
nRMSE itself, in nRMSE per step; it is not an unnormalized physical L2 error.

The optional frozen-grid adapter and its evaluated convention are documented in
[RESOLUTION_TRANSFER.md](RESOLUTION_TRANSFER.md).

## Release scope

The release includes the SGNO pipeline, component and propagator controls,
task generators, and the reported table records. The optional `--corrector fno`
code path is an architecture substitution inside the same outer construction;
it does not reproduce the standalone FNO main baseline. Main-table external
baselines retain the protocols described in the paper. No remote cluster account, private
filesystem, experiment tracker, or author-specific configuration is required.
Raw benchmark trajectories and trained checkpoints can be generated using the
provided commands and are not included in the download.
