# Additional frozen-grid transfer experiment

This experiment uses an additional **2D KdV-type** configuration (`dispconv2d_s1`),
separate from the paper's 13-configuration main comparison. Five existing models
trained on 64×64 are frozen. Each evaluates the same 30 initial states for 200
recursive steps at 32×32, 64×64 and 128×128. No retraining or fitting is performed.

## Results

Values below are means across five training runs. Every run first averages nRMSE
over the 30 initial states, then takes the geometric mean across the stated window.

| Grid / evaluation scheme | GMean1–100 | GMean101–200 | nRMSE at 200 |
|---|---:|---:|---:|
| 64×64, native | 0.01491 | 0.03643 | 0.04869 |
| 32×32, direct frequency adaptation | 0.02278 | 0.06967 | 0.09445 |
| 128×128, direct frequency adaptation | 0.01748 | 0.04789 | 0.06489 |
| 32×32, resample through 64×64 | 0.01489 | 0.03645 | 0.04870 |
| 128×128, resample through 64×64 | 0.01491 | 0.03643 | 0.04868 |

All 25 run/configuration groups completed with finite errors. Direct transfer has
higher error than the native grid, with a smaller increase at 128×128. Resampling
through 64×64 closely reproduces native-grid accuracy. This second scheme measures
compatibility through resampling, rather than recovery of newly resolved fine scales.
This additional experiment compares SGNO transfer schemes; it includes no external
baseline comparison.

[Run-level results](../results/resolution_transfer/runs.csv),
[medians, means and sample SDs](../results/resolution_transfer/summary.csv), and
[reference convergence checks](../results/resolution_transfer/convergence.json)
are included in the release.

The per-trajectory nRMSE arrays are also provided under
`results/resolution_transfer/metrics/`. Recompute all 25 groups with:

```bash
python scripts/verify_resolution_results.py
```

## Direct frequency adaptation

`sgno.resolution.FrozenGridPredictor` implements the evaluated inference convention:

- Keep every learned parameter tensor at its original shape and value.
- Reuse integer Fourier modes in the open native Nyquist band,
  `abs(k_x) < 32` and `k_y < 32`; assign zero outer multiplier to new higher modes.
- Scale the fitted Fourier offset by `N / 64` for the orthonormal 2D FFT.
- Evaluate the output filter at the same physical frequencies, using `2*k/64`.
- Rebuild grid-dependent masks and spectral indices and clear the zero-input cache.

At 64×64 the original predictor is used unchanged. Load a fresh model for each
grid/policy; the adapter freezes it for inference and is not a training module.
For a checkpoint in the release's model-state format:

```python
import torch
from sgno import build_model
from sgno.resolution import FrozenGridPredictor

saved = torch.load("checkpoint.pt", map_location="cpu", weights_only=False)
model = build_model(ndim=2, seed=0)
model.load_state_dict(saved["model"], strict=True)
predictor = FrozenGridPredictor(model, grid=128, policy="frequency_direct")
# state: float32 tensor with shape (batch, 1, 128, 128)
next_state = predictor(state)
```

Use `policy="project_to64"` for periodic Fourier resampling to the training grid,
the unchanged model update, and resampling back, at every prediction step. Nyquist
coefficients are split or merged when changing an even-sized grid.

## Reference and scope

The physical PDE is held fixed across grids. The original difficulty coefficients
`gamma=(0,0,0,-14,-9)` and `delta=(0,-2,0)` are converted at the training grid
N=64 using Exponax's difficulty-to-normalized-coefficient conversion, with maximum
absolute field value 1. The domain length and output interval are both one.
The reference is generated on 128×128 with 64 internal substeps per output, then
Fourier-projected to each test grid.

Three independent validation initial states check 64 versus 128 temporal substeps
on 128×128, and 128×128 versus 256×256 at 128 substeps. The maximum per-time relative
L2 discrepancy must be at most 0.01. All checks passed; the largest temporal
discrepancy was 0.00736. Checkpoint and learned-parameter hashes remained unchanged.
The full convention is in [protocol.json](../results/resolution_transfer/protocol.json).

These results establish empirical transfer for this PDE and these three grids.
The original fixed-grid implementation requires the specified runtime adaptation
for direct transfer. A discretization-convergence theorem is not established by
this experiment. The additional task's datasets and checkpoints are not bundled;
the adapter accepts a compatible 2D checkpoint, but the main-task
`generate_data.py` commands alone do not regenerate this additional experiment.
