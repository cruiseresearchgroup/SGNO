# Paper and code correspondence

| Manuscript component | Implementation / release material |
|---|---|
| Structured initial estimate | `_fit.py`, `_affine.py`: adjacent-state fitting and constrained Fourier multiplier |
| Learned correction and shared generator | `sgno.py`: feature attenuation, nonlinear spectral source, pointwise affine path |
| Zero correction at initialization and joint training | `_increment.py`, `scripts/train.py` |
| Mean handling and output processing | `_mean.py`, `_filter.py`, `_consistency.py` |
| Main comparison and component studies | `results/main_comparison.tex`, `results/main_ablation.tex`, `results/paper/` |
| Long-horizon prediction | `scripts/evaluate.py`, `metrics.py`, `assets/long_horizon.png` |
| KS amplitude/phase, norm and correlation diagnostics | `metrics.field_diagnostics`, `assets/ks_accuracy.png`, `results/paper/ks_*` |
| Additional resolution study | `resolution.FrozenGridPredictor`, `docs/RESOLUTION_TRANSFER.md` |

The README uses the final overview and architecture diagrams, the KdV qualitative
comparison for Figure 3, and the KS endpoint diagnostic for Figure 4. The overview's
extra 2D example uses IS-FNO(20); Figures 3 and 4 use IS-FNO(1).

Main and compact ablation results use medians across five runs. Figure 4 uses
means and sample SDs of normalized squared components; field-norm and correlation
values in the text use medians. `scripts/aggregate.py` exports both summaries and
the individual run values. The late slope is fitted to nRMSE.

The released evidence concerns prediction accuracy. The phase-advance example
isolates update error when amplitudes are preserved. The additional resolution
study is documented separately from the manuscript's fixed-grid main protocol.
