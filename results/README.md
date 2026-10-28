# Results accompanying the paper

These are recorded experiment results. Training/evaluation commands create new
outputs under `runs/`. The release's software checks are separate.

| Paper content | Records |
|---|---|
| Table 1: all 13 configurations | [Compact table](main_comparison.tex), [all 310 run scores](paper/main_seed_scores.csv), [detailed table](tables/main_all_seeds.tex) |
| Table 2: initialization, adaptation, layer paths | [Compact table](main_ablation.tex), [all reported runs](paper/ablation_seed_scores.csv), [initialization table](tables/ablation_all_seeds.tex), [path table](tables/ablation_paths_seeds.tex) |
| Linear fitted-only comparison | [Table](tables/linear_fitted_only.tex) |
| Additional objectives and capacity | [Table](tables/additional_compact.tex) |
| Output-processing controls | [All runs](paper/output_processing_seed_scores.csv), [table](tables/output_processing_seeds.tex) |
| Long-horizon errors and slopes | [All runs](paper/temporal_seed_scores.csv), [table](tables/temporal_compact.tex), [slope summary](late_growth.md) |
| Figure 4: KS amplitude and phase errors | [All ten run-level points](paper/ks_endpoint_phase_values.csv) |
| KS field norms and correlation | [All ten run-level points](paper/ks_field_endpoint_scores.csv) |
| Additional grid-transfer experiment | [All 25 run/configuration results](resolution_transfer/runs.csv), [summary](resolution_transfer/summary.csv), [protocol](../docs/RESOLUTION_TRANSFER.md) |

The compact main and ablation tables report medians. [summary.csv](paper/summary.csv)
contains recomputed medians, means, sample SDs, and values A–E for every group in
the supplied paper CSVs. A–E correspond to seeds 0–4. All five runs are retained
for each reported setting. `NF` or an empty source CSV cell denotes a non-finite
result; summaries do not substitute a surviving-run average.

In `ablation_seed_scores.csv`, A = fixed identity, B = trainable identity,
C = fixed fitted map, D = trainable fitted map. These method codes are distinct
from the A–E run columns in summary tables. `sgno` denotes the original
parameterization; path interventions retain that parameterization. Output-processing
controls are separately trained groups and have their own full-processing reference.

The main results compare FNO, CNO, KNO, and IS-FNO(1). Additional training objectives
and the larger Refiner use the separate protocols described in Appendix A3.3.
The extra 2D KdV-type qualitative example and resolution study are separate from
the main-table 1D KdV task.

The `.tex` files are manuscript table fragments and require its labels and table
packages when compiled. Full datasets and pretrained checkpoints are not bundled;
the main-task generation and training commands are in the root README.
