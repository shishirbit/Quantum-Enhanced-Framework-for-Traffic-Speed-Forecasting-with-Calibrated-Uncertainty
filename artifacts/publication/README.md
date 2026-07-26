# Publication artifact package

This package consolidates all non-smoke experiments completed through decision D023.
PeMSD8 test results are intentionally absent because the SA-CQR validation gate failed.

## Tables

- `table01_confirmatory_point_forecasting.csv`
- `table02_quantum_matched_control_effects.csv`
- `table03_resource_efficiency.csv`
- `table04_confirmatory_paired_statistics.csv`
- `table04b_confirmatory_friedman_statistics.csv`
- `table05_quantum_architecture_ablation.csv`
- `table06_quarts_v2_pilot.csv`
- `table07_quarts_uq.csv`
- `table08_quarts_rac_validation.csv`
- `table09_raq_asymmetry_validation.csv`
- `table10_pemsd4_prospective.csv`
- `table11_pemsd4_independent_backbones.csv`
- `table12_pemsd4_paired_statistics.csv`
- `table13_pemsd4_recent_conformal.csv`
- `table14_sacqr_pemsd8_validation.csv`
- `table15_experiment_gate_summary.csv`

## Main figures

- `figure01_confirmatory_point_forecasting.pdf`
- `figure02_quantum_matched_control_effects.pdf`
- `figure03_uncertainty_stage_synthesis.pdf`
- `figure04_pemsd4_independent_backbones.pdf`
- `figure05_calibration_repair_and_transfer.pdf`
- `figure06_accuracy_efficiency_tradeoff.pdf`

## Supplementary figures

- `figureS01_horizon_performance.pdf`
- `figureS02_missing_input_metr_la.pdf`
- `figureS03_missing_input_pems_bay.pdf`
- `figureS04_quantum_ablation.pdf`
- `figureS05_quantum_noise.pdf`
- `figureS06_quarts_v2.svg`
- `figureS07_quarts_uq.svg`
- `figureS08_quarts_rac.svg`
- `figureS09_raq_asymmetry.svg`
- `figureS10_pemsd4_prospective.svg`
- `figureS11_pemsd4_head_seeds.png`
- `figureS12_sacqr_validation.png`

## Integrity exclusions

- Smoke and preflight runs are excluded.
- The pre-protocol METR-LA seed-17 batch-128 run is excluded.
- Validation-only branches are labeled and not mixed with canonical test results.
- PeMSD8 canonical test targets were not evaluated.
