# Reviewer revision analysis outputs

This directory contains the publication level summaries added during peer review. Raw model checkpoints and cached feature tensors are excluded because they are large, but every table in this directory can be regenerated from the experiment scripts and the documented dataset layout.

## Regeneration commands

Run the following commands from the repository root in the `research-gpu` environment:

```powershell
python scripts/run_prospective_pemsd4.py --seed 17 --backbone-batch-size 16 --output artifacts/runs/prospective_pemsd4_seed17 --cache artifacts/cache/prospective_pemsd4_seed17
python scripts/run_prospective_pemsd4.py --seed 101 --backbone-batch-size 16 --output artifacts/runs/prospective_pemsd4_seed101 --cache artifacts/cache/prospective_pemsd4_seed101
python scripts/run_expert_count_ablation.py
python scripts/reviewer_revision_analysis.py --bootstrap-replicates 5000
python scripts/generate_reviewer_revision_figures.py
```

The five PeMSD4 backbone seeds are 17, 42, 73, 101, and 202. Statistical intervals use 5,000 hierarchical bootstrap replicates. The outer unit is the independently trained backbone, and the inner unit is a circular block of 288 five minute forecast origins, corresponding to one day.

## Principal files

`dependence_aware_point_summary.csv` contains the revised point comparison intervals. `pemsd4_replication_inference.json` contains the five backbone Fourier minus MLP result. `pemsd4_conditional_calibration.csv` and `pemsd4_node_calibration.csv` contain the post hoc calibration diagnostics. `expert_count_ablation_summary.json` contains all validation only runs for one to five experts. `circuit_structure_expressivity.csv`, `simulator_scaling.csv`, and `quantum_simulator_cost.csv` document the circuit and simulator analyses.
