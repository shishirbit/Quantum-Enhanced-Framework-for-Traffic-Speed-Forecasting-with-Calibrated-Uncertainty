# Reproducibility record

This release is the permanent computational record for the manuscript
"Controlled Evaluation of Quantum and Classical Residual Calibration for
Probabilistic Traffic Forecasting."

## Archived scope

The release contains:

* the complete Python implementation under `src/quarts/`;
* immutable experiment configurations under `configs/`;
* experiment, statistical analysis, audit, and figure-generation scripts
  under `scripts/`;
* the Jupyter entry point under `notebooks/`;
* dataset acquisition instructions and integrity records under `data/` and
  `research/`;
* run-level JSON manifests and metrics under `artifacts/runs/`;
* the analysis summaries and completion audit under `artifacts/analysis/`;
* publication tables, figures, diagnostic outputs, and the publication
  manifest under `artifacts/publication/`;
* the reviewer-analysis evaluation outputs under
  `artifacts/reviewer_revision/`; and
* the Springer Nature manuscript source and compiled article under
  `manuscript/quarts_springer/`.

The completion audit records 50 reference runs, 10 official STAEformer runs,
and nine circuit ablations with no missing run. The publication manifest
records hashes for the tables and figures used by the manuscript.

## Deliberate exclusions

The archive does not redistribute the benchmark datasets, full forecasting
backbone checkpoints, cached feature tensors, or the main benchmark
per-forecast prediction arrays. The public datasets retain their established
distribution routes. These excluded binary arrays and checkpoints occupy more
than 11 GB and are not required to verify the reported aggregate results
because run-level manifests, metrics, statistical summaries, decision records,
and final publication artifacts are included. The much smaller diagnostic
calibrator checkpoints and evaluation arrays used for the expert-count analysis
are retained under `artifacts/reviewer_revision/`. Dataset acquisition
instructions, expected layouts, dimensions, and cryptographic hashes identify
the inputs used in the experiments.

## Environment

The reported runs used Windows 11, Python 3.12.13, PyTorch 2.13.0 with CUDA
13.0, PennyLane 0.39, NumPy 2.0.2, pandas 2.2.3, SciPy 1.18.0, and
scikit-learn 1.9.0 on an NVIDIA GeForce RTX 5060 Laptop GPU. Quantum circuits
used analytic state-vector simulation. The local Conda environment was named
`research-gpu`.

Install the dependencies and verify the implementation from the repository
root:

```bash
python -m pip install -r requirements.txt
python scripts/run_smoke.py
pytest -q
```

## Regenerating reported evidence

The main publication artifacts are regenerated with:

```bash
python scripts/analyze_results.py
python scripts/build_publication_package.py
python scripts/generate_system_architecture.py
python scripts/reviewer_revision_analysis.py --bootstrap-replicates 5000
python scripts/generate_reviewer_revision_figures.py
```

The canonical audit files are:

* `artifacts/analysis/completion_audit.json`;
* `artifacts/publication/publication_manifest.json`; and
* `artifacts/publication/reviewer_revision/revision_completion_audit.json`.

PeMSD8 canonical test targets are absent by design because the validation gate
failed. This is a protocol decision rather than missing output.

## Version identity

The permanent archive identifies version 1.0.0 and is registered at
[https://doi.org/10.5281/zenodo.23191444](https://doi.org/10.5281/zenodo.23191444).
It corresponds to GitHub release and tag
[`v1.0.0`](https://github.com/shishirbit/Quantum-Enhanced-Framework-for-Traffic-Speed-Forecasting-with-Calibrated-Uncertainty/releases/tag/v1.0.0).
The release page records the exact commit identifier. The DOI, tag, and release
URL are also recorded in `CITATION.cff`, the repository README, and the
manuscript.
