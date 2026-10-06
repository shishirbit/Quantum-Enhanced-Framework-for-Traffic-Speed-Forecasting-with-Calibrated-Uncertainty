# Controlled Evaluation of Quantum and Classical Residual Calibration for Probabilistic Traffic Forecasting

This repository contains the complete implementation and publication artifacts
for:

> Controlled Evaluation of Quantum and Classical Residual Calibration for
> Probabilistic Traffic Forecasting

QUARTS is a Quantum Uncertainty Aware Residual Traffic System. The study tests
whether an entangled variational quantum circuit improves point forecasts or
predictive intervals when it receives the same inputs and training budget as a
separable circuit, a classical Fourier head, and a multilayer perceptron.

The confirmatory experiments do not support a quantum advantage. Across five
seeds, the entangled head remained substantially behind STAEformer on METR-LA
and PEMS-BAY, and its differences from the matched separable and Fourier heads
were below 0.40 percent. A Fourier regime calibrator produced a small
prospective PeMSD4 interval score gain, but that gain changed sign for one of
three independently trained backbones and failed the aggregate coverage gate.
The PeMSD8 test split was left untouched after validation failed.

## Repository contents

* `src/quarts/` contains models, metrics, data processing, and statistical
  utilities.
* `scripts/` contains resumable experiment, analysis, audit, and figure
  generation commands.
* `configs/` contains immutable experiment configurations.
* `notebooks/01_full_experiment.ipynb` is the Jupyter entry point.
* `research/` contains the experimental protocol, decision log, literature
  records, and dataset audit.
* `artifacts/publication/` contains the final CSV tables, vector figures,
  manifest, and experiment gate summary.
* `artifacts/analysis/` contains statistical summaries and the completion
  audit.
* `manuscript/quarts_springer/` contains the Springer Nature LaTeX source,
  BibTeX database, system architecture, and compiled manuscript.

Raw benchmark data and trained checkpoints are excluded because the public
datasets retain their original distribution routes and the binary outputs are
large. Acquisition instructions and integrity hashes identify the exact inputs
used in the study.

## Environment

The recorded experiment environment used Windows 11, Python 3.12.13, PyTorch
2.13.0 with CUDA 13.0, PennyLane 0.39, NumPy 2.0.2, pandas 2.2.3, SciPy 1.18.0,
and scikit learn 1.9.0. Experiments ran on an NVIDIA GeForce RTX 5060 Laptop
GPU. Quantum circuits used analytic state vector simulation. No physical
quantum hardware was evaluated.

Install the declared Python dependencies and run the smoke test:

```bash
python -m pip install -r requirements.txt
python scripts/run_smoke.py
pytest -q
```

The local Anaconda environment used during the reported runs is named
`research-gpu`. The notebook can be opened from the repository root.

## Reproduce tables and figures

Publication artifacts are regenerated from the completed run ledger:

```bash
python scripts/analyze_results.py
python scripts/build_publication_package.py
python scripts/generate_system_architecture.py
```

The completion record is
`artifacts/analysis/completion_audit.json`. It expects 50 reference runs, 10
official STAEformer runs, and nine circuit ablations. Smoke tests,
preflight runs, and the preprotocol batch size 128 run are excluded.

## Compile the manuscript

The manuscript follows the Springer Nature December 2024 journal article
template. From `manuscript/quarts_springer`, compile `main.tex` with a current
LaTeX distribution or Tectonic. The compiled manuscript is retained as
`manuscript/quarts_springer/build/main.pdf`.

## Data and integrity

Place benchmark files under `data/raw/<dataset>/` and follow
`data/README.md`. The loaders use chronological partitions and calculate
scaling statistics from training observations only. Dataset hashes are stored
with the experiment manifests. PeMSD8 canonical test targets were not
evaluated because the SA-CQR validation gate failed.

## Citation

The permanent version 1.0.1 archive is available from Zenodo at
[https://doi.org/10.5281/zenodo.23191444](https://doi.org/10.5281/zenodo.23191444).
It corresponds to the GitHub release and tag
[`v1.0.1`](https://github.com/shishirbit/Quantum-Enhanced-Framework-for-Traffic-Speed-Forecasting-with-Calibrated-Uncertainty/releases/tag/v1.0.1).
Machine-readable citation metadata are provided in `CITATION.cff`.
