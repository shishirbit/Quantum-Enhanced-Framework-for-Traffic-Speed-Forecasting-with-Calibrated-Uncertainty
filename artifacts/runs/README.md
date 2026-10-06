# Run-level archival records

This directory contains the lightweight JSON and CSV records for the reported
experiments. Each retained record identifies the dataset, split, seed, model
treatment, configuration, metrics, timing, and completion state when provided
by the corresponding runner.

Large binary files are deliberately excluded from the permanent software
archive:

* `*.npz` per-forecast prediction arrays;
* `*.pt` and `*.pth` trained checkpoints; and
* execution logs and cached tensors.

These exclusions reduce an otherwise 11.6 GB run directory to the auditable
manifests and metrics required to trace the publication results. Aggregate
statistics, decision records, tables, figures, and integrity manifests remain
under `artifacts/analysis/` and `artifacts/publication/`.
