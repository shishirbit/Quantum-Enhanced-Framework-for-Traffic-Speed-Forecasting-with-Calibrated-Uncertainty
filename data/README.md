# Dataset placement and provenance

## Primary: METR-LA

- Origin: Los Angeles County freeway loop-detector speeds.
- Canonical benchmark packaging: DCRNN (`liyaguang/DCRNN`).
- Expected shape: 34,272 five-minute observations x 207 sensors.
- Raw benchmark file: `data/raw/metr_la/metr-la.h5` or `.npz` with a `data`
  array shaped `[time, node, feature]`.
- Graph: place `adj_mx.pkl` or a numeric adjacency matrix in the same folder.

## External validation: PEMS-BAY

- Origin: Caltrans Performance Measurement System.
- Canonical benchmark packaging: DCRNN.
- Expected shape: 52,116 five-minute observations x 325 sensors.
- Raw benchmark file: `data/raw/pems_bay/pems-bay.h5` or `.npz`.

Caltrans requires a free account for direct PeMS access. Do not redistribute a
copy unless its terms permit that redistribution. Record file hashes in every
run. The experiment code never silently downloads an unversioned mirror.

## Integrity rule

Chronological train/validation/test splits (70/10/20) are created on the raw
timeline *before* forecasting windows are formed. Scaling statistics are fitted
on training data only. Artificial missingness/noise is injected after the clean
split and is accompanied by an explicit mask.

