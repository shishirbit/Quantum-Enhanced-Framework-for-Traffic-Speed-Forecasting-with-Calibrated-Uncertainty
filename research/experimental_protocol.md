# Frozen experimental protocol (v0.2, executable profile)

The machine-executable profile in `configs/full_experiment.yaml` is authoritative
for confirmatory runs. Version 0.1 described an aspirational maximum grid. The
resource-bounded values below were already encoded in the run manifests before
matched residual and quantum experiments began; the documentation discrepancy
is recorded as D010 rather than hidden or corrected retroactively.

## Executed resource-bounded profile

- Maximum 30 epochs, patience 7, minimum validation-NLL improvement 0.0005.
- Batch size 256, Adam learning rate 0.001, weight decay 0.0001.
- Reference quantum model: 4 qubits and depth 2.
- Sensitivity grid: q in {2,4,6} and depth in {1,2,3}, one fixed seed per cell.
- Five confirmatory seeds for every matched reference model and STAEformer.
- The broader q=8/depth=4 grid, 100-epoch budget, target adaptation, and real
  hardware remain follow-up experiments, not completed confirmatory evidence.

## Data and split

- METR-LA primary; PEMS-BAY external validation.
- 12 historical five-minute observations -> 12 future observations.
- Chronological 70/10/20 split on timestamps before windowing.
- Train-only mean/std; masks retained; no test-derived preprocessing.
- Primary horizons: steps 3, 6, 12 (15, 30, 60 minutes); also report all-step
  average.

## Models

At minimum: persistence, historical average, GRU, DCRNN, Graph WaveNet,
STAEformer, backbone-only, backbone+MLP, backbone+Fourier, backbone+separable
VQC, and QUARTS. Every residual model uses the same frozen or jointly trained
backbone according to a single declared regime. Parameter count, FLOPs where
meaningful, wall time and peak memory are reported.

## Tuning

- Common validation metric: masked MAE; probabilistic heads break ties by NLL.
- Equal number of optimization trials for the four matched residual models;
  trainable parameter counts must be within 10%, with exact counts reported.
- Learning-rate, hidden width and regularization grids are declared before test.
- Early stopping uses validation only under the executed 30-epoch/patience-7
  profile above.
- Seeds: 17, 42, 73, 101, 202. No failed numerical run is silently dropped.

## Robustness and shift

1. Missing completely at random: 10%, 30%, 50% observed-input deletion.
2. Block sensor outage: contiguous 30/60/120-minute gaps for 10% sensors.
3. Gaussian measurement noise: 1%, 5%, 10% of training standard deviation.
4. Peak/non-peak, weekday/weekend and chronologically late test slices.
5. Cross-network: train METR-LA; evaluate zero-shot compatible shared modules on
   PEMS-BAY, then 1%, 5%, 10% target adaptation.

Corruption is generated once per seed and shared by every model.

## Quantum experiments

- Qubits q in {2,4,6}; depth L in {1,2,3} in the executed sensitivity grid.
- Entangled ring versus no entanglers.
- Angle re-uploading versus single upload.
- Analytic expectations and {128,512,2048} shots.
- Depolarizing probabilities {0,0.001,0.005,0.01}; amplitude damping
  {0,0.001,0.005,0.01}.
- Record raw and transpiled depth, one/two-qubit gate counts, circuit calls,
  shots and backend/calibration timestamp.
- Real hardware, if available, is an inference-only stratified subset selected
  before observing model error.

## Metrics

Point: MAE, RMSE, masked MAPE with explicit denominator threshold, WAPE.
Probabilistic: Laplace NLL, empirical 90% coverage, mean interval width,
interval score and calibration error. Resources: parameters, train/inference
time, peak CPU/GPU memory, circuit evaluations, shots and transpiled depth.

## Statistical analysis

- Primary comparison: QUARTS vs classical Fourier residual.
- Unit-level paired errors are aggregated first by sensor-day to reduce serial
  pseudoreplication.
- Across seeds/datasets/horizons: Friedman test when comparing >2 models,
  followed by Holm-corrected pairwise Wilcoxon signed-rank tests.
- Report median paired difference, 95% block-bootstrap CI and Cliff's delta.
- Paired t-test is secondary only if difference normality is defensible.
- Alpha=0.05, two-sided. Practical thresholds are declared before test:
  >=1% relative MAE improvement or >=5% relative calibration-error improvement,
  unless runtime is >10x worse, in which case the trade-off is explicit.

## Stopping and integrity

Do not stop because a preferred model wins. Stop after the complete predeclared
matrix or a documented compute/safety failure. Preserve every config, log,
checkpoint hash and prediction artifact. Exploratory modifications use new run
groups and never overwrite confirmatory outputs.
