# QUARTS v2 exploratory improvement plan

QUARTS v1 completed its frozen confirmatory study, but did not beat STAEformer
and did not demonstrate a quantum or entanglement advantage. Those results are
preserved unchanged. QUARTS v2 is a separately labelled exploratory model.

## Model change

QUARTS v2 freezes the official STAEformer checkpoint and learns a lightweight
sensor-local residual and Laplace-scale calibrator. Its input contains the
12-horizon base forecast, four recent-speed statistics, and cyclical
time-of-day/day-of-week features. The residual is bounded to two normalized
target units and multiplied by a learned per-horizon gate initialized near
zero. After training, a horizon is allowed to change the base forecast only if
it reduces validation MAE.

The candidate head remains the four-qubit, depth-two, data-reuploading
entangled VQC. The required controls are an equal-size separable VQC, a
Fourier head, a small MLP, and the unmodified frozen backbone.

## Scientific gate

The pilot is successful only if the entangled VQC lowers test MAE relative to
the frozen backbone and also beats both the Fourier and separable controls.
Lower NLL or better interval calibration is useful secondary evidence, but
cannot replace the point-forecast gate. Test results will not be used to retune
the seed-42 pilot.

Only after this gate is met should the unchanged design be repeated on seeds
73 and 202. A new multi-dataset, five-seed confirmatory matrix is not justified
until that independent exploratory replication succeeds.
