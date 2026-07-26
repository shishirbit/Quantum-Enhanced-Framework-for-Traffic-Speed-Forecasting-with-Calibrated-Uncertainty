# RAQ-STAEformer validation-only exploratory plan

RAQ-STAEformer tests whether asymmetric predictive intervals improve the
successful classical Fourier-RAC architecture. The frozen STAEformer forecast
remains the distribution median and is never corrected. Separate ordered lower
and upper radii represent skewed uncertainty during congestion, incident-like
transitions, and recovery periods.

The experiment reuses only the QUARTS-RAC training and validation feature
cache. The chronological validation period remains divided into 40% early
stopping, 30% conformal calibration, and 30% held-out evaluation. The canonical
METR-LA test split is not loaded.

The full model uses a three-expert Fourier regime gate and direct weighted
interval-score training. Required ablations remove regime gating, restrict
inputs to the former local feature set, remove graph-neighbour features, and
remove the projected STAEformer representation. The frozen symmetric
Fourier-RAC results are the primary paired reference.

Promotion requires at least 3% lower mean conformal WIS than symmetric
Fourier-RAC, nominal 90% coverage between 89% and 91%, a lower 90% interval
score in at least two of three paired seeds, an unchanged point forecast, and
continued non-use of the canonical test split. Failure ends this asymmetric
branch; success permits one frozen prospective evaluation.
