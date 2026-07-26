# QUARTS-RAC validation-only exploratory plan

QUARTS-UQ showed that an entangled scale head can consistently edge a Fourier
control, but not a separable VQC. QUARTS-RAC changes the quantum module's role
from direct scale regression to selection among shared uncertainty regimes.
The expert bank, output intervals, optimization objective, and inputs are held
constant across the learned controls.

The frozen STAEformer point forecast is never changed. Context combines the
12-horizon forecast, recent local statistics, graph-neighbour aggregates,
local-neighbour disagreement, network-wide regime summaries, cyclical time
features, and a fixed random projection of the frozen backbone representation.
Three ordered central intervals are trained directly with weighted interval
score and recalibrated with split conformal factors.

No canonical METR-LA test examples participate in this pilot. The chronological
validation period is divided into 40% early-stopping, 30% conformal calibration,
and 30% held-out evaluation segments. Results from the final segment may only
decide whether the unchanged design deserves prospective evaluation on a fresh
dataset.

Promotion requires at least 5% lower mean conformal WIS than both Fourier and
separable gates, 89--91% coverage for the nominal 90% interval, at least two of
three paired seed wins, and an exactly unchanged point forecast. Failure stops
this architecture; it must not trigger a search over the already-observed
METR-LA or PEMS-BAY test sets.
