# Phase-gated research dossier

**Working title:** QUARTS: A NISQ-aware, uncertainty-calibrated quantum residual
benchmark for robust network-wide traffic forecasting

**Evidence cut-off:** 16 July 2026. Sources were checked at publisher,
conference, government, official repository, or original preprint pages. Search
results are leads, not evidence, until represented by a verified source in
`references.bib`.

## Executive research decision

The most defensible contribution is not another large architecture labelled
"quantum". It is a controlled, reproducible test of a precise claim left open by
the closest 2025 study: whether a small data-reuploading VQC, used as a shared
local residual and uncertainty calibrator, offers a useful trade-off on
network-wide, multi-input/multi-output traffic forecasting under missingness,
noise, shift, and cross-network transfer.

The proposed quantum block is meaningful because it computes the nonlinear
residual mean and scale from compressed spatiotemporal states. It is not the
graph encoder, and its qubit count does not grow with the road network. Its
benefit is *not assumed*: an equally tuned classical Fourier residual, a generic
MLP, a separable VQC, and the backbone alone are mandatory controls.

## Phase 1 -- Domain exploration

**Objective.** Find transportation problems with a defensible quantum role.

**Method.** Screened prediction, control, routing, EV, graph learning,
uncertainty, and hybrid optimization; contrasted the density/maturity of each
area using the 2025 quantum-annealing transportation review
\cite{mohammed2025annealingreview}, the 2024 USDOT workshop report
\cite{usdot2024quantum}, current quantum traffic forecasting papers, and
classical traffic forecasting literature.

**Evidence and findings.** Routing and scheduling map naturally to QUBO but are
already the dominant subfield (56 of 73 papers in the 2025 review). Traffic
forecasting has direct public need and mature benchmarks but only limited direct
QML evidence. The closest peer-reviewed data-reuploading study uses one Athens
detector, one-step forecasting, and classical simulation. It explicitly calls
for multi-input/multi-output tasks, generalization, and transfer
\cite{schetakis2025trafficqnn}. A second 2025
paper combines many graph concepts but calls its mechanism quantum-inspired,
does not expose an executable gate-level resource analysis, provides data only
on request, and has no located code-availability statement
\cite{rajagopal2025mthqgnn}.

**Decision.** Select robust probabilistic network-wide speed forecasting. Avoid
route optimization because novelty would be harder to defend and real quantum
scale is still tiny.

**Assumptions/limitations.** The search is structured but not yet a PRISMA-grade
systematic review. Scopus/Web of Science export and dual-review screening remain
needed before submission.

**Risk.** A 2026 paper could close part of the gap. The claim must be rerun just
before submission.

**Next action/files.** See `literature_matrix.csv`, `references.bib` and Phase 4.

## Phase 2 -- Dataset identification and evaluation

**Objective.** Choose benchmark-quality data that test accuracy, robustness and
cross-network transfer without changing the task.

**Method.** Compared METR-LA, PEMS-BAY, LargeST, PeMSD4 and the Athens quantum
study record on cadence, graph, size, variables, access, prior baselines,
quality, and quantum simulation cost.

**Decision.** METR-LA is primary; PEMS-BAY is external validation; Athens is a
reproduction-only dataset; LargeST is a late scalability test. METR-LA and
PEMS-BAY use the same speed target and five-minute cadence while changing node
count and metropolitan network. This isolates transfer better than mixing speed
and flow tasks.

**Data facts.** The downloaded canonical DCRNN file has 207 METR-LA sensors and
34,272 time steps from 1 March through 27 June 2012. PEMS-BAY has 325 sensors
and 52,116 time steps from 1 January through 30 June 2017. Both use 12 past
steps to predict 12 future steps.
LargeST covers up to 8,600 sensors and 525,888 five-minute intervals across five
years. Caltrans states PeMS collects real-time data from nearly 40,000 detectors
and requires a free approved account \cite{caltranspems}.

**Quality caveat.** Published missing ratios depend on whether zeros are treated
as missing and on the packaged file. The chosen file's hash and observed mask
rate, not a copied literature number, will be authoritative in experiments.

**Preprocessing.** Chronological 70/10/20 split; fit scaling on train only;
construct windows within each split; preserve masks; declare interpolation;
augment with time-of-day/day-of-week only; never impute test targets for scoring.

**Files.** `dataset_matrix.csv`, `../data/README.md`.

## Phase 3 -- Deep literature review

**Objective.** Situate the work across statistical, ML, graph, transformer,
QML, uncertainty, efficiency and reproducibility dimensions.

**Method.** Extracted objectives, datasets, method, baselines, metrics, findings,
limitations, reproducibility, unresolved issues and relevance. Foundational
DCRNN/STGCN/Graph WaveNet are retained because they define current protocols;
PDFormer, STAEformer and TESTAM cover modern dynamic/transformer/expert models
\cite{jiang2023pdformer,liu2023staeformer,lee2024testam}.

**Synthesis.** Classical models have progressed from fixed graph convolutions to
learned graphs, adaptive embeddings, delayed propagation and mixtures of
experts. These gains make weak quantum comparisons unacceptable. QML theory
shows data encoding controls a Fourier spectrum, but systematic benchmarking
shows classical models frequently win and entanglement can be irrelevant at
today's scales \cite{bowles2026benchmark}. Noise-induced barren plateaus
constrain depth \cite{wang2021noisebp}. Therefore the
only credible experiment is a matched, ablated, noise-aware comparison.

**Limitations.** Reported performance is not copied into a leaderboard because
split, masking and metric conventions differ. Values will be regenerated under
one pipeline or explicitly labelled non-comparable.

**Files.** `literature_matrix.csv`, `references.bib`.

## Phase 4 -- Research gap identification

Scores: novelty (N), feasibility (F), importance (I), resource fit (R), and
publication potential (P), each 1--5. Ranking is provisional until a formal
systematic search is complete.

| Rank | Gap | Type | N/F/I/R/P | Evidence-based interpretation |
|---:|---|---|---|---|
| 1 | No controlled network-wide, multi-horizon QML traffic benchmark with parameter-matched classical nonlinear controls | methodological/reproducibility | 5/4/5/4/5 | Closest direct study is single-detector and one-step; broad QML evidence demands stronger controls. |
| 2 | Quantum traffic predictors lack calibrated uncertainty and robustness evaluation under missing sensors, observation noise and shift | uncertainty/robustness | 5/4/5/4/5 | Direct study discusses quantum noise but does not run task-level noise/missingness calibration experiments. |
| 3 | Cross-network transfer with qubit count independent of network size is untested | generalization/scalability | 4/4/5/4/5 | Existing direct study has one location; graph-scale quantum encoding is impractical. |
| 4 | Entanglement's contribution is not isolated from Fourier-like data re-uploading | interpretability/methodology | 5/5/4/5/5 | QML benchmark finds separable circuits can match entangled circuits. |
| 5 | Real-device versus ideal/noisy-simulator evidence is scarce for forecasting | deployment | 4/3/4/3/4 | Simulation cost and hardware noise remain central limitations. |
| 6 | Large-network resource scaling is not reported end-to-end | computational efficiency | 3/3/5/3/4 | LargeST exposes unrealistic scale of standard small benchmarks. |

**Selected gap.** A fair, uncertainty-aware and robustness-centred evaluation of
a shared gate-level VQC residual module on standard spatiotemporal graphs.

## Phase 5 -- Final problem statement

### Problem

Given a road graph \(G=(V,E,A)\), historical observations
\(X_{t-P+1:t}\in\mathbb{R}^{P\times N\times F}\), and optional observation mask
\(M\), estimate a predictive distribution for
\(Y_{t+1:t+H}\in\mathbb{R}^{H\times N}\). Current QML traffic evidence does not
establish whether an executable VQC contributes beyond generic Fourier-like
nonlinearity on network-wide, multi-horizon data, or whether any benefit
survives corruption, transfer and quantum noise.

### Proposed direction

A classical spatiotemporal backbone produces a base forecast and compressed
node-local residual state. A shared shallow data-reuploading VQC transforms this
state into a horizon-specific mean correction and positive scale. The identical
backbone is paired with parameter-matched classical, separable-quantum and
no-residual controls.

### Research questions

1. Does the entangled VQC improve test MAE/NLL/calibration over the same backbone?
2. Does it outperform parameter-matched MLP and Fourier residual modules?
3. Are differences larger under missingness, noise or temporal shift?
4. Does a model trained on METR-LA transfer/calibrate on PEMS-BAY after limited
   target adaptation better than controls?
5. How do qubits, depth, shots and hardware noise affect trainability, runtime
   and forecast quality?

### Testable hypotheses

- **H1:** QUARTS has lower paired absolute error than the backbone-only model.
- **H2:** QUARTS has lower NLL or calibration error than a parameter-matched
  classical Fourier residual.
- **H3:** The relative degradation from clean to corrupted data is smaller for
  QUARTS than for matched controls.
- **H4:** Removing entanglers measurably changes performance; otherwise no
  benefit may be attributed to entanglement.
- **H5:** Increasing depth eventually reduces gradient signal under noise.

All null hypotheses are retained unless corrected \(p<0.05\), effect size and
confidence interval support a practically meaningful difference on both seeds
and sensor-time errors.

### Scope and success

Scope is five- to sixty-minute speed forecasting, 4--8 logical qubits, ideal and
noisy simulation, and a small real-hardware inference subset if credentials and
queue permit. Success does not require best MAE: a statistically supported gain
in calibration, robustness, transfer, or parameter efficiency qualifies.

## Phase 6 -- Proposed methodology

### Architecture

1. Mask-aware inputs and temporal covariates.
2. Graph propagation \(\bar X_t=\tilde D^{-1/2}(A+I)\tilde D^{-1/2}X_t\).
3. A shared temporal encoder/backbone produces node state \(h_{t,i}\) and base
   forecast \(\mu^{(0)}_{i,h}\).
4. Linear compression \(z_i=\tanh(W_c h_{t,i}+b_c)\in[-1,1]^q\).
5. Scale angles \(x_i=\pi z_i\) and apply \(L\) re-uploading blocks:
   \[
   |\psi_i\rangle=\prod_{\ell=1}^{L}U_{\rm ent}(\theta_\ell)
   U_{\rm enc}(x_i)|0\rangle^{\otimes q}.
   \]
   Encoding uses per-qubit RY rotations; trainable Rot gates and a ring of CNOTs
   provide local hardware-efficient entanglement.
6. Measure local Pauli-Z expectations \(m_{i,k}=\langle Z_k\rangle\).
7. Produce \(\Delta\mu_{i,1:H}=W_\mu m_i\) and
   \(b_{i,1:H}=\mathrm{softplus}(W_bm_i)+\epsilon\).
8. Final median \(\mu=\mu^{(0)}+\Delta\mu\), with Laplace likelihood.

### Justification

The graph/temporal encoder handles classical structure at realistic scale. The
VQC is shared over nodes, so qubits are \(O(q)\), not \(O(N)\). Data re-uploading
exposes controllable Fourier frequencies and the entanglers test cross-feature
interactions. Local measurements and shallow depth limit barren-plateau risk.

### Loss

\[
\mathcal{L}=\frac1{|\Omega|}\sum_{(i,h)\in\Omega}
\left(\log(2b_{i,h})+\frac{|y_{i,h}-\mu_{i,h}|}{b_{i,h}}\right)
+\lambda\|\theta\|_2^2.
\]

Only observed targets are in \(\Omega\). A quantile/pinball alternative is a
predeclared sensitivity analysis.

### Resources and complexity

The classical graph encoder cost depends on backbone, typically
\(O(BP|E|d+BPNd^2)\). Exact statevector VQC simulation costs approximately
\(O(BNL2^q)\) per local evaluation, while hardware execution cost scales with
shots and circuit calls. The initial circuit is q=4, L=2, ring entanglement,
four local Z measurements; q={2,4,6,8}, L={1,2,3,4} are sensitivity settings.

### Noise

Evaluate ideal analytic expectations, finite shots {128,512,2048}, depolarizing
and amplitude-damping channels over a declared grid, and at least one calibrated
backend noise model where available. Training and inference noise are varied
separately. Report transpiled depth, one/two-qubit gates and shot count.

## Phase 7 -- Baselines and state of the art

| Family | Model | Role / fair configuration |
|---|---|---|
| Naive | Historical average, persistence | Same horizons and masks. |
| Statistical | SARIMA/VAR on feasible subsets | No cross-protocol copied scores. |
| ML | Ridge, SVR, LightGBM | Lag features and train-only scaling. |
| Recurrent | GRU/LSTM, DCRNN | Official architecture with common split. |
| Convolutional/GNN | STGCN, Graph WaveNet, AGCRN | Official code/config adapted only at I/O layer. |
| Transformer | PDFormer, STAEformer | Same 12-to-12 task, early stopping and seed budget. |
| Event-aware | TESTAM | Same graph and clean/corruption protocol. |
| Quantum | QNN re-uploading reproduction | Athens setup separately; no graph claim. |
| Matched hybrid | backbone + entangled VQC | Proposed. |
| Critical controls | backbone only; +MLP; +classical Fourier; +separable VQC | Equal tuning budget; report exact parameter counts. |

Official implementations verified by live repository HEAD checks on 2026-07-16:
DCRNN, STGCN_IJCAI-18, Graph-WaveNet, AGCRN, LibCity and STAEformer.

## Phase 8 -- Implementation

The workspace contains an executable scaffold with deterministic synthetic data,
chronological splitting, mask-aware windows, graph-temporal backbone, actual
PennyLane VQC, classical Fourier control, metrics, NLL and statistical tests.
It records configuration, environment and file hashes. Full third-party SOTA
wrappers and dataset downloads are deliberately not silently vendored.

**Risk.** Exact VQC simulation per node is slow. Predeclared engineering options
are node minibatching for calibrator pretraining, caching a frozen backbone,
vectorized circuit broadcast and asynchronous hardware inference. These cannot
change the statistical unit or cherry-pick nodes.

## Phase 9 -- Experimental protocol

See `experimental_protocol.md`. Core settings: horizons 15/30/60 minutes; five
fixed seeds; clean and corrupted tests; METR-LA->PEMS-BAY transfer; ideal/noisy/
finite-shot simulation; resource logs; paired statistics; no tuning on test.

## Phases 10--13 -- Results, statistics, ablation and figures

**Status: not executed.** No results, significance statements or figures are
claimed. Required artifacts are predeclared:

- per-seed prediction parquet/NPZ and immutable config;
- clean/corruption/transfer tables with MAE, RMSE, masked MAPE, WAPE, NLL,
  90% coverage and mean interval width;
- Wilcoxon, bootstrap CI, Holm correction and Cliff's delta;
- ablations of entanglement, encoding, q, depth, shots, likelihood, graph,
  temporal encoder and uncertainty head;
- architecture/circuit diagram, reliability plot, corruption curves, critical
  difference diagram, qubit-depth-resource plot and peak/off-peak error plots.

Negative or inconclusive results remain in the artifact store and paper.

## Phase 14 -- Research readiness assessment

| Criterion | Status | Evidence / risk |
|---|---|---|
| Problem and gap defined | provisional pass | Supported by verified closest studies; systematic database export pending. |
| Distinct contribution | provisional | Distinct as a benchmarked probabilistic residual study, not yet proven novel. |
| Quantum role justified | pass at design level | Explicit mean/scale residual function and matched controls. |
| Fair protocol | pass at protocol level | Frozen splits, seeds, budgets and critical ablations. |
| Reproducible code | partial | Scaffold exists; SOTA wrappers and data hashes pending. |
| Results/statistics/ablation | fail/not run | Mandatory blocking gate. |
| Practical relevance | provisional | Robust forecast distributions are operationally relevant; latency must be measured. |
| Q1 sufficiency | not assessable | Depends on cross-dataset effect sizes, hardware evidence and honest limitations. |

**Strengths.** Falsifiable premise, unusually strong matched controls, explicit
uncertainty/robustness, network-size-independent qubit design.

**Weaknesses.** No experimental evidence yet; simulation bottleneck; same-state
California datasets limit geographic transfer; no theoretical quantum speedup.

**Main publication risk.** The VQC may equal a small classical Fourier module at
far higher runtime. That remains a valid and publishable benchmarking result only
if experiments are comprehensive and interpretation is disciplined.

## Phase 15 -- Provisional manuscript outline

This outline is **not approved** and manuscript prose must not begin yet.

| Section | Purpose and content | Planned elements | Approx. words |
|---|---|---|---:|
| Title/Abstract/Keywords | Precise finding after results; no quantum-advantage wording unless proven | one result sentence; resources and limits | 250 |
| 1 Introduction | ITS need, QML opportunity and benchmarking problem | contributions; RQs | 1,200 |
| 2 Related Work | traffic forecasting; probabilistic robustness; QML/VQC; quantum transportation | literature taxonomy table | 1,800 |
| 3 Gap and Motivation | closest-study comparison and design requirements | evidence table | 700 |
| 4 Problem Formulation | graph, masks, predictive distribution, objectives | notation table; equations | 900 |
| 5 Method | backbone, compression, VQC, outputs, loss, complexity/noise | architecture/circuit; algorithm | 2,000 |
| 6 Experimental Setup | data, splits, baselines, tuning, hardware, metrics, integrity | dataset/baseline tables | 1,800 |
| 7 Results | clean, robustness, transfer, efficiency, quantum noise | main tables/curves | 2,000 |
| 8 Ablation/Statistics | component causality and uncertainty of differences | ablation table; CD plot | 1,200 |
| 9 Discussion | interpretation without causal overclaim; where quantum did/did not help | trade-off figure | 1,000 |
| 10 Practical Implications | latency, deployment and decision use | resource table | 500 |
| 11 Limitations/Threats | simulator, datasets, multiplicity, implementation validity | threat matrix | 700 |
| 12 Conclusion/Future Work | findings supported directly by evidence | no new claims | 400 |
| Declarations | data/code availability, contributions, conflicts | repository/DOIs after release | 250 |
| Supplement | hyperparameters, all seeds, extra plots, negative runs | full configs and tables | unrestricted |

## Phase 16 -- Manuscript writing

**Gate closed.** Per the master prompt, writing begins one section at a time only
after experiments are validated and the outline is reviewed and approved.
