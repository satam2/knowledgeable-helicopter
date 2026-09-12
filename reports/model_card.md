# Model Card

Selected candidate: `residual_long_proxy_specialist`. Final run: `release-20260911T182332345949-6c34f3be`.

## Purpose And Information Boundary

Retrospective total departure taxi-out in seconds: actual take-off minus airport off-block.
All features are derived from the officially supplied ranking-visible observations. Airport
block times and taxi-time labels are removed before feature generation for both phases.
IDs are join/grouping keys only. Arrival movement events supply traffic context. No flight
rotation inference, external weather/layout, leaderboard feedback or target reconstruction
from hidden fields is used. NM AOBT is a noisy observed clock, never replaced with airport AOBT.

## Training And Selection

Final component rows: direct 250,000, residual 247,127, missing specialist 22,470; all labels are raw 2025 departures.
The direct/residual fit is capped at 250,000 ID-stably sampled rows after a full-data
refit exceeded the resource limit. A retained missing-proxy specialist uses its whole
eligible small cohort; per-component counts and ID hashes are in the final manifest.
Frozen component iterations: `{"direct": 178, "missing": 131, "residual": 180}`.
The final tree-count rule is the median of F1/F2/F3 selected counts, fixed before C1.
Each development fold has separate initial fit, tuning, refit and score intervals. Full
score cohorts are retained, including negative and extreme labels. There is no output cap
or nonnegative clamp. Serialization uses nearest-even rounding and checked int32 conversion.

F1 scores July, F2 October and F3 November. G1 scores October after a six-month gap and
overlaps F2, so it is not pooled. C1 scores December and was previously exposed by the
broad November/December diagnostic: this is reused confirmation, not an independent holdout.

| Fold | Rows | RMSE seconds | NM-available RMSE | Missing-proxy RMSE | Missing-proxy SSE share |
|---|---:|---:|---:|---:|---:|
| F1 | 190,713 | 369.370 | 299.753 | 1531.597 | 35.5% |
| F2 | 185,674 | 269.313 | 253.279 | 936.715 | 12.5% |
| F3 | 162,332 | 275.712 | 235.000 | 1595.104 | 28.0% |
| G1 | 185,674 | 285.575 | 261.826 | 1153.409 | 16.8% |
| C1 | 165,677 | 266.406 | 238.059 | 1222.068 | 20.9% |

Development seasonal proxy: **331.176 seconds RMSE**.
Weights are the verified January/July 2026 template proportions; this is not a claim
that November perfectly represents January. Only the 2026 evaluation labels are hidden.

## Method And Evidence

The retained model uses raw categorical location/aircraft/route fields, calendar and clock
differences as configured in `configs/release.yaml`.
Production traffic uses exact UTC month-isolated [t-window,t) windows, excludes ties and includes boundary coverage.
Category tokens are collision-safe. Custom target means are standalone baselines, not
in-sample predictor features. CatBoost learns categorical statistics from fit rows only.

The selected residual component learns target minus the NM proxy on 0-7200-second
observed proxies. Its correction is also applied to finite positive proxies above that
range. Negative proxies use the shared direct model; missing proxies use the dedicated
all-cohort specialist. Quality flags remain separate from these deterministic routes.
No learned blend weights or score-label routing is used. See `comparison.json`,
`selection_decision.md` and `experiments.csv` for measured comparisons and negative findings.

## Limitations

Only one labelled year is available. There are no surface trajectories, aircraft registrations,
exact physical queues, de-icing labels or disruption causes. Rare extreme labels dominate
some error slices, particularly missing NM matches and Rome observations. Timestamp identity
does not establish operational correctness. Airport-day bootstrap/leave-day-out diagnostics
are stability checks, not guarantees about 2026 generalization or causal airport congestion.

Long-clock correction extrapolates the fitted residual beyond its usual proxy range.
Much of its July benefit comes from one extreme day. The selected routing variant retains
a small July benefit after removing any single day, but bootstrap intervals include zero
and 2026 transfer remains uncertain. See the recorded paired sensitivity results.

One arrival has no taxi-in label/block time and is retained only as an observed movement.
Six near-boundary events are stored in the next monthly file; actual UTC month controls
folds and context. Raw files are unchanged. No external datasets are used. No fuel/CO2 savings
or avoidable-delay claim follows directly from these predictions.

## Reproduction And Release

Final fit runtime: 95.4 seconds; peak process RSS: 1.17 GiB.
Four model threads and one substantial training process were used. The resource pilot and
per-run manifests record estimates, actual times and available-memory checks.

Run `.venv/Scripts/python -m taxiout.cli reproduce --config configs/release.yaml --offline`
after installing the locked environment and supplying the hashed raw pack. See README for setup.
Saved-model and repeated inference tolerance is 1e-9 seconds on the tested platform;
cross-platform bitwise equivalence is not promised. Serialized outputs have an exact SHA-256.

Prepared submission: 344,841 rows, SHA-256 `823408382b15b41a5b353b219ef7e5ebbfb07e81e8c3c833f3ebf66295c88e13`.
This is a local prepared artifact; competition upload acceptance remains pending.
The repository README describes source publication and reproduction. Source is GPLv3; raw
challenge data is excluded. Model/prediction redistribution terms are not presumed.
