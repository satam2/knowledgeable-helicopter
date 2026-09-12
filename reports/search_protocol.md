# Declared Development Protocol

Recorded before production model score inspection on 2026-09-10.

1. Reproduce the legacy 100k diagnostic and measure resources.
2. Retain an airport-mean anytime artifact with all required ranking IDs.
3. Evaluate raw-loss direct and residual-with-direct-fallback on F1/F2/F3.
4. Compare shared versus missing-proxy specialist only if the pilot supports full fitting.
5. Bounded optional candidates: clock quality/local calendar and Huber loss.
6. Refit key clock and traffic ablations, inspect largest-error airport-days.
7. Run G1 for the shortlisted recipe(s); use paired airport-day bootstrap and
   leave-one-day-out sensitivity for promotion decisions.
8. Freeze configuration and median F1/F2/F3 tree counts, run C1 once, then final fit.

Initial ceiling: twelve distinct configurations. Required direct/residual use
all eligible rows when the pilot and live 2 GB memory reserve permit it. Optional
screening may use 100k training rows but always scores complete fold cohorts;
such runs are labelled explicitly. No label cleaning, output clipping, learned
blend, inferred aircraft rotations, ranking-derived thresholds or external data.

Promotion needs at least max(2 seconds, 0.5%) lower seasonal score. Neither F1
nor F3 may worsen by more than max(5 seconds, 2%). Missing-proxy RMSE must not
worsen by more than 5% without an explicit reason and champion decision. F2/G1
regressions need investigation. Stability diagnostics cannot establish 2026
generalization. If complexity fails these gates, retain the simpler direct model.

The season score uses the verified January/July ranking proportions and squared
errors. G1 overlaps October in F2 and is never pooled with development folds.
C1 is reused confirmation, not a pristine holdout. No public leaderboard score
is used to guide any experiment.

## Resource Adjustment Before Production Scoring

The initial full-data F1 run `direct-F1-20260910T180638084531-fb902d89`
completed tuning on 822,377 rows in 561.8 seconds (239 selected trees), then
its 1,005,519-row refit exceeded the configured 30-minute per-fit limit.
It produced no qualified model or July model score. The tool host also stalled
during this period; the exact contribution of host scheduling versus model
cost is unresolved. The failed run and its full-data baseline predictions remain.

The new declared production bound is 250,000 ID-stably sampled fit/refit rows,
180 maximum trees, four threads and the same 30-minute/2 GB resource guards.
All score cohorts remain complete. This resource decision was made before any
production model score was inspected. Final reproduction uses the same documented
sample bound; it must not be described as a successful all-label model fit.
The small dedicated missing-proxy model uses every eligible missing-proxy row
instead of sampling that rare cohort. Its component row count and ID hash are
recorded separately. The shared direct and residual models retain the 250k cap.

## Long-Proxy Development Candidate

After the first bounded direct F1 score (583.742 seconds), inspection found one
largest-error observation whose approximately day-long NM proxy closely matched
its raw label. This motivates `residual_all_finite.yaml`, which fits corrections
on every finite supplied proxy and falls back for missing proxies. Its finite
bounds cover the entire supported timestamp range; no day-shift correction or
label/output cap is applied. This is a new development candidate informed by F1,
not an independent post-hoc improvement claim. Evaluate F1/F2/F3/G1 and the same
promotion checks before selection. Direct fallback training is identical and may
be reused despite differing inference-only proxy routing thresholds.

## Disjoint-Route Combination

The dedicated missing-proxy model's F1 RMSE on its route fell from 3186.77 to
1531.60 seconds, while residual correction independently improved the observable
proxy route on all seasonal folds. Two bounded configurations may combine these
models using the fixed observed missingness rule: `residual_specialist` and
`residual_all_finite_specialist`. There are no learned blend weights or error-based
row routes. Reuse only compatible components from the same fit/tune/refit fold,
record all component run IDs, and evaluate complete score cohorts before promotion.
Confirmation trains the same component procedure using only its declared fit and
tuning periods. Final component iterations are frozen before confirmation.

## Isolating Long-Clock Routing

The all-finite combined candidate's F1 improvement disappears when 15 July is
removed, and F3 regresses 1.958 seconds. Before confirmation, add one narrower
candidate: `residual_long_proxy_specialist`. It reuses the validated residual
model fitted on 0-7200-second proxies and applies its correction to finite
positive proxies above 7200 seconds as well. Negative clocks retain the direct
fallback. Missing clocks retain the all-cohort specialist. No new model weights
or routing thresholds are fitted to score labels; this is a logged development
choice motivated by the preceding diagnostics. Test the same complete folds and G1.

Keep the initial development ceiling at twelve configurations (including the
failed full-data attempt). Quality and robust-loss comparisons use the existing
250k direct reference, so an additional 100k reference is unnecessary. Run key
feature ablations on the selected architecture; keep all failed alternatives.

The provisional champion after F1/F2/F3/G1 is the narrower long-proxy combined
recipe (season score 331.176s, G1 285.575s). Clock-feature and traffic ablations
are refitted on this architecture using the same sampling/folds and component
procedures. The clock-feature ablation retains the structural NM proxy in the
residual equation and removes the model's supplemental clock/schedule inputs;
the separate legacy diagnostic provides the complete direct-model clock ablation.
Quality/local-calendar and Huber-loss alternatives are screened against the
same 250k shared direct reference on F1/F3. Noncompetitive screens do not need F2.

The first Huber run used CatBoost's default Newton leaf estimation and produced
an unstable three-tree fit (F1 RMSE 19609.445s). Retain this failed configuration
and test explicit Gradient leaf estimation under a new config hash. This consumes
one search slot: keep the twelve-configuration ceiling by using the already
reproduced direct-model clock ablation as clock evidence and refit the selected
architecture's traffic ablation. Do not present the unrun selected-architecture
clock-input configuration as measured evidence.

Before scoring the selected-architecture traffic ablation: a version without
traffic may replace the provisional champion if its season score is no worse,
F1/F3 and missing-proxy regression gates pass, and full F2/G1 checks are acceptable.
The two-second minimum applies to added complexity; removing seven traffic and
coverage inputs does not require that minimum if the measured seasonal score is
preserved. Any positive seasonal regression retains the current champion.
