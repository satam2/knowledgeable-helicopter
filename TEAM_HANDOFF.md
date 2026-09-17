# PRC 2026: Current Research Handoff

This branch preserves the implementation and important research context for a
teammate. It replaces the proposed large ZIP handoff. Start here rather than the
historical baseline reproduction section in README.md.

## Current Result

| Recipe | Local development RMSE | Official RMSE |
| --- | ---: | ---: |
| V2 | 292.813346 s | 294.626000 s |
| Accepted V3 | 268.662991 s | 278.014600 s |

All 344,841 required ranking departures were accepted for V3. The official
submission file SHA256 is
`a91b5f24dd4560bd0bc4f2f5297e0eac3dae529976ac57430a62c333cfd70a1f`.
These are the recorded results from September 16, 2026, not a live leaderboard.
No later experiment replaced V3. The target remains below 240 seconds; that
would require approximately 25.5% less MSE than the current official score.

Local development uses repeatedly exposed July/November 2025 folds. Official
evaluation uses January/July 2026. The 9.351609-second difference is not an
isolated estimate of overfitting: evaluation periods and fitted models differ.
All 2025 months have been exposed during development. We have no certified
complete-model score on untouched local data.

## What Is In This Branch

- The active `src/taxiout/` screening implementation and its tests/configuration.
- [Research history and next experiments](docs/handoff/RESEARCH_HISTORY.md).
- [Exact accepted recipe](docs/handoff/V3_RECIPE.json), including routing,
  component weights, fitting counts and score caveats.
- [Dependency and setup notes](docs/handoff/ENVIRONMENT.md), plus CPU/GPU package
  snapshots under `docs/handoff/dependencies/`.
- [Private artifact requirements](docs/handoff/PRIVATE_ARTIFACTS.md), including
  the inherited V2 specialists needed by V3.
- [Later research source snapshot](research_snapshot/README.md): the external
  experiment, feature, fitting, evaluation and verification Python sources.
  `research_snapshot/SOURCE_MANIFEST.json` records original relative paths and
  SHA256 hashes. It includes failed and superseded approaches as history.

Raw data, model weights, feature caches, individual predictions, private run
receipts and the full conversation are not in Git. Dependency pins are not
downloaded wheels. The source snapshot is not a standalone, one-command V3
reproduction package: original runners also bind private manifests and features.
Access to this branch alone does not reproduce the fitted V3 result.

## Accepted Model

Define proxy as actual takeoff minus the observed NM off-block timestamp.
The supervised residual is disagreement between NM and airport off-block
timestamps; this is partly a recording/source-reliability problem, not simply
a physical taxi-congestion problem.

For finite proxy values in [0, 7200], V3 combines LGB387, PLE387, CatBoost225,
TabM225, and LGB449. Other finite proxies use LGB225. Missing proxies use
`0.5625 * V2 + 0.1875 * ExtraTrees + 0.25 * normalized`. LGB225 is still required
even though its weight in the ordinary mixture is zero. Missing-clock V2
specialists also remain required despite being older fitted models.

All new finite-clock fits used 2,062,577 eligible 2025 rows; the new missing-clock
specialists used 22,470. There are 339,377 ordinary, 174 finite-nonordinary, and
5,290 missing-proxy ranking rows. Preserve exact feature order, fitted encoders,
source availability, all rows and raw targets. Serialize in template order with
nearest-even int32 rounding and no clipping.

## Latest Decision And Priorities

The completed robustness workflow separated fitting, stopping, fixed-count
refitting, calibration and evaluation. Its three-expert mixture shrinkage failed
the promotion gates. A historical complete-cohort hybrid scored 275.356259,
6.693268 seconds worse than V3. Missing-clock ExtraTrees leaf20 lost to leaf1 in
all six panels; normalized correction transferred inconsistently. Keep the
improved evaluation machinery and rejected results; retain V3 as the reference.
These comparisons also changed history and expert membership, so they do not
isolate V3's overfit penalty.

1. Investigate observable clock/source reliability with matched controls.
   Distinguish absent records, unreliable valid clocks, timestamp offsets and
   actual taxi variation using only information available at prediction time.
   Do not define an inference route by its observed target or residual.
2. Apply clean chronological calibration/evaluation to the actual V3 expert
   set and exceptional routes. Hold information, history and capacity constant
   when isolating a change. Report seasonal and influential-row dependence.
3. Test January-through-July training emphasis as one separate, predeclared
   hypothesis. It is different from January-plus-July emphasis; neither has
   been validated or applied to the accepted submission.

Do not repeat broad architecture/coefficient searches without an evidence-based
hypothesis. A new scaler or removing extreme training outcomes has no established
path to the required improvement. Rejected experiments are useful evidence.

## Getting Started

1. Read the research history, environment and private-artifact documents.
2. Obtain authorized data/artifacts separately and create the external sibling
   layout in `research_snapshot/README.md`. Never place them in this checkout.
3. Recreate the appropriate environment from the dependency snapshots. Verify
   package consistency and a bounded native replay before training. Cross-machine
   setup and full V3 replay from this branch have not been executed.
4. Preserve the source snapshot. Copy it to the documented external `review_work`
   directory before using historical runners; their relative-root assumptions
   do not match execution inside `research_snapshot/`.
5. Inspect the selected protocol before execution. Some runners require original
   absolute-path relocation and private frozen manifests. Use versioned wrappers
   and fresh output directories; do not bulk-edit historical evidence or run
   upload/finalization scripts as setup commands.

For the core package, use the environment variables documented in
[SCREENING.md](docs/SCREENING.md): `TAXIOUT_RAW_DIR` and
`TAXIOUT_ARTIFACT_ROOT` must be absolute, disjoint paths outside every Git repo.
The root package's old release command targets the historical baseline, not V3.

## Working Rules

- Preserve raw targets, every scored row, failed attempts and original manifests.
- Separate stopping, refit, calibration and evaluation labels; fit preprocessing
  on the permitted training prefix. New chronology does not undo prior exposure.
- Supplied retrospective batch covariates and chronological label isolation are
  different constraints. Do not claim real-time forecasting from the latter.
- For multi-step work, build a dependency DAG and delegate independent work with
  exclusive ownership. The main agent integrates and verifies. Use two CPU threads
  per fitting job, one heavy GPU job at a time and explicit memory reservations.
- Report predictive evidence separately from engineering/test verification.
  Keep gains and failed gates visible; never access hidden organizer labels.
- Keep credentials, private observations, fitted binaries and row predictions
  outside Git. This handoff does not perform a new submission or publication.

The earlier large-archive preparation files remain outside the repository. No
source data, fitted model or historical experiment was deleted for this handoff.
