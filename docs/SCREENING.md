# Private Screening

Historical early-screening plan. Read [the current handoff](../TEAM_HANDOFF.md)
for the accepted V3 recipe, completed later experiments and next research steps.
The pending statuses below are preserved as historical context.

The next target is 230 seconds; 280 is an intermediate milestone. A local
F1/F3 seasonal estimate is not a leaderboard score. The initial batch is four
independent changes on both complete score cohorts, plus baseline reproduction.

Dependency order: external storage and tests -> raw audit and feature caches ->
baseline F1/F3 -> A/B/C/D F1/F3 -> paired diagnostics -> evidence-based next batch.
Independent fits can use separate processes after a resource pilot. Each uses
four threads and keeps the host memory reserve. Never change the score cohort.

Required environment variables:

- `TAXIOUT_RAW_DIR`: absolute path to the original private pack outside Git.
- `TAXIOUT_ARTIFACT_ROOT`: absolute path to an external artifact folder, disjoint
  from the raw pack. All audit, feature, model and screening outputs go there.

Use `taxiout audit` and `taxiout evaluate` with `configs/release.yaml` as the
reference; `configs/screen_{rows,trees,prefix,rome}.yaml` define the screens.
The Rome candidate requires the compatible baseline to have completed first.
It fits a schedule residual on all usable missing-clock training rows, with
airport context, and applies it only to Rome missing-clock rows with a usable
schedule. Labels never choose the route. Full prediction and model files are
private artifacts. Only synthetic tests belong in Git.

The historical scripts for release packaging and publication have not been
migrated for private use. Do not run them. This branch supports screening; the
Rome candidate is blocked from final training pending wider validation.

Promotion requires at least max(2s, 0.5%) seasonal improvement, F1/F3 regressions
no greater than max(5s, 2%), missing-clock RMSE regression no greater than 5%,
and paired leave-one-day-out checks. Before release, run F2/G1 and seed checks.
December has already been exposed. Do not call it an independent holdout.
