# Private Files Needed Separately

Paths below are relative to an external workspace root, not this repository.
No data or fitted weights are committed. Obtain authorized inputs through the
challenge or the project owner. Git source history does not include past private
training outputs, and a clone alone cannot load the accepted model.

## Raw Data

`data/09-15-2026-18-55-03_files_list/` contains twelve monthly 2025 training
Parquets, `ranking.parquet` and `submitting.parquet` (14 files). The existing
repository `reports/input_manifest.json` records original input names and hashes.
Preserve arrival observations as well as departures. There are 2,085,047 training
departures and 344,841 ranking departures; ranking departure truth is hidden.

## Current V3 Fitted Assets

- `private_runs/tail240_20260916/final_ordinary/v2/`: the `model.joblib` files in
  `cat225`, `lgb225`, `lgb387`, `lgb449`, and `tabm225`, with their protocols,
  manifests, feature receipts and fit evidence. Native CatBoost/LightGBM exports
  are useful for existing verification scripts.
- `private_runs/tail240_20260916/state/final_ple387/model_v2/fit/model.joblib`:
  the full-year PLE387 model. Preserve the parent protocol and fit manifests.
- `private_runs/tail240_20260916/forensics/final_missing/v1/models/`:
  `forest.joblib`, `normalized.txt`, and `normalized_encoder.joblib`.
- `private_runs/submission_v2/components/missing/model.cbm`, plus
  `components/rome_20260910_classifier/model.cbm`,
  `components/rome_20260910_regressor/model.cbm`, and the corresponding
  classifier/regressor pairs for `20260911` and `20260912` under that same V2 root.
  Preserve `private_runs/submission_v2/rome_means.json` too. These inherited V2
  missing specialists are active dependencies of V3.

The joblib bundles contain encoders/preprocessing and neural parameters; matching
source modules must be importable when loading them. Only load trusted model files.
Older V2 direct/residual/gate models are not part of V3's prediction mixture, but
some old full-V2 preflight scripts still load them. Use the missing-only replay
when verifying just the inherited V3 dependency.

## Feature And Release Evidence

Keep the V3 submission and receipts under
`private_runs/tail240_20260916/final_submission_v3_v2/`, and independent verification
under `private_runs/tail240_20260916/validation/final_submission_best/`.
Required feature sources are enumerated and hash-bound by the individual model
protocols. Useful roots include:

- `private_runs/screening_230/data/interim/`: base audit/feature tables.
- `private_runs/submission_v2/`: ranking base, schedule, metadata, inputs,
  frozen V2 predictions and protocols used by the missing route.
- `private_runs/tail240_20260916/state/final_ple387/ranking_cache_v1/` and
  `ranking_flat8_v1/`: ranking base/conventions/flattened sequence tables.
- `private_runs/breakthrough_20260916/`: batch context, geometry, weather,
  retrospective research and flattened sequence feature sources.
- `private_runs/mechanism_20260916/information/retrospective_v2/`.
- `private_runs/tail240_20260916/forensics/final_missing/v1/`: prepared missing
  query features, metadata and peer features.

This is an artifact map, not a proof that these short directory names form a
complete executable replay bundle. Historical verifiers may inspect training
feature files and `matrix.float32` scratch even when only ranking predictions
are requested. Regenerate missing caches or write a versioned, bounded replay
wrapper. Preserve original hashes and path bindings as evidence; do not edit
private manifests in place to make a verification pass.

## Historical Evidence

Authoritative aggregate source reports remain privately at:

- `output/tail240_20260916/FINAL_SUBMISSION_RESULT.md`
- `output/tail240_20260916/SCORE_GAP_ANALYSIS.md`
- `output/tail240_20260916/PIPELINE_REVIEW.md`
- `output/tail240_20260916/robustness/RESULTS.md`
- `output/tail240_20260916/robustness/tail/chronological/RESULTS.md`
- `output/tail240_20260916/robustness/seasonal_design.md`
- `output/breakthrough_20260916/REPRODUCE_CLOSING.md`

Their important aggregate conclusions are summarized in the repository's
`RESEARCH_HISTORY.md`. Original private receipts and individual predictions
must be requested separately for a full independent numerical audit.
