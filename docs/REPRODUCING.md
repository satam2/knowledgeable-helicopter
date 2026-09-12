# Reproduction and Development Guide

See the [project README](../README.md) for the short setup and training workflow.
Run all commands below from the repository root.

Reproducible Python and CatBoost code for the PRC 2026 Data Challenge. The
pipeline estimates total departure taxi-out time in seconds at the ten airports
in the supplied data: EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH and LTFM.
It covers data auditing, feature generation, temporal validation, model training,
inference and submission validation.

The target is `TAXITIME_SEC_mvt`: actual take-off (`MVT_TIME_UTC_mvt`) minus
airport off-block time (`BLOCK_TIME_UTC_mvt`). This is retrospective estimation
using the observations supplied for ranking. Total taxi-out includes pushback
and waiting; it is not a live forecast or a measure of avoidable congestion.

The selected recipe, `residual_long_proxy_specialist`, produces **344,841**
predictions. Its local seasonal development score is **331.176 seconds RMSE**;
December reused confirmation RMSE is **266.406 seconds**. Competition upload
and acceptance remain pending. These numbers are local validation results.

Source is licensed under **GPL-3.0-only**. Raw challenge data, fitted models,
caches and individual predictions are excluded from this public repository.
You need your own authorized copy of the challenge data to reproduce the model.

## Contents

- [Install](#install)
- [Supply the data](#supply-the-data)
- [Reproduce the selected model](#reproduce-the-selected-model)
- [Use a trained model](#use-a-trained-model)
- [How the model works](#how-the-model-works)
- [Validation and results](#validation-and-results)
- [Run development experiments](#run-development-experiments)
- [Command reference](#command-reference)
- [Repository layout](#repository-layout)
- [Troubleshooting](#troubleshooting)
- [Publication and competition submission](#publication-and-competition-submission)
- [License and attribution](#license-and-attribution)

## Install

The verified environment is **CPython 3.12 on Windows x86_64**, with CPU training.
The package requires Python `>=3.12,<3.13`. The recorded machine had 12 logical
CPUs and approximately 16 GiB RAM. Training uses four threads, one substantial
model process, a 2 GB available-memory reserve and a 30-minute limit per fit.
Allow disk space for the raw pack, environments, feature caches and model runs;
repeated experiments retain their artifacts.

Run the following in PowerShell. All later examples assume the repository root
is the working directory, and invoke the virtual environment directly without
requiring activation.

```powershell
git clone https://github.com/satam2/knowledgeable-helicopter.git
cd knowledgeable-helicopter
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest
```

The 13 tests use synthetic fixtures and do not need challenge data. The lock file
pins runtime, build, plotting and test dependencies. If the Windows `py` launcher
is unavailable, use a Python 3.12 executable for the environment creation step.
On Linux/macOS, the equivalent paths are `python3.12 -m venv .venv` and
`.venv/bin/python`; those platforms have not been validated for exact reproduction.

## Supply the data

Obtain the files through the organizers' [official data access instructions](https://prc-data-challenge-2026.netlify.app/data.html).
Place the twelve monthly 2025 training files, `ranking.parquet` and
`submitting.parquet` in `data/raw/prc/`, preserving their names and contents:

```text
data/raw/prc/
  training_2025-01-01_2025-02-01.parquet
  training_2025-02-01_2025-03-01.parquet
  ... (one file per month)
  training_2025-12-01_2026-01-01.parquet
  ranking.parquet
  submitting.parquet
```

[input_manifest.json](../reports/input_manifest.json) lists all 14 exact filenames,
sizes, schemas and SHA-256 hashes. Training contains 4,167,797 movements, including
2,085,047 departures. Ranking contains 689,534 movements; only its 344,841
departures need predictions. Keep arrivals because their movement times supply
traffic context. The submission template specifies the required IDs and order.

No API credentials, external datasets or absolute machine-specific paths are
needed by the model. The pipeline verifies the input pack against its manifest.
Reproduction requires the original Parquet files, not re-encoded copies.

## Reproduce the selected model

After installing dependencies and supplying the data, run:

```powershell
.\.venv\Scripts\python.exe -m taxiout.cli reproduce --config configs/release.yaml --offline
.\.venv\Scripts\python.exe -m taxiout.cli validate-submission --file submissions/reproduced_candidate.parquet
Get-FileHash submissions/reproduced_candidate.parquet -Algorithm SHA256
```

This is the complete path from a fresh clone to predictions. It verifies the
inputs, rebuilds missing audit and feature caches, fits the frozen recipe,
reloads the model for inference and writes a validated submission. It uses the
committed [freeze](../reports/freeze.json) and [confirmation](../reports/confirmation.json)
records, so historical training bundles are not needed. It does not rerun the
model search or score December again. `--offline` means the pipeline retrieves no
network resources; dependencies and raw files must already be available locally.

| Output | Contents |
|---|---|
| `models/reproduction-<timestamp>-<config-hash>/` | CatBoost models, preprocessing bundle and run manifest |
| `data/interim/` | Rebuilt audit and feature caches |
| `data/processed/reproduced_predictions.parquet` | Unrounded predictions with IDs and routing metadata |
| `submissions/reproduced_candidate.parquet` | Two-column competition-format predictions |
| `submissions/reproduced_candidate.validation.json` | Schema, row count and checksum evidence |
| `reports/reproduction.json` | Run ID and reproduction result |

Expected submission schema and reference hash:

```text
MVT_ID_mvt: double
TAXITIME_SEC_mvt: int32
rows: 344841
sha256: 823408382b15b41a5b353b219ef7e5ebbfb07e81e8c3c833f3ebf66295c88e13
```

On a fresh clone, compare the printed hash to this reference. When the original
local `submissions/local_candidate.parquet` is present, `reproduce` also requires
exact equality of the serialized values and records whether the bytes match.
Cross-platform bitwise equality is not promised.

The recorded final fit took 95.4 seconds with 1.17 GiB peak process RSS. Cold
reproduction takes longer because it also constructs audit and feature caches.
The final models used 250,000 direct rows, 247,127 residual rows and all 22,470
eligible missing-proxy rows. A full-data fit exceeded the resource limit, so the
released model must not be described as trained on every departure label.

The earlier isolated-wheel verification rebuilt empty caches, passed all 13 tests
and reproduced byte-identical Parquet. See [clean_verification.json](../reports/clean_verification.json).
Run IDs, timestamps and environment paths in historical reports describe that
execution. Reproduction generates new run IDs and updates local reports.

The publication check on 2026-09-12 also passed from an isolated Git checkout and
fresh wheel, with no initial caches, models or reference predictions. All 41
frozen file hashes matched, all 13 tests passed, and the rebuilt submission
matched the reference hash. See [publication_verification.json](../reports/publication_verification.json).

## Use a trained model

After reproduction, select the new bundle using the run ID in its report:

```powershell
$reproduction = Get-Content reports/reproduction.json -Raw | ConvertFrom-Json
$bundle = "models/$($reproduction.run_id)"
.\.venv\Scripts\python.exe -m taxiout.cli predict --bundle $bundle --output data/processed/ranking_predictions.parquet
.\.venv\Scripts\python.exe -m taxiout.cli build-submission --bundle $bundle --output submissions/local_candidate.parquet
.\.venv\Scripts\python.exe -m taxiout.cli validate-submission --file submissions/local_candidate.parquet
```

`predict` writes unrounded predictions and diagnostics. `build-submission` runs
inference from the bundle's configured ranking file and joins predictions to the
template by movement ID; it does not consume a previous `predict` output. A fresh
clone does not contain the historical bundle
`models/release-20260911T182332345949-6c34f3be`, which remains a local artifact.

Submission construction requires a one-to-one ID match, preserves template order
and float64 IDs, rounds with `numpy.rint` (nearest even), checks int32 bounds and
reads the saved Parquet back for validation. Missing, extra or duplicate IDs,
schema changes, nonfinite values and overflow cause failure. Predictions are
neither capped nor clamped to zero.

## How the model works

The observation sanitizer removes airport block times and taxi-time targets
before feature generation. Numeric movement and flight IDs serve only as join
or grouping keys. No external data or leaderboard feedback is used.

The selected configuration uses 30 features from airport, runway, stand, route,
aircraft/operator categories, UTC calendar values, observed clock differences
and prior traffic counts. Airport/location combinations use collision-safe
category tokens. CatBoost handles categorical predictors; custom target means
are separate baselines rather than input features.

Traffic counts use `[t-window, t)` intervals within each actual UTC month.
Simultaneous events are excluded, timestamp units are normalized and coverage
features describe incomplete windows at month starts. Arrival movement events
contribute counts without exposing arrival block times or labels.

The NM proxy is take-off time minus the observed `AOBT_3_flt` clock. It is a
noisy observation, not the hidden airport off-block time. Routing is fixed:

| Observed proxy | Prediction |
|---|---|
| Finite, 0 to 7,200 seconds | Proxy plus a CatBoost residual correction |
| Finite, above 7,200 seconds | Proxy plus the same residual correction |
| Finite, negative | Direct CatBoost prediction of total taxi-out |
| Missing/nonfinite | Dedicated CatBoost model for missing-proxy departures |

The residual learner is fitted only on proxies between 0 and 7,200 seconds.
Applying its correction above that range is extrapolation. The direct/residual
training sample is deterministic: movement IDs are sorted before seeded sampling
with a 250,000-row cap. The missing-proxy component uses its entire eligible
cohort. Frozen tree counts are direct 178, residual 180 and missing specialist 131.

## Validation and results

Every fold has separate initial fit, iteration tuning, refit and score intervals.
Intervals are start-inclusive and end-exclusive. Related departure flights
crossing boundaries are purged. Exact dates are in [folds.yaml](../configs/folds.yaml).
All score rows are retained, including extreme and negative supplied labels.

| Fold | Score period | Role | Rows | RMSE (seconds) |
|---|---|---|---:|---:|
| F1 | July 2025 | Summer development | 190,713 | 369.370 |
| F2 | October 2025 | Autumn development | 185,674 | 269.313 |
| F3 | November 2025 | Winter development | 162,332 | 275.712 |
| G1 | October 2025 | Six-month-gap stress check | 185,674 | 285.575 |
| C1 | December 2025 | Reused confirmation | 165,677 | 266.406 |

The seasonal score weights F1 and F3 **mean squared errors**, then takes the
square root. Weights are the ranking departure proportions: July
192,122/344,841 and January 152,719/344,841. F3 is a winter proxy for January;
it is not an observation of January 2026 performance. F2 is required for complete
development evidence but is not in this seasonal formula. G1 overlaps F2's score
period and is reported separately.

C1 was already exposed by an earlier November/December diagnostic, so it is
reused confirmation, not an independent holdout. The model recipe and final
iteration counts were frozen before the confirmation run.

Rare extreme labels dominate some errors, especially when the NM proxy is
missing. Much of the selected long-proxy route's July gain comes from one extreme
day; bootstrap intervals include zero. Only one labelled year is available.
These limits matter when interpreting the local scores or extrapolating to 2026.
Full comparisons, rejected variants and cohort errors are in the
[selection decision](../reports/selection_decision.md), [model card](../reports/model_card.md)
and [experiment table](../reports/experiments.csv).

## Run development experiments

Use a separate working copy for new experiments: development commands update
reports, and the published freeze describes one particular model selection.
For example, to rerun the direct baseline and selected architecture:

```powershell
.\.venv\Scripts\python.exe -m taxiout.cli audit --config configs/base.yaml
.\.venv\Scripts\python.exe -m taxiout.cli make-splits --config configs/folds.yaml
.\.venv\Scripts\python.exe scripts/develop.py --configs configs/direct.yaml configs/residual_long_proxy_specialist.yaml --folds F1 F2 F3
.\.venv\Scripts\python.exe -m taxiout.cli evaluate --config configs/residual_long_proxy_specialist.yaml --fold G1
.\.venv\Scripts\python.exe scripts/summarize_experiments.py
```

`develop.py` runs sequentially and reuses completed runs with matching candidate,
fold, configuration and input hashes. Each run writes its ID to the console and
stores models, metrics, score predictions, training evidence and checksums under
`models/<run-id>/`. Failed or interrupted runs remain marked incomplete.

Use the completed run IDs to compare candidates. Replace the placeholders below
with the actual F1/F2/F3 IDs for each candidate:

```powershell
.\.venv\Scripts\python.exe -m taxiout.cli compare --reference direct --runs DIRECT_F1_ID DIRECT_F2_ID DIRECT_F3_ID SELECTED_F1_ID SELECTED_F2_ID SELECTED_F3_ID
```

Configurations inherit defaults through `extends`; [base.yaml](../configs/base.yaml)
defines the sampling, feature and resource settings. Other configurations cover
residual routing, a missing-proxy specialist, quality features, robust loss and
feature ablations. The exact earlier diagnostic uses
`benchmark --config configs/baseline_reproduction.yaml`; it preserves its original
100,000-row sample and continuous traffic context, unlike production validation.
The `anytime` command creates a simple airport-mean fallback artifact.

For a new release cycle, `freeze` requires completed F1/F2/F3/G1 runs for one
configuration; `confirm --fold C1` then records the confirmation gate, and
`train-final` fits only after that gate passes. The existing freeze and
confirmation are intentionally protected against casual reuse or overwrite.
Do not replace them simply to bypass a checksum or failed gate. The historical
search and resource exception are documented in [search_protocol.md](../reports/search_protocol.md).

## Command reference

Run `.\.venv\Scripts\python.exe -m taxiout.cli --help`, or append `--help` to any
subcommand. The installed `taxiout` entry point exposes the same commands when
the virtual environment is active.

| Command | Purpose and main arguments |
|---|---|
| `audit` | Inspect raw inputs and build labelled caches; `--config` |
| `make-splits` | Materialize temporal fold membership; `--config configs/folds.yaml` |
| `benchmark` | Reproduce the historical diagnostic; `--config configs/baseline_reproduction.yaml` |
| `anytime` | Build an airport-mean fallback; `--config` |
| `evaluate` | Train and score one development/stress fold; `--config`, `--fold` |
| `compare` | Compare completed runs; `--runs`, optional `--reference` |
| `freeze` | Record a new selected recipe; `--config`, `--runs` |
| `confirm` | Run the frozen confirmation procedure; `--config`, `--fold C1` |
| `train-final` | Fit the frozen recipe; `--config configs/release.yaml` |
| `reproduce` | Rebuild and validate the frozen result; `--config configs/release.yaml --offline` |
| `predict` | Save unrounded predictions; `--bundle`, optional `--input`, `--output` |
| `build-submission` | Predict and serialize in template order; `--bundle`, optional `--template`, `--output` |
| `validate-submission` | Check a Parquet against the template; `--file`, optional `--template` |

## Repository layout

| Path | Role |
|---|---|
| `src/taxiout/availability.py`, `schema.py`, `audit.py`, `io.py` | Observation boundary, ID/target contracts, audit and verified input loading |
| `src/taxiout/features/`, `cache.py` | Calendar, clocks, traffic and reproducible feature caches |
| `src/taxiout/models/`, `train.py` | Baselines, CatBoost components and bounded training |
| `src/taxiout/splits.py`, `metrics.py`, `release.py` | Temporal validation, comparisons, freeze and reproduction |
| `src/taxiout/predict.py`, `submission.py`, `cli.py` | Saved-model inference, Parquet validation and CLI |
| `configs/` | Development variants, fold definitions and frozen release settings |
| `tests/` | Synthetic contract and model tests |
| `scripts/` | Development orchestration, summaries, figures and local release utilities |
| `reports/` | Public aggregate metrics, hashes, model card and reproduction evidence |
| `data/`, `models/`, `submissions/`, `dist/` | Local inputs and generated artifacts, excluded from Git; output directories are created as needed |

`scripts/finalize.py --run-id RUN_ID` records local final-model inference parity
and submission evidence. `scripts/plot_validation.py` and
`scripts/make_model_card.py` regenerate summaries from locally available
experiment bundles. Those historical bundles are not included in Git.
`scripts/refresh_rules.py` retrieves official rule pages and needs network access.

`scripts/verify_clean.py` is a maintainer check: it requires an independently
created `.venv-verify` with the locked dependencies, the reference local candidate
and the raw pack on a filesystem supporting hardlinks. It makes an isolated tree
under `data/interim/`, installs a fresh wheel there and reruns tests/reproduction.
The normal fresh-clone workflow is the `reproduce` command above.

## Troubleshooting

| Failure | Resolution |
|---|---|
| Python/package installation rejected | Use 64-bit Python 3.12 and install the lock file before installing this package. |
| `No module named taxiout` | Use the repository's `.venv` interpreter and run the editable install step. |
| Repository root not found | Run from the clone root or a directory beneath it. |
| Missing input or `Input pack changed` | Compare all 14 filenames and hashes to the manifest; restore the original files for frozen reproduction. |
| `Source changed after freeze` | Use an unmodified checkout of the published source. `.gitattributes` preserves checksum-sensitive file bytes. |
| Corrupt/incompatible audit or feature cache | Preserve any needed local diagnostics, move the affected generated cache out of `data/interim/`, then reproduce so it can be rebuilt. |
| Insufficient available memory or fit timeout | Free memory and stop competing workloads; the frozen resource settings are in `configs/release.yaml`. |
| Unknown/incomplete model bundle | Use the complete run ID from the new `reports/reproduction.json`; historical bundles are local only. |
| Prediction hash differs | Check the exact raw pack, frozen source, Python and locked package versions. Compare the validation report; do not overwrite the reference to conceal a mismatch. |

## Publication and competition submission

The source repository is [satam2/knowledgeable-helicopter](https://github.com/satam2/knowledgeable-helicopter).
The README, GPLv3 license, code, configurations, tests and aggregate reproduction
evidence constitute the public source publication. Historical JSON reports record
the local preparation state at their timestamps; fields such as `published` or
`uploaded` in those records are not live GitHub or competition status checks.

The original validated candidate is `submissions/local_candidate.parquet` with
the reference SHA-256 above. Competition delivery still requires verifying the
registered team name, bucket access and next submission version, then using the
organizers' required `<team-name>_v<integer>.parquet` filename and confirming
acceptance. A valid local file or a GitHub push does not establish acceptance.
Use the current [ranking instructions](https://prc-data-challenge-2026.netlify.app/ranking.html)
and [eligibility requirements](https://prc-data-challenge-2026.netlify.app/eligibility.html)
for external delivery; the earlier [rules review](../reports/rules_review.md) is a
dated snapshot.

`.gitignore` excludes raw input directories, Parquet files, fitted models,
generated caches, local archives, environments and credentials. Public reports
retain aggregate statistics and hashes. Individual movement IDs and timestamps
from the six file-boundary anomalies have been removed from the public audit
report; rerunning the frozen audit reconstructs those details locally. Review
regenerated reports before staging or packaging them because they can contain
individual raw records. Never force-add private artifacts.

Transient development progress files, the anytime fallback receipt and new
comparison snapshots also remain local. The retained comparison history contains
distinct experiment evidence; the final comparison is `reports/comparison.json`.

`scripts/package_release.py` builds `dist/prc-2026-source.zip` and a separate
`dist/prc-2026-local-artifacts.zip` from a completed local release in a Git checkout.
The source archive includes only tracked source and report files; stage intended
new files before packaging. Generated progress logs and caches stay excluded.
The local artifacts archive contains models and predictions and is for local use.
These archives and their
historical checksums in [package_validation.json](../reports/package_validation.json)
are separate from the GitHub commit. Regenerate packages after documentation
changes, inspect their contents, and keep raw records excluded before publishing
an archive. Model/prediction redistribution terms are not presumed.

## License and attribution

Project source is licensed under the [GNU General Public License version 3](../LICENSE)
(`GPL-3.0-only`, as declared in `pyproject.toml`). Dependency and reused diagnostic
attribution is in [THIRD_PARTY.md](../THIRD_PARTY.md). Dependencies retain their own
licenses. The source license does not grant redistribution rights to challenge
data, which must be obtained separately through the organizers.
