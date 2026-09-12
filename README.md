# PRC 2026 Taxi-Out Estimation

Python and CatBoost solution for estimating departure taxi-out time in the
[PRC 2026 Data Challenge](https://prc-data-challenge-2026.netlify.app/).
The pipeline audits the supplied data, builds features, trains and evaluates
models, and creates a validated Parquet submission for all **344,841 departures**.

Taxi-out is actual take-off minus airport off-block time, in seconds. This is
retrospective estimation using the supplied observations, including pushback
and waiting. It is not a live forecast or an estimate of avoidable delay.

## Reports

- [Model card](reports/model_card.md): method, results and limitations.
- [Selection report](reports/selection_decision.md): candidate comparisons and rejected experiments.
- [Reproduction guide](docs/REPRODUCING.md): complete setup, commands, troubleshooting and release details.
- [Publication verification](reports/publication_verification.json): frozen source hashes, tests and exact prediction reproduction.

The selected model is `residual_long_proxy_specialist`. Local seasonal development
RMSE is **331.176 seconds**; December reused confirmation RMSE is **266.406 seconds**.
These are local validation results. Competition upload and acceptance remain pending.

## Install

Use **64-bit Python 3.12**. The verified platform is Windows; commands below use
PowerShell and run from the repository root.

```powershell
git clone https://github.com/satam2/knowledgeable-helicopter.git
cd knowledgeable-helicopter
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest
```

The 13 tests use synthetic data. Training runs on CPU with four threads and a
2 GB available-memory reserve. The recorded machine had approximately 16 GiB RAM.
See the [guide](docs/REPRODUCING.md#install) for platform and resource details.

## Folder structure

```text
.
|-- configs/                 # Model variants, validation folds, frozen recipe
|-- data/                    # Private raw inputs and generated caches
|-- docs/REPRODUCING.md       # Detailed usage and reproduction guide
|-- models/                  # Local fitted models and run manifests
|-- reports/                 # Aggregate results, model card and verification
|-- scripts/                 # Experiment and local release utilities
|-- src/taxiout/              # Python package and command-line interface
|-- submissions/             # Local prediction files and validation records
|-- tests/                   # Synthetic contract and model tests
|-- pyproject.toml
|-- requirements.lock.txt
|-- README.md
`-- LICENSE
```

Raw data, fitted models, caches and individual predictions are excluded from Git.
The public repository contains the source, configuration and aggregate evidence
needed to reproduce the result with your own authorized data copy.
The `data/`, `models/` and `submissions/` directories are local working directories:
create the raw-input folder when adding data; generated directories are created
by the pipeline as needed.

## Prepare the dataset

Obtain the challenge files through the [official data instructions](https://prc-data-challenge-2026.netlify.app/data.html).
Create the local input folder:

```powershell
New-Item -ItemType Directory -Force data/raw/prc
```

Place all twelve monthly 2025 training files, `ranking.parquet` and
`submitting.parquet` in `data/raw/prc/`:

```text
data/raw/prc/
  training_2025-01-01_2025-02-01.parquet
  training_2025-02-01_2025-03-01.parquet
  ... (one file per month)
  training_2025-12-01_2026-01-01.parquet
  ranking.parquet
  submitting.parquet
```

Preserve the original files. [input_manifest.json](reports/input_manifest.json)
lists all 14 exact names and SHA-256 hashes. Keep arrival rows because they supply
traffic context. The reproduction command builds missing preprocessing caches
automatically; no manual notebook execution is required.

## Train and reproduce

To refit the selected recipe and generate predictions from a fresh clone:

```powershell
.\.venv\Scripts\python.exe -m taxiout.cli reproduce --config configs/release.yaml --offline
.\.venv\Scripts\python.exe -m taxiout.cli validate-submission --file submissions/reproduced_candidate.parquet
Get-FileHash submissions/reproduced_candidate.parquet -Algorithm SHA256
```

`--offline` requires dependencies and raw files to be available locally. The
command verifies the frozen inputs and source, rebuilds missing caches, trains
the final models and validates the submission. It uses the committed freeze and
confirmation records; it does not repeat the historical search.

| Output | Location |
|---|---|
| Trained models and preprocessing | `models/reproduction-<timestamp>-<config-hash>/` |
| Unrounded predictions | `data/processed/reproduced_predictions.parquet` |
| Competition-format predictions | `submissions/reproduced_candidate.parquet` |
| Run ID and validation result | `reports/reproduction.json` |

Expected output: **344,841 rows**, `MVT_ID_mvt: double` and `TAXITIME_SEC_mvt: int32`.
The reference SHA-256 is:

```text
823408382b15b41a5b353b219ef7e5ebbfb07e81e8c3c833f3ebf66295c88e13
```

Cold reproduction from an isolated Git checkout and fresh wheel passed all 13
tests and produced this exact hash. The recorded original final fit took 95.4
seconds; rebuilding caches adds time. Training used 250,000 direct rows, 247,127
residual rows and all 22,470 eligible missing-proxy rows after a full-data fit
exceeded the resource limit. Cross-platform bitwise equality is not promised.

## Predict and build a submission

Reuse the model produced by reproduction:

```powershell
$reproduction = Get-Content reports/reproduction.json -Raw | ConvertFrom-Json
$bundle = "models/$($reproduction.run_id)"
.\.venv\Scripts\python.exe -m taxiout.cli predict --bundle $bundle --output data/processed/ranking_predictions.parquet
.\.venv\Scripts\python.exe -m taxiout.cli build-submission --bundle $bundle --output submissions/local_candidate.parquet
.\.venv\Scripts\python.exe -m taxiout.cli validate-submission --file submissions/local_candidate.parquet
```

`predict` saves unrounded predictions and routing metadata. `build-submission`
runs inference from the bundle's ranking input, joins by movement ID, preserves
template order and applies checked nearest-even int32 rounding. Predictions are
not clipped. It does not read the previous `predict` output.

For upload, verify the registered team name, team bucket and next version, then
use `<team-name>_v<integer>.parquet`. Check acceptance separately according to the
[submission instructions](https://prc-data-challenge-2026.netlify.app/ranking.html#submission-instructions).

## Method and validation

The selected model uses 30 categorical, calendar, clock and prior-traffic features.
Airport block times and taxi-time targets are removed before feature generation;
numeric IDs are join/grouping keys only. Traffic windows use prior events within
each actual UTC month. No external datasets or leaderboard feedback are used.

The NM proxy is take-off minus observed `AOBT_3_flt`. A CatBoost residual correction
is added to nonnegative finite proxies, a direct model handles negative proxies,
and a dedicated model handles missing proxies. The residual learner fits on
0-7,200-second proxies; applying its correction above that range is extrapolation.

| Fold | Score period | RMSE (seconds) |
|---|---|---:|
| F1 | July 2025 | 369.370 |
| F2 | October 2025 | 269.313 |
| F3 | November 2025 | 275.712 |
| G1 | October 2025 after a six-month gap | 285.575 |
| C1 | December 2025, reused confirmation | 266.406 |

Each fold separates fitting, iteration tuning, refitting and scoring. The seasonal
score weights F1/F3 mean squared errors using the ranking July/January proportions.
F2 is an additional development check; G1 overlaps its score period. C1 was exposed
in an earlier diagnostic and is not an independent holdout. Rare extreme labels
and missing NM observations remain major limitations. See the
[validation details](docs/REPRODUCING.md#validation-and-results).

## Development

The supported workflow uses `.py` modules and command-line scripts. Jupyter is
not required for training, inference or reproduction.

Use `python -m taxiout.cli --help` with the installed environment for the command
list. `scripts/develop.py` runs sequential, resumable experiments;
`scripts/plot_validation.py` creates figures from local run artifacts. The guide
covers [development experiments](docs/REPRODUCING.md#run-development-experiments),
the [command reference](docs/REPRODUCING.md#command-reference) and
[troubleshooting](docs/REPRODUCING.md#troubleshooting).

## License and acknowledgements

Source is licensed under [GPL-3.0-only](LICENSE). Dependency and diagnostic
attribution is in [THIRD_PARTY.md](THIRD_PARTY.md). Challenge data must be obtained
separately; the source license does not grant data redistribution rights.

The documentation layout takes inspiration from
[Resourceful Quiver's PRC 2025 repository](https://github.com/PRC-Data-Challenge-2025/resourceful-quiver).
