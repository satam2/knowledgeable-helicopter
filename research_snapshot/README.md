# Later Research Source Snapshot

`review_work/` preserves the later external Python research sources byte-for-byte.
`SOURCE_MANIFEST.json` maps each file to its original workspace-relative path,
size and SHA256. These are experimental runners, analyses and tests, including
failed and superseded versions. Their presence is not a recommendation to run all
of them. The accepted recipe and decisions are in `../TEAM_HANDOFF.md`.

Only source is included here. No generated research JSON, report dumps, raw data,
model binaries, row predictions or credentials were copied. The active core
`src/taxiout/` changes are in the normal package tree rather than this snapshot.

## Restore The Historical Layout

Use an outer workspace that is not itself a Git checkout:

```text
workspace/
  knowledgeable-helicopter-screening/  # clone of this branch
    research_snapshot/review_work/    # preserved source snapshot
  review_work/                        # working copy of snapshot sources
  data/                               # authorized private inputs
  private_runs/                       # private manifests/features/models
  output/                             # private generated reports
```

The historical scripts infer their workspace root from their location. Running
them directly inside `research_snapshot/` changes that root and can violate the
external-artifact checks. From the repository root, this PowerShell command
copies the snapshot to a new external sibling without overwriting a directory:

```powershell
$researchDestination = Join-Path (Split-Path (Get-Location).Path -Parent) 'review_work'
if (Test-Path -LiteralPath $researchDestination) { throw 'Use a fresh workspace or compare the existing research sources first.' }
Copy-Item -LiteralPath .\research_snapshot\review_work -Destination $researchDestination -Recurse
```

The default historical checkout name is `knowledgeable-helicopter-screening`.
Several scripts also bind original absolute paths, raw-file locations and private
source hashes. Copying source does not resolve all of these. Inspect the selected
entry point and preserve its original before creating a versioned relocation
wrapper. Some baseline helpers may require the original sibling checkout too.

## Useful Entry Points

- `review_work/campaign_20260916/`: shared contracts and experiment infrastructure.
- `review_work/breakthrough_20260916/models/`: neural/tree adapters and encoders.
- `review_work/tail240_20260916/final_ordinary/`: V3 full-year ordinary training.
- `review_work/tail240_20260916/state/final_ple387/`: PLE387 feature/preparation/fit.
- `review_work/tail240_20260916/forensics/final_missing/`: missing-clock final fit.
- `review_work/tail240_20260916/final_submission/`: V3 composition/report/upload history.
- `review_work/tail240_20260916/validation/final_submission_best/`: independent V3 verification.
- `review_work/tail240_20260916/robustness/`: separated chronological experiments.

Read selected scripts and protocols before invoking them. Upload/finalization
sources are retained as history and are not setup steps. Original private
artifacts are required for most historical commands, and full fresh-clone
reproduction has not been verified for this snapshot. Preserve failed attempts
and versioned outputs; do not convert old directories into new run destinations.
