"""Render a model card from completed local evidence; no invented measurements."""
import json

import pandas as pd

from taxiout.artifacts import read_json
from taxiout.config import ROOT


def main():
    release = read_json(ROOT / "reports/release_manifest.json")
    comparison = read_json(ROOT / "reports/comparison.json")
    freeze = release["freeze"]
    candidate = release["candidate"]
    report = comparison["candidates"][candidate]
    final = read_json(ROOT / "models" / release["run_id"] / "manifest.json")
    config = final["config"]
    lines = ["# Model Card", "", f"Selected candidate: `{candidate}`. Final run: `{release['run_id']}`.", "",
             "## Purpose And Information Boundary", "",
             "Retrospective total departure taxi-out in seconds: actual take-off minus airport off-block.",
             "All features are derived from the officially supplied ranking-visible observations. Airport",
             "block times and taxi-time labels are removed before feature generation for both phases.",
             "IDs are join/grouping keys only. Arrival movement events supply traffic context. No flight",
             "rotation inference, external weather/layout, leaderboard feedback or target reconstruction",
             "from hidden fields is used. NM AOBT is a noisy observed clock, never replaced with airport AOBT.", "",
             "## Training And Selection", "",
             f"Final component rows: direct {final['training']['direct']['rows']:,}, residual {final['training']['residual']['rows']:,}, missing specialist {final['training']['missing']['rows']:,}; all labels are raw 2025 departures.",
             "The direct/residual fit is capped at 250,000 ID-stably sampled rows after a full-data",
             "refit exceeded the resource limit. A retained missing-proxy specialist uses its whole",
             "eligible small cohort; per-component counts and ID hashes are in the final manifest.",
             f"Frozen component iterations: `{json.dumps(final['iterations'], sort_keys=True)}`.",
             "The final tree-count rule is the median of F1/F2/F3 selected counts, fixed before C1.",
             "Each development fold has separate initial fit, tuning, refit and score intervals. Full",
             "score cohorts are retained, including negative and extreme labels. There is no output cap",
             "or nonnegative clamp. Serialization uses nearest-even rounding and checked int32 conversion.", "",
             "F1 scores July, F2 October and F3 November. G1 scores October after a six-month gap and",
             "overlaps F2, so it is not pooled. C1 scores December and was previously exposed by the",
             "broad November/December diagnostic: this is reused confirmation, not an independent holdout.", "",
             "| Fold | Rows | RMSE seconds | NM-available RMSE | Missing-proxy RMSE | Missing-proxy SSE share |",
             "|---|---:|---:|---:|---:|---:|"]
    for fold in ["F1", "F2", "F3", "G1", "C1"]:
        if fold == "C1":
            metrics = release["confirmation"]["metrics"]
        else:
            run_id = report["run_ids"][fold]
            metrics = read_json(ROOT / "models" / run_id / "metrics.json")
        whole = metrics["overall"]
        missing = metrics["slices"]["proxy_status"]["missing"]
        available_rmse = ((whole["sse"] - missing["sse"]) / (whole["n"] - missing["n"])) ** .5
        lines.append(f"| {fold} | {whole['n']:,} | {whole['rmse_sec']:.3f} | {available_rmse:.3f} | {missing['rmse_sec']:.3f} | {missing['sse_share']:.1%} |")
    lines += ["", f"Development seasonal proxy: **{report['season_score']:.3f} seconds RMSE**.",
              "Weights are the verified January/July 2026 template proportions; this is not a claim",
              "that November perfectly represents January. Only the 2026 evaluation labels are hidden.", "",
              "## Method And Evidence", "",
              "The retained model uses raw categorical location/aircraft/route fields, calendar and clock",
              "differences as configured in `configs/release.yaml`.",
              ("Production traffic uses exact UTC month-isolated [t-window,t) windows, excludes ties and includes boundary coverage."
               if config["features"]["traffic"] else "Traffic/coverage features were removed following the refitted ablation; no current-event count is retained."),
              "Category tokens are collision-safe. Custom target means are standalone baselines, not",
              "in-sample predictor features. CatBoost learns categorical statistics from fit rows only.", "",
              "The selected residual component learns target minus the NM proxy on 0-7200-second",
              "observed proxies. Its correction is also applied to finite positive proxies above that",
              "range. Negative proxies use the shared direct model; missing proxies use the dedicated",
              "all-cohort specialist. Quality flags remain separate from these deterministic routes.",
              "No learned blend weights or score-label routing is used. See `comparison.json`,",
              "`selection_decision.md` and `experiments.csv` for measured comparisons and negative findings.", "",
              "## Limitations", "",
              "Only one labelled year is available. There are no surface trajectories, aircraft registrations,",
              "exact physical queues, de-icing labels or disruption causes. Rare extreme labels dominate",
              "some error slices, particularly missing NM matches and Rome observations. Timestamp identity",
              "does not establish operational correctness. Airport-day bootstrap/leave-day-out diagnostics",
              "are stability checks, not guarantees about 2026 generalization or causal airport congestion.", "",
              "Long-clock correction extrapolates the fitted residual beyond its usual proxy range.",
              "Much of its July benefit comes from one extreme day. The selected routing variant retains",
              "a small July benefit after removing any single day, but bootstrap intervals include zero",
              "and 2026 transfer remains uncertain. See the recorded paired sensitivity results.", "",
              "One arrival has no taxi-in label/block time and is retained only as an observed movement.",
              "Six near-boundary events are stored in the next monthly file; actual UTC month controls",
              "folds and context. Raw files are unchanged. No external datasets are used. No fuel/CO2 savings",
              "or avoidable-delay claim follows directly from these predictions.", "",
              "## Reproduction And Release", "",
              f"Final fit runtime: {final['runtime_sec']:.1f} seconds; peak process RSS: {final['peak_rss_bytes'] / 1024**3:.2f} GiB.",
              "Four model threads and one substantial training process were used. The resource pilot and",
              "per-run manifests record estimates, actual times and available-memory checks.", "",
              "Run `.venv/Scripts/python -m taxiout.cli reproduce --config configs/release.yaml --offline`",
              "after installing the locked environment and supplying the hashed raw pack. See README for setup.",
              "Saved-model and repeated inference tolerance is 1e-9 seconds on the tested platform;",
              "cross-platform bitwise equivalence is not promised. Serialized outputs have an exact SHA-256.", "",
              f"Prepared submission: {release['prediction_rows']:,} rows, SHA-256 `{release['submission']['sha256']}`.",
              "This is a local prepared artifact; competition upload acceptance remains pending.",
              "The repository README describes source publication and reproduction. Source is GPLv3; raw",
              "challenge data is excluded. Model/prediction redistribution terms are not presumed."]
    (ROOT / "reports/model_card.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
