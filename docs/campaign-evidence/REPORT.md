# Campaign score evidence (through September 25)

![Matched F1 gains and separate official ranking scores](campaign_scores.png)

The F1 bars show **control RMSE minus candidate RMSE** on the *same complete 2025 F1 cohort*. Positive values mean the candidate won that specific paired comparison. Each row has its own same-origin comparator; the bar heights are **not** an absolute model-quality ranking. Every plotted F1 arm failed its predeclared advancement gate and stopped before F3, full-year fit, or upload.

The largest paired gain is same-stand versus airport-wide ARR context (+3.260 seconds in July), but it also had to beat the unchanged base, pass the top-ten-row removal, and survive its June/December checks. It failed the frozen F1 gate; July's gain over base was +0.588 seconds, its paired-day MSE interval crossed zero, and its top-ten-row removal reversed the gain. Source-aware learning improved +1.448 seconds against its equal-capacity rich-only comparator in July but *lost* 4.315 seconds against the unchanged clean base. Extra-time weather gained +1.192 seconds in December while losing 0.212 seconds against routine-only weather in July. Route-duration centering gained +0.026834 seconds against its byte-identical raw/clean comparator in July and +0.042361 in December, with a June regression of 0.008747 seconds; July's paired-day 95% interval crosses zero (-0.011951 to +0.066614 seconds). Its independent postscore audit reproduced the failed F1 gate with zero metric delta.

Scheduled-inbound density **lost 0.006124 seconds to same-peer landed density in July** and lost 0.013206 seconds to the unchanged clean control on the identical 190,713 rows. Its December gains were +0.050162 and +0.075849 seconds against landed and clean respectively; June regressed against both. The July paired-day lower bound, one-day removal and top-ten removal all failed against **both** controls. The independent postscore audit reproduced all nine full-panel RMSEs and six comparisons with zero reported numeric delta. It stopped at F1 with no official upload.

The current-expert context gate improved **0.069855 seconds against its May-fitted intercept-only gate** in July on all 190,713 rows; it gained 0.069855 seconds against unchanged clean F1 as a separate check. The paired-day lower bounds, one-day removals and top-ten removals stayed positive against both, and December regressed by 0.355083 seconds against intercept and 0.355082 against clean, within the one-second guard. The predeclared July improvement needed **at least 5 seconds against both comparators**, so this gate stopped at F1. Its independent audit reproduced nine RMSEs and six comparisons with zero reported numeric delta. It did not proceed to F3, ranking fit or upload. None of these directions supplies a five-second robust, transported complete-cohort gain.

The lower panel is deliberately on a **different, absolute RMSE scale**: official 2026 V3 = **278.0146** and V4 = **281.7265**, both with `Succeeded` receipts. V3 remains the best of these verified submissions. Neither official point is joined to the local 2025 bars, and historical V3 local 268.662991 is omitted: its July/November fit and evaluation history differs from these earlier-origin F1 arms. A lower local F1 absolute score is not an estimate of official 2026 improvement.

![Separate October and November 2025 ARR diagnostic against unchanged clean predictions](arr_diagnostic.png)

The September-trained ARR innovation residual **worsened** complete October
and November 2025 RMSE by 0.244839 and 0.849403 seconds against unchanged
clean predictions. Both paired-day intervals are below zero. The result was
independently replayed and stopped before F1/F3, ranking inference or upload.
It has a different fit history and cohorts from the F1 bars and official 2026
points; it is shown separately for that reason.

[`aggregate_scores.csv`](aggregate_scores.csv) has twenty F1 month rows, two
separate ARR diagnostic month rows and two official receipt rows. Each row
includes the exact score receipt and SHA-256; local rows also name their
independent audit or postscore report and its SHA-256. The [`build_chart.py`](build_chart.py)
script reads **only these aggregate receipts**, verifies pinned hashes and
score-to-audit binding, checks original full-cohort row counts and `control -
candidate` arithmetic, and regenerates the CSV and both PNGs. The route-duration
audit additionally verifies raw/clean equality on all three 2025 F1 months;
the scheduled and current-gate audits verify both July comparisons and their
failed gates. From the PRC workspace root, run `& .\.review-venv\Scripts\python.exe
knowledgeable-helicopter-screening\docs\campaign-evidence\build_chart.py`.
Rebuilding requires the private local aggregate receipts named in the CSV;
the checked charts and CSV travel with this branch.

The F1 figure excludes the July-only backlog result (no matched December), clean F1/F3 controls (no candidate comparison), and other input-only screened directions (no RMSE). Exposed 2025 cohorts and paired-day intervals do not independently validate transfer to 2026; December 2025 is reused transport evidence, not an independent ranking trial. No row represents an unmeasured gain, ranking-month labels, or an unsubmitted official score.
