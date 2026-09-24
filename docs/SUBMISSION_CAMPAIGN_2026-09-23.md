# PRC 2026 submission campaign, September 23-27

This guide governs the active Codex goal through Sunday, September 27, 2026
(America/Los_Angeles). Aim for 240 and then 235 official RMSE, but report
measured scores rather than promising either. Work on model quality every day;
use **at most five** official submissions per organizer day. Five is a cap, not
a requirement to upload weak candidates. Continue the local research loop even
when no candidate qualifies.

The links into `../../review_work`, `../../private_runs` and `../../output` refer
to local private workspace artifacts outside this Git checkout. They document
receipts for the local campaign; the branch contains no data, fitted model,
row-level prediction or credential needed to reproduce those private results.

## How to use the five daily slots

Keep one frozen complete baseline and change one identifiable information or
modeling element per candidate. Write its hypothesis, matched control, score
cohorts and stop rule before fitting. Screen input availability and leakage
first, then run the same chronological local comparison for both arms. Only a
candidate that survives the complete-cohort and influence gates below can
consume an official slot. Prepare multiple independent qualifying candidates
when evidence supports them; do not manufacture five minor variants to fill
the allowance.

An official result is a transfer check for an already frozen model, not a way
to estimate hidden month, airport, route, or row errors. Do not choose the next
weight or five near-identical uploads from aggregate leaderboard feedback.
Track local and official RMSE as separate series. Mark 260, 250, 240 and 235
seconds only when a comparable complete score actually crosses each milestone;
an official milestone requires an official receipt. If several locally
qualified but distinct directions are ready on the same day, release them one
at a time with independent quota and byte-integrity preflights.

## Rules and current state

The [organizer's Ranking page](https://prc-data-challenge-2026.netlify.app/ranking.html)
was rechecked September 23. It requires exact template IDs, ranks each team's
best RMSE, limits submissions to five per day and 1 GB per bucket, and warns
that attempts to learn from or exploit the ranking process are unfair. It does
not specify the quota reset timezone. Never submit fabricated route offsets,
synthetic perturbations, or repeated variants designed to infer hidden labels
or subgroup errors. A submission must be a genuine, independently motivated
complete model candidate. The official aggregate score may confirm transfer
of a frozen candidate, not select weights, rows, rules, or the next variant.

| Submission | Local evidence | Official RMSE | Decision |
| --- | --- | ---: | --- |
| V3, September 16 | Exposed July/November 2025, 268.662991 complete seasonal RMSE | 278.0146 | Retained official reference |
| V4, September 23 | January/July 2025 specialist trailed full history locally | 281.7265 | Reject; 3.7119-second official regression |
| Destination-context F1, September 24 | Matched July complete gain +0.008268 sec; paired-day interval crosses zero | Not submitted | Stop before F3; influence checks fail |
| Stand-arrival missing-NM F1, September 24 | July +0.588 sec vs base, +3.260 vs airport control; June regresses | Not submitted | Stop before F3; July base interval and both top-ten checks fail |
| Clean nine-component F1 control, September 24 | June/July/December complete RMSE 318.758/334.130/234.744; earlier Jan-Apr fit and May weights | Not submitted | Diagnostic reference only; no matched candidate gain |

V4 already used at least one September 23 submission slot. The prior V4 bucket
preflight counted zero same-UTC-day objects, but the official reset boundary is
not documented. Count existing objects and organizer responses before *every*
upload. Preserve V3, V4, all failed results, and unrelated dirty files.

## Dependency DAG

```text
source and eligibility check ─┬─> observed-feature hypothesis ─> feature audit ─┐
                              └─> clean V3 chronology/transfer reference ───────┤
matched baseline, cohort and score contract ───────────────────────────────────────┤
                                                                               F1 fit
                                                                                  |
                                                                           freeze/audit F1
                                                                                  |
                                                                            score/gate F1
                                                                                  |
                                                                          F3 fit if F1 passes
                                                                                  |
                                                                     freeze/audit/score F3
                                                                                  |
                                                                      full-year fit + ranking
                                                                                  |
                                                                  independent release check
                                                                                  |
                                                                   quota/credentials preflight
                                                                                  |
                                                                        one official upload
                                                                                  |
                                                                     readback/score/ledger
```

Independent feature, source, review and score-contract tasks may use subagents
with exclusive files. The main agent owns GPU/RAM admission, model integration,
final verification, quota and upload sequencing. Never run competing heavy
fits to fill agent slots; prior 443/457 tree refits approached 14 GiB each.

## Completed directions and next dependency

The first matched test compared same-destination context against temporal context
on identical 443-field history. The fourteen-field candidate and matched
control are frozen under `private_runs/lead235_20260923/destination_features/v3/`
in the PRC workspace (one directory above this Git checkout); the independent
postbuild review passed in
`review_work/lead235_20260923/destination_validation/postbuild_v3/verification.json`
in that same workspace.
The changed observable selection covers 253,281/2,085,047 training departures,
but zero November Rome missing-NM cases. The 457-field control and candidate
passed prefix/source and complete-prediction audits. F1 July gained only
0.008268 seconds on complete matched cohorts; its paired-day interval crossed
zero, and day/top-ten removal checks failed. The independent postscore audit
passed. This arm is closed without F3 or upload; see the private ledger.

The second input hypothesis was completed arrival context at an exact departure
stand for missing-NM rows. The [input-only support report](../../output/lead235_20260923/stand_reuse_support_v1/REPORT.md)
found unique completed same-stand arrivals within six hours for 16,931/22,470
missing-NM 2025 rows and 3,969/5,290 ranking rows, but measured no predictive
gain or aircraft continuity. The [frozen feature protocol](../../review_work/lead235_20260923/stand_reuse_features_v1/PROTOCOL.md)
compares four same-stand features with four equal-width airport-wide features
on the same source and rows. Both 14-month input tables are complete. A separate
input-only verifier checked all 2,429,888 exact departure IDs and 2,713 sampled
dual-arm rows; its receipt is at
`../../review_work/lead235_20260923/stand_reuse_validation_v1/receipt.json`.
The 21 producer/support, five independent-verifier, five composition, six fit,
three postfit, and six scorer synthetic tests passed September 24. The separate [matched missing-NM protocol](../../review_work/lead235_20260923/stand_reuse_fit_v1/PROTOCOL.md)
uses the accepted chronological **ExtraTrees leaf20** route and unchanged
ordinary/finite-exceptional composition; the feature protocol's provisional
CatBoost reference is not the accepted route. Both ET20 fits used the same
4,712 clean F1 refit labels; native predictions were independently replayed.
Three complete F1 panels were frozen and reviewed before labels, and the
[one-time score](../../private_runs/lead235_20260923/stand_reuse_scoring_v1/F1/score_v1/result.json)
passed independent arithmetic audit. July gained 0.588 seconds versus the
unchanged base and 3.260 versus the airport context, but its paired-day
interval versus base crosses zero and removing the ten most beneficial rows
reverses both gains. June regressed 1.625 seconds versus base. Rome accounts
for 99.5% of July SSE gain versus base. This direction is closed without F3,
ranking inference, or upload. Do not tune the stand arm on the exposed F1
failures. The source-aware model contrast below tests a different mechanism;
source discovery still needs an input-supported signal.

The **clean chronological full-V3 reference** keeps later comparisons from
confusing the original expert/calibration history. The
fixed origins are F1: Jan-Mar train, April stop, Jan-Apr refit, May calibration,
June/July/December evaluation; F3: Jan-Jul train, August stop, Jan-Aug refit,
September calibration, October/November/December evaluation. Preserve the
original route ownership and all rows. This is a diagnostic, not an arm that
can pass a five-second *matched candidate* gate by itself. Finish F3 when the
F1 control passes its lineage, ID and complete-panel validity checks; only a
subsequent candidate-versus-control comparison uses the July/December gain
gate. Do not submit the control merely because it has a new fit date.

The next source-discovery hypothesis must offer a *new observable distinction*
for the winter missing-NM/Rome errors, not another coefficient or generic model
size. A separate, bounded modeling question is now specified: can known-source
labels improve a rich missing-source model through shared parameters? Its
[source-aware design](../../review_work/lead235_20260924/source_aware_design_v1/PROTOCOL.md)
compares identical-capacity 451-field missing models with versus without
finite-label gradients in a shared trunk. It replaces only the 0.25 normalized
missing component, against both the matched rich-only arm and the unchanged
clean V3-like complete control. This is a design and synthetic canary, **not**
a fitted model or demonstrated gain. It requires the independently verified
nine-component clean F1/F3 reference, an executable frozen fit protocol, and a
generalized missing-route scorer before evaluation. The five-second robust
complete-cohort gate applies against both comparators; the previously failed
pooled model and fragile rich-only tune gains make success uncertain.

Missing-NM grouped-slope ideas currently have only synthetic evidence, and
the previously tested grouped-slope candidate lost to V3. Any new feed still
requires an input-only support study, matched strong control and fit-only label
contract before scoring. Previously rejected January/July-only training, weighting,
backlog, clock density, retrieval, shared representation, template, operator,
prefix, arrival and fitted-selector branches remain closed unless genuinely
new source evidence changes their premise.

The September 24 [arrival-schedule source audit](../../review_work/lead235_20260924/source_gap_v1/REPORT.md)
found a narrow untested distinction: recent *completed* arrivals' in-block
minus scheduled-arrival offsets, compared with taxi-in context on the exact
same peers. Its [frozen 14-month input screen](../../output/lead235_20260924/arrival_offset_support_v1/REPORT.md)
confirmed substantial peer variation, but failed the predeclared Rome
missing-NM January support gate: only 4 qualifying days in 2025 and 8 in
2026 versus 10 required for each. Both Julys passed. Independent ID, sample
and four-cell arithmetic checks, including a [separately owned raw replay](../../output/lead235_20260924/arrival_offset_verify_v1/REPORT.md),
reproduced the input result. This branch is
closed without supervised fitting or an upload. It is retrospective batch
information, with no demonstrated live publication-time availability.

The [transfer audit](../../review_work/lead235_20260924/transfer_audit_v1/REPORT.md)
proposes a separate retrospective leave-July/November-2025-out comparison to
test recipes with more training history. The [metadata-only cohort gate](../../review_work/lead235_20260924/lomo_protocol_v1/README.md)
passed independent reconstruction of every stage and earlier-month prior bank
without opening labels; no full-recipe LOMO model has been fitted. This view
cannot replace F1/F3 or turn exposed 2025 labels into a blind estimate of
2026 performance. A [model-opportunity review](../../review_work/lead235_20260924/model_opportunity_v1/REPORT.md)
found that seasonal/decaying airport state had already tested negative; a
direct-horizon source-error forecast was screened next. Its [metadata-only
feasibility check](../../review_work/lead235_20260924/source_forecast_feasibility_v1/REPORT.md)
can construct six-month 2025 tasks but cannot replay July 2026's full-year
history followed by six unlabeled months: January 2025 has no earlier 2025
labels. Stop before a forecaster/downstream fit unless genuinely earlier
labeled history changes this limit.

An independent [runway-transition input screen](../../review_work/lead235_20260924/fresh_signal_v1/REPORT.md)
also stopped before a fit. Comparing query-runway use over the last 15 minutes
with the preceding 45 minutes varied on ordinary departures, but Rome missing-
NM January support and contrast failed its frozen four-cell gate. July-only
variation does not establish a winter missing-route improvement.

The [V3/V4 submitted-prediction audit](../../output/lead235_20260924/prediction_shift_v1/REPORT.md)
verified exact alignment on 344,841 IDs and bitwise unchanged predictions on
all 5,464 protected-route rows. V4 changed 332,916 submitted predictions,
with a 37.297-second RMS difference, concentrated in some ordinary airport-
month and low-support cells. This locates expert disagreement, not hidden
error; no cell-based correction or route selection follows from it.

The `clean_v3_v2` prefit freeze was superseded before fitting because its
source hashes omitted imported modules. The corrected
[v3 F1 recipe](../../review_work/lead235_20260924/clean_v3_recipe_v3/README.md)
binds 48 source files; the [independent prefit receipt](../../review_work/lead235_20260924/clean_validation_v1/recipe_prefit_v3.json)
verified protocol SHA-256
`5f5979a8c0b7e50714d689c3a465b38b0cfa0e73d4c0b320acbc20d1f2402e25`,
reused predictions and both label-free canaries. Seven recipe tests and 22
independent validation tests passed. All five new F1 components finished their
chronological fits. The [clean composer](../../review_work/lead235_20260924/clean_composer_v1/README.md)
passed an [independent nine-component and May-weight review](../../review_work/lead235_20260924/composer_validation_v1/nine_component_postfit_v1.json).
All 539,532 June/July/December rows were frozen and independently recomposed
with zero arithmetic difference before outcomes were opened. The one-time
[complete F1 diagnostic](../../private_runs/lead235_20260924/clean_control_score_v1/F1/diagnostics.json)
measured 318.758152/334.129587/234.743663 RMSE, and its [independent score
audit](../../review_work/lead235_20260924/clean_control_score_audit_v1/receipt_v1.json)
replayed every label join and full/month/route arithmetic. One extreme June
flight adds 55.976 seconds to the June RMSE, so all headline scores retain it.
These absolute clean-prefix numbers are not comparable to historical V3's
268.662991 seasonal local score or its 278.0146 official result. F3 and a
source-aware real fit remain downstream of the clean-control validity gate. Use the
verified LOMO metadata as a separate retrospective check only when a matched
full recipe is feasible. In parallel, seek a distinct input-supported source
mechanism; do not retune stopped input screens on their observed failures.

## Candidate qualification

Before fitting, record the hypothesis, one changed information/modeling element,
availability on both 2025 and 2026 inputs, control, full cohorts, allowed
training IDs, fixed capacity/seed, resource budget, composition and rejection
rule. Never project a departure target or departure block into feature
generation; an arrival in-block clock is eligible only from an ARR-phase
filtered read under the frozen event-order contract. Fit preprocessing only
on eligible supervised training cohorts. Freeze *both* complete-cohort prediction
arms, exact ID order and SHA-256s before an evaluator opens its outcomes.

Use the [existing experiment charter](../../review_work/clock_campaign_20260923/EXPERIMENT_CHARTER.md)
and the [full-cohort evaluator](../../review_work/lead235_20260923/validation/README.md)
for historical-V3 diagnostic scoring. That evaluator is bound to historical
V3, computes a combined seasonal bootstrap interval, and does not implement
the matched-control or per-month interval gates below. Before qualifying a
candidate, separately freeze and independently audit a matched-control scorer
for both complete monthly cohorts, per-month paired-day intervals, and the
October/December transport panels:

- Primary local score is `sqrt((192122*July_MSE+152719*November_MSE)/344841)`
  on all original rows, not an average of monthly RMSEs or a route-only score.
- Require at least **five seconds** of matched complete seasonal gain, positive
  gains in both primary months with a positive paired-day bootstrap lower
  bound, and gains surviving every one-day and top-ten-beneficial-row removal.
- Check June/October adjacent-month behavior and both December origin-gap
  panels. October/December complete panels must remain within the predeclared
  one-second regression guard when comparable. December outcomes are shared,
  not independent replication. Stop at F1 if its July or December gate fails.
- Use matched capacity, information and fit history for the actual claim.
  Historical V3 versus earlier-prefix candidate compositions are useful
  diagnostics, not a clean causal estimate of a candidate's benefit. All 2025
  evaluation periods have already been exposed in prior project work.
- Independently replay a sample of native predictions and every final
  arithmetic/ID/route contract before a full-year fit. The full-year ranking
  candidate must preserve 344,841 IDs in exact template order, finite int32
  nearest-even predictions without clipping, a unique versioned filename,
  source/model/prediction hashes and the original unchanged rows where declared.

The five-second gate is an intentionally demanding campaign threshold. If no
candidate qualifies, record the failed mechanism and keep researching locally.
Do not loosen the gate after seeing a result or upload a weak arm to occupy a
daily slot.

## Daily run and submission ledger

For September 24, 25, 26 and 27, run a morning campaign session and continue
the independent work through the day as resources allow. Review previous
receipts, current goal status, the organizer rules and bucket state. Work the
most promising untested node of the DAG. Freeze every experiment's protocol,
model, predictions and score receipts; append one entry to the private ledger
at `../../output/submission_campaign_20260923/LEDGER.md` even for stopped arms.
The guide contains no raw data, prediction rows or credentials.

For each *qualified* full candidate, prepare a versioned one-use release
package. Use the established V4 release verifier/uploader as a pattern, but
its script is pinned to V4 and cannot be rerun for a new version. Verify the
candidate against `submitting.parquet`, source/model hashes, independent replay,
and exact schema before opening network access. Verify unused object key,
current object count and bucket bytes immediately before uploading. The
organizer may use a different day boundary than the uploader's UTC count;
stop on ambiguity or an organizer rate-limit response. Never overwrite a
published object. Read back and hash the uploaded bytes, poll the result,
record only status/score and preserve the original HTTP upload and SHA-256
readback receipts. Do not
put the access key, secret, or result truth references in files or logs.

The last upload used credentials supplied through a one-use stdin path. No
persistent credential profile was present at campaign setup. Scheduled runs
may prepare and audit candidates autonomously; if no secure credential
channel is available at release time, they must mark the upload blocked and
surface that concrete requirement. Do not embed secrets in the schedule or
repository. Claim an upload only after HTTP success and remote readback;
claim acceptance and a score only after the organizer reports `Succeeded`.

The private ledger records date/time and assumed quota day, candidate ID,
one-line hypothesis, changed components, baseline/arm hashes, fitting IDs,
local complete and month scores, influence/transport result, frozen ranking
SHA-256, submission key, remote hash, official score, decision and unused
slots. Chart **comparable complete RMSE** against experiment order, separated
into local development and official ranking series; annotate different fit
history and dates. A 240 or 235 crossing requires an actual official receipt.
Do not treat five related variants as independent confirmation.

On Sunday, September 27, issue a final report with best verified local and
official scores, every candidate and stop reason, resource/slot usage, transfer
limits, and one justified next direction. End the recurring schedule and mark
the goal complete only if the campaign work and final report are actually done;
do not mark the *235 target* achieved without an official score below 235.
