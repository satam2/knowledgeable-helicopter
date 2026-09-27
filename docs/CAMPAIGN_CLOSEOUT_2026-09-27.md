# PRC 2026 model campaign closeout

27 September 2026, America/Los_Angeles. The September 23-27 campaign ended
without a qualifying V5 model or another official upload. The best verified
official result remains **V3: 278.0146 seconds** on all 344,841 January/July
2026 ranking departures. The 240/235 target was **not reached**. The
[campaign guide](SUBMISSION_CAMPAIGN_2026-09-23.md) and private
[`LEDGER.md`](../../output/submission_campaign_20260923/LEDGER.md) retain the
protocol, each stopped arm, input audit and one-use receipt. This public report
contains no raw challenge data, fitted model, row-level prediction or secret.

## Verified scores and submissions

| Result | RMSE (s) | Cohort and decision |
| --- | ---: | --- |
| V3 official | **278.0146** | `Succeeded`, 344,841 January/July 2026 ranking rows; retain. |
| V4 official | 281.7265 | `Succeeded`, same 344,841 rows; Jan/Jul-only 2025 specialist lost 3.7119 s to V3. This is the one known successful upload during this campaign. |
| Historical V3 local | 268.662991 | July/November 2025 development folds, 190,713/162,332 rows; not an official forecast or an untouched holdout. |

The historical local score is
`sqrt((192122*July_MSE + 152719*November_MSE)/344841)`, using the 2026
ranking-month counts as weights. Its retained MSEs are
90,789.05806870645/48,769.177138419 square-seconds. An independent
[score audit](../../review_work/lead235_20260927/score_audit_v1/REPORT.md)
rehashed the V3/V4 organizer and V3 local receipts, recomputed the weighted
score exactly, and replayed all 24 aggregate chart rows and pinned source/audit
hashes without opening row-level outcomes. The 9.3516-second local/official gap
does not identify a unique overfitting penalty: year, cohort, model fit origin
and final fixed-weight policy differ.

A public team-filtered [Sunday status
receipt](../../output/submission_campaign_20260923/TEAM_PUBLIC_STATUS_20260927T150209Z.json)
at 15:02 UTC returned only V1-V4 successful entries, `nextCursor=null`, each
on 344,841 pairs. Its SHA-256 is
`70e63d0692ed9b1a199c7c26392859f55da5302a7312bb15eec5b5ef15b34a50`.
This endpoint excludes failed attempts and cannot establish unused daily slots,
the organizer's reset timezone, current authenticated bucket bytes or credential
availability. Those quantities remain **unknown**. No V5+ package or full-year
ranking fit qualified, so no quota/credential/upload preflight was warranted.

## Model decisions

The [score chart](campaign-evidence/REPORT.md) separates matched local F1
gains, retrospective ARR diagnostics and absolute official results. Each F1
direction uses its own equal-origin comparator; absolute F1 RMSEs are not
comparable to historical V3 or the official 2026 score. All ten stopped at the
frozen July/December advancement gate, before F3 or ranking inference:

| Direction | Complete July 2025 matched gain (s) | Stop reason |
| --- | ---: | --- |
| Clock-cell density weighting | -0.231 | July/December regression and robustness failure. |
| Destination ninth peer | +0.008 | Below five seconds; day/top-ten influence failed. |
| Same-stand ARR context | +3.260 versus airport-wide | Only +0.588 versus unchanged clean; top-ten and June/Rome transport failed. |
| Source-aware joint learning | +1.448 versus rich-only | Lost 4.315 versus unchanged clean; top-ten and December failed. |
| Clock-gap peer rank | +0.011 | Lost 0.004 versus clean; day/top-ten checks failed. |
| ARR-shared stem | -0.630 | July and December regression. |
| Extra-time IEM weather | -0.212 | July loss to routine-only; source eligibility also unresolved. |
| Route-duration centering | +0.027 | Paired-day interval crossed zero; below five seconds. |
| Scheduled-inbound density | -0.006 | July loss to landed-peer control and failed robustness. |
| Current-expert contextual gate | +0.070 | Below five seconds; December lost 0.355. |

The separate scheduled-pending backlog F1 lost 0.091 seconds on complete July.
September-fitted 2025 ARR innovation worsened complete October and November by
0.245/0.849 seconds, respectively. Clean F1/F3 controls are references, not
new model wins. Input-only and source screens have no RMSE and were never
treated as a model result. The independent [decision
audit](../../review_work/lead235_20260927/decision_audit_v1/REPORT.md) binds
these decisions to the ledger and confirms no reviewed V5+ release package.

## Source and execution gates

The source screens stopped OPDI, airport/operator traffic, A-CDM, NOTAM,
ADSB.lol, VDL IQ, TAF, GFS, IMERG and related feeds at combinations of open
data rights, historical exact publication/version, join identity or airport
coverage. The newly checked D-ATIS runway archive covers US airports only.
The July 2026 EDDF missing-NM stand shift concerns 59 rows (0.0171% of the
ranking cohort) with unknown hidden errors; it is not evidence of a join bug
or a whole-cohort correction. Same-input signed clock regimes, shared
finite/missing learning, LOBT anchoring and custom attention/queue variants
overlap prior negative or sub-threshold tests. The ledger has the separate
source and architecture receipts, including the retrospective threshold
erratum for the EDDF screen.

MSG Cloud Mask remains the most concrete distinct continuous observation:
current catalogue slots and the provider's CC BY 4.0 grant are documented,
but neither establishes original item bytes at historical takeoff, valid
airport pixels, nor organizer prize acceptance. Sunday's independent
[rules audit](../../review_work/lead235_20260927/rules_audit_v1/REPORT.md)
found no public ruling on CC BY 4.0, later-published retrospective features,
or using ranking-month arrival outcomes to train a departure model. These are
three separate holds. The prepared [organizer clarification
draft](../../review_work/lead235_20260925/li_organizer_query_v1/DRAFT.md)
was not sent; contacting a third party still requires explicit authorization.
The strict pre-takeoff publication gate is this campaign's as-of policy, not
an express rule on the organizer's retrospective task.

Available host memory was about 14.3 GiB on Sunday morning, below the
unchanged 24/26 GiB full-fit launch floors and 8 GiB reserve. No PRC training
process was visible. Earlier full-capacity resource canaries do not justify
weakening that admission rule. The April/August nine-expert OOF ARR diagnostic
would require 30 serial training invocations and has neither a positive
preliminary negative-control result nor current resource admission.

## Next evidence gate

Obtain a dated organizer answer on the named MSG Cloud Mask CC BY 4.0 prize
treatment and retrospective-publication policy; for strict as-of use, obtain
publisher-certified or contemporaneous first-publication/version/hash proof
for exact selected bytes. Then authorize a bounded airport-pixel and four-cell
input audit before freezing a matched F1 experiment. A positive input audit
would not itself predict a five-second gain: complete chronological F1 and F3,
per-month and influence robustness, full-season score, independent release
verification and quota preflight remain required. Ranking-month ARR-label
supervision needs its own dated ruling before use.

No official 240/235 receipt exists. This closes the finite campaign, not the
longer-term research question.
