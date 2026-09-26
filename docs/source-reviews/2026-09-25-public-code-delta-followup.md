# Public code delta follow-up: no admitted new method

Checked 25 September 2026, approximately 23:21 Pacific, against the
[24 September public-method pin](../../../review_work/lead235_20260924/public_method_refresh_v1/REPORT.md),
[25 September delta](../../../review_work/lead235_20260925/public_method_delta_v1/REPORT.md)
and [gap screen](../../../review_work/lead235_20260925/public_method_gap_v1/REPORT.md).
**Decision: no newly justified fit or release candidate.** This is a bounded
read of public source and author disclosures. Three HTTP responses were read:
two GitHub commit comparisons and one raw, public source file, below the
15-request cap. Initial sandbox-denied socket attempts returned no HTTP
content. No challenge rows or labels, external datasets, model artifacts,
predictions, scoring operation, upload or contact were involved. Competitor
code was neither run nor copied into the local implementation.

| Repository | Current delta from prior pin | Decision |
| --- | --- | --- |
| [Kind-Mango comparison, `07f40ce` to HEAD](https://api.github.com/repos/skylinkapi/prc-data-challenge-2026-kind-mango/compare/07f40ce98c1719cc8475a811d79bfacca762a5c9...HEAD) | `ahead_by: 2`; current tip [`16863677177e69c5af0bc8a33ba806070b930481`](https://github.com/skylinkapi/prc-data-challenge-2026-kind-mango/commit/16863677177e69c5af0bc8a33ba806070b930481). The intervening [`7f8d3a6`](https://github.com/skylinkapi/prc-data-challenge-2026-kind-mango/commit/7f8d3a62bc6b1fc0bb9085336bb0292f2ed1f950) prices a second, deeper CatBoost member; the author says its incremental 2025 holdout gain was only 71 MSE, below its 150-MSE build bar, so v70 was not built. The tip adds seven runway-queue summaries (v71), a paired fixed-round holdout report, and an author-published result. | No distinct model-family or validated whole-cohort gain. The earlier delta already covered CatBoost blending; the queue feature's reported transfer is flat. |
| [Victor Alcadi comparison, `5dccaca` to HEAD](https://api.github.com/repos/victoralcadi/prc-data-challenge-2026/compare/5dccacafc696b87adb1ec3c2eb76939c04f057d4...HEAD) | `identical`, `ahead_by: 0`; no new source since the 17 September pin. | No code delta to assess. Its previously documented concepts remain at the earlier disposition. |

The [v71 feature builder at the current tip](https://github.com/skylinkapi/prc-data-challenge-2026-kind-mango/blob/16863677177e69c5af0bc8a33ba806070b930481/src_v3/build_runway_queue_v71.py)
reads supplied movement airport, phase, observed takeoff/landing time, runway,
wake category and destination. It counts prior departures on the query runway
over five and ten minutes, heavy prior departures and their share, prior
departures toward destinations within 30 degrees over ten minutes, and age
since the active 30-minute runway set last changed. The builder's actual
windows are backward-looking in observed movement time, with same-instant
departure peers excluded from counts. The bearing summary additionally reads
airport coordinates from an `external/airports/airports.csv` path; this check
did not establish that file's exact provenance, license or publication
semantics. The source's retrospective event ordering does not establish when
the final records were first available in real time.

The author's [fixed-round comparison](https://github.com/skylinkapi/prc-data-challenge-2026-kind-mango/blob/16863677177e69c5af0bc8a33ba806070b930481/src_v3/measure_mf2.py)
holds one LightGBM recipe and 2025 holdout rows constant, changing seven
columns; its [published aggregate receipt](https://github.com/skylinkapi/prc-data-challenge-2026-kind-mango/blob/16863677177e69c5af0bc8a33ba806070b930481/models/v3/mf2.holdout.json)
reports -559, -524, -586 and -596 MSE at 500, 1,000, 1,500 and 2,000
rounds. The author's [v71 result JSON](https://github.com/skylinkapi/prc-data-challenge-2026-kind-mango/blob/16863677177e69c5af0bc8a33ba806070b930481/submission/kind-mango_v71.result.json)
states 281.8501 seconds on 344,841 pairs, versus its v67 281.865; its
[recap](https://github.com/skylinkapi/prc-data-challenge-2026-kind-mango/blob/16863677177e69c5af0bc8a33ba806070b930481/RECAP.md)
calls the -0.015-second change inside noise and does not accept it. This
published JSON is an author copy, not a result independently checked here
against the organizer API; the holdout attribution is likewise the author's
calculation. The served recipe changes only the LightGBM half of the blend,
so the published result does not isolate the seven columns' effect on a
locally matched campaign control.

The [local queue-mechanism audit](../../../review_work/lead235_20260925/queue_mechanism_audit_v1/REPORT.md)
already inventories prior 5/15/30-minute departures, query-runway discharge
share, active runways, wake/ordered peer summaries and interval proxies in
union387. It distinguishes observed throughput from the physical taxi queue
and records a separately frozen recent-versus-older runway-share input stop.
The newly named same-direction bearing and runway-set-age transformations are
more specific formulas over substantially the same supplied movement bank;
they are not a newly released observation. They may be mathematically absent
as explicit columns, but there is no evidence here for a material gain on
the complete local cohort or a newly lawful event feed. The already stopped
[same-runway prior-landing screen](../../../review_work/lead235_20260924/past_event_direction_v1/PROTOCOL.md)
must not be revived by changing the window or event formula after its gate.

**Operational consequence:** preserve prior source, chronology and campaign
stops. This delta authorizes no input-row acquisition, local fit, F1/F3 score,
ranking inference or upload; a competitor's minor author-reported change is
not a route to a 235-second claim. Reopen only on a genuinely distinct dated
input or independent validation mechanism with its rights, historical
availability and frozen matched control established before outcome access.
