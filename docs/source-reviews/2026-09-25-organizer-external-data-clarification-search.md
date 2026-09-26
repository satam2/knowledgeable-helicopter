# Organizer clarification search: external data and ranking ARR labels

Checked 25 September 2026 PDT (26 September 05:24 UTC). Read-only public-source
check; no organizer contact, satellite observation body, challenge row, label,
fit, score, leaderboard response, or upload. The two external-data questions in
the [unsent draft](../../../review_work/lead235_20260925/li_organizer_query_v1/DRAFT.md)
remain independent. The [earlier ARR rule audit](../../../review_work/lead235_20260924/arr_target_month_eligibility_v1/REPORT.md)
defines the separate ranking-month training question.

| Question | Exact first-party wording and scope | Result |
| --- | --- | --- |
| Does a CC BY 4.0 satellite dataset satisfy the prize's additional-data license condition? | The [eligibility page](https://prc-data-challenge-2026.netlify.app/eligibility.html#eligibility-for-prize) says: "All used external datasets are openly accessible/usable and documented" and "All additional datasets used are openly available under an open source license." It separately requires produced source code on GitHub under GNU GPLv3. | **No product-specific interpretation published on the checked organizer pages.** The EUMETSAT catalogue's CC BY 4.0 grant is documented in the local LI/MSG source reviews, but the organizer has not said whether it accepts that data/content license under this literal condition. Prize/source admission remains **HOLD**. |
| May an archived satellite observation sensed before takeoff but first publicly available afterward be used retrospectively? | The [rationale](https://prc-data-challenge-2026.netlify.app/rationale.html) says a taxi-out estimate "can be used in post-operations analysis"; the [overview](https://prc-data-challenge-2026.netlify.app/) defines a 2025 learning set and January/July 2026 ranking. The [data dictionary](https://prc-data-challenge-2026.netlify.app/data.html) defines `MVT_TIME_UTC_mvt` as best-available takeoff time for departures. | **No rule about sensing time versus first publication was found.** A retrospective application is described, but this does not grant use of a later-published product or waive the campaign's stricter publication-by-takeoff policy. Retrospective-source admission remains **HOLD**. |
| May `PHASE_mvt=ARR` ranking-month labels train or adapt a model? | The [data page](https://prc-data-challenge-2026.netlify.app/data.html) calls 2025 files the "Training Dataset" and January/July 2026 the "Ranking Dataset." It says the ranking data "contains the same columns as the training data" and only departure `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` "have been blanked out." The [ranking page](https://prc-data-challenge-2026.netlify.app/ranking.html) warns against attempts "to learn from or exploit the ranking process." | **Availability is not training authorization.** Neither page explicitly permits or prohibits fitting on the retained ARR outcomes; the ranking warning does not define this case. Target-month ARR-label training remains **HOLD** under the existing campaign protocol. |

All five linked organizer sections ([overview](https://prc-data-challenge-2026.netlify.app/),
[eligibility](https://prc-data-challenge-2026.netlify.app/eligibility.html),
[data](https://prc-data-challenge-2026.netlify.app/data.html),
[ranking](https://prc-data-challenge-2026.netlify.app/ranking.html),
[rationale](https://prc-data-challenge-2026.netlify.app/rationale.html))
returned HTTP 200 during this check. Their responses had no `Last-Modified`
header, so the check date is **not** a claimed publication date. Navigation
exposed no FAQ or dedicated updates page; `/sitemap.xml` returned 404. The
overview links 2024/2025 GitHub organizations, while ranking credits a
[contributor's 2026 leaderboard repository](https://github.com/arquicanedo/prc-challenge-2026).
Its [public issue list](https://github.com/arquicanedo/prc-challenge-2026/issues)
was empty on this check, and that repository is not presented as an organizer
rules authority. The site points to an [OSN Discord channel](https://discord.com/channels/1252625490930307142/1304403918515605504)
for updates; the in-app browser redirected to Discord login. Its messages were
**not inspected**. This is silence in the accessible published material, not
proof that no guidance exists behind login and not permission to relax a gate.

**Gate delta: none.** Retain the independent license, retrospective-publication,
and ranking-month ARR-supervision holds. A dated, attributable organizer ruling
with the exact products/use and supervision scope would be needed to change
them; this search itself does not authorize observation acquisition or fitting.
