# Airport-operator daily traffic: source gate, 25 September 2026

**Stop at the documentation gate.** EUROCONTROL's Airport Traffic dataset is a
distinct named operational source for airport-operator daily IFR departure
counts, but the publisher's notice restricts reuse and does not supply the
open-source data license required for prize eligibility. The current CSV
archive also supplies no historical first-publication or revision clock for
each airport-day. No dataset body, challenge row, hidden departure value,
prediction, model or score was read; no API query, fit, organizer contact or
upload occurred. All web checks below were made on 25 September 2026.

The active [campaign guide](../SUBMISSION_CAMPAIGN_2026-09-23.md) asks for a
distinct source with January/July 2025/2026 coverage and a frozen identity,
timing and matched-control gate before row acquisition. I read the prior
[commercial off-block screen](../../../review_work/lead235_20260924/external_offblock_feasibility_v1/REPORT.md),
[A-CDM/NM source screen](../../../review_work/lead235_20260925/acdm_source_gate_v1/REPORT.md),
and [OPDI rights/as-of conclusion in the guide](../SUBMISSION_CAMPAIGN_2026-09-23.md).
This candidate is an *airport-level reported count*, not another purported
per-flight off-block event, ADS-B trace, or NM message archive. It does not
inherit eligibility from any of those screens.

## Evidence and precise status

| Gate | First-party evidence | Finding |
| --- | --- | --- |
| Named observation | The [Airport Traffic metadata](https://ansperformance.eu/reference/dataset/airport-traffic/) defines `FLT_DATE` and `APT_ICAO`; `FLT_DEP_IFR_2` is **airport-operator IFR departures**, while `FLT_DEP_1` is the separate Network Manager count. The proposed observation is only `FLT_DEP_IFR_2`, not a target-derived taxi-out statistic. | **Pass at schema/meaning level.** It is airport-date aggregate input, not a flight-linked measurement. Whether it adds information beyond counts derived from supplied movements has not been tested and must not be asserted. |
| Archive date span | The publisher's [data catalogue](https://ansperformance.eu/data/) lists daily airport traffic for **January 2016 through August 2026**. Its [CSV index](https://ansperformance.eu/csv/#aptflt-csv) lists `airport_traffic_2025.csv` and `airport_traffic_2026.csv`. | **Pass at advertised calendar/product level** for full 2025 and January/July 2026. Actual completeness for each of EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LTFM and LSZH in all required airport-month cells is **unknown**; no CSV body was opened. |
| License and prize rule | The publisher's [copyright notice](https://ansperformance.eu/about/disclaimer/) allows copying with EUROCONTROL attribution only when **not used for commercial purposes** and prohibits modification without prior written permission. It does not identify a dataset-specific open-source license. The [organizer's eligibility rule](https://prc-data-challenge-2026.netlify.app/eligibility.html#eligibility-for-prize) requires external data to be openly accessible/usable and all additional datasets to be openly available under an open-source license; GPLv3 applies separately to solution code. | **Fail the present prize/source admission gate.** Free public CSV availability is not the required license. This is a conservative documented-source decision, not a legal finding about every possible private grant or whether this particular contest counts as commercial activity. No separate dataset grant was found on the checked metadata/catalogue pages. |
| Chronology | `FLT_DATE` is the **date of flights**, not first publication of that day's aggregate. The catalogue/index expose an archive but no per-airport-day first-visible timestamp, original immutable snapshot or revision log. The separate [EUROCONTROL Data app API guide](https://data-app.eurocontrol.int/api) illustrates `syncDate` and `appUpdated` for one **country** day; it does not certify when the Airport Traffic CSV's particular airport-day count first appeared. | **Unknown as-of availability.** A same-day completed total would contain operations after many query departures and is invalid as a prior observation. Even a previous-day total cannot be claimed available at a flight's query time without dataset-specific first-publication/version evidence. File current presence in September 2026 proves only retrospective accessibility. |
| Movement join | The [challenge dictionary](https://prc-data-challenge-2026.netlify.app/data.html) identifies the ten airports, 2025 training and Jan/Jul 2026 ranking periods and `MVT_ID_mvt` as movement identity. The airport traffic metadata exposes `APT_ICAO` and `FLT_DATE` but no shared movement or NM flight ID. A fixed airport/date mapping could attach one count to many departures if date convention and one-record-per-key are established. | **Potential many-to-one context join, not a unique flight match.** No per-cell key uniqueness, timezone/date convention against challenge UTC movement time, missingness or airport coverage has been measured. Those are **unknown**, not zero coverage. |

## Next gate

No acquisition, feature build, fit or upload is authorized from this screen.
Reopen this exact source only after a first-party dataset-specific grant that
permits the proposed reproducible use under a named organizer-compatible open
data license, plus historical first-publication/version evidence for the
airport-day aggregates. Then freeze a *prior-published-day only* rule and
airport/date convention before reading values; verify one aggregate per key
and input-only offered/available/joined counts separately for every 2025
month and both 2026 ranking months at all ten airports. Compare an equal-
information control before any supervised fit. The current documented rights
failure makes even that sample premature.
