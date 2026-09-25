# EUROCONTROL Airport Corner events: documentation gate

25 September 2026. **Stop before event-data acquisition.** Airport Corner is a
distinct airport-reported operational context source, but its first-party pages
do not establish an openly licensed historical event archive or when individual
versions first became public. No event records, challenge rows, labels,
predictions or scores were read; no model was fit or submitted.

| Gate | First-party evidence and decision |
| --- | --- |
| Observation and distinction | [Airport Corner](https://www.eurocontrol.int/tool/airport-corner) says airports report events such as runway maintenance, industrial action and unusual demand, along with contextual network impact. The publisher explicitly says these reports **complement NOTAMs**. This is an airport-reported *event/context* signal, not an EAD NOTAM, NM ATFM regulation, per-flight A-CDM message or surveillance-derived movement. It might flag temporary airport-side constraints affecting taxi-out; incremental predictive value is unmeasured. |
| Rights and prize use | The [product page](https://www.eurocontrol.int/tool/airport-corner) offers a public view of **non-confidential** airport information; a deeper view is restricted to airports, ATC centres and airlines. Neither that page nor the [public landing page](https://ext.eurocontrol.int/airport_corner_public/) identifies an open data license granting reproducible use of event records. The [organizer's eligibility rule](https://prc-data-challenge-2026.netlify.app/eligibility.html#eligibility-for-prize) requires additional datasets openly available under an open-source license, separately from GPLv3 for solution code. **Fail present prize/source admission**: publicly viewable material is not a documented license. This is not a claim that no separate grant can exist. |
| Calendar and ten-airport support | The [product page](https://www.eurocontrol.int/tool/airport-corner) advertises 130+ partner airports but no event-record inventory for all of 2025 or January/July 2026. The static airport selector on the [public landing page](https://ext.eurocontrol.int/airport_corner_public/) lists EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LTFM and LSZH. **Pass only for today's selectable airport identifiers**, not for event presence, public/non-confidential completeness, past availability, or coverage in each required airport-month cell. No Events page or event API was opened. |
| Historical first publication | The [product page](https://www.eurocontrol.int/tool/airport-corner) says airports can update strategic information at any time and report events; it also describes post-operational impact analyses. The checked pages publish no versioned event archive, original/revision bytes, publication timestamp, or historical query-time SLA for these months. The [publisher's 2020 article](https://www.eurocontrol.int/article/airport-corner-enhancing-network-awareness-performance) describes pre-tactical reports and real-time notifications, which establish an operational channel, **not** a reproducible first-visible time for a historical record. **Fail the campaign's as-of documentation gate**; an event's effective date or a current final description would not show what a past departure could know. |
| Join and mechanism | The [challenge dictionary](https://prc-data-challenge-2026.netlify.app/data.html) identifies the ten departure airports and movement/takeoff fields. A *conditional many-to-one* join would match an Airport Corner airport identifier and a prepublished event validity interval to departures, excluding reports first published after the query time. The checked Airport Corner documentation does not provide a stable event key, UTC interval/revision schema, public-release timestamp, or one-to-one flight identity. **Only airport identity is established; no event-time match or unique per-flight join is claimed.** |

Do not access event records, train, or upload from this source under the present
evidence. Reopen only with a dataset-specific, organizer-compatible open grant,
a documented immutable history of original and revised public events spanning
full 2025 and January/July 2026, publication timestamps, and an airport/UTC
validity schema. Before reading outcomes, freeze an as-of cutoff and an
input-only ten-airport, four-month event-availability/ambiguity audit, then an
equal-information matched control. Static airport dropdown coverage alone does
not justify a prediction claim.
