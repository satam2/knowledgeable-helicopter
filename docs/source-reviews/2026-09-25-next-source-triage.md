# EAD NOTAM airport/runway state: next-source triage

25 September 2026. **STOP at the source-documentation gate.** EUROCONTROL's
European AIS Database (EAD) International NOTAM Operations (INO) is a genuinely
different operational observation from the campaign's stopped flight-event,
surveillance, airport-volume, ATFM, and weather arms. A NOTAM-derived airport
or runway availability state could describe restrictions at a departure's
airfield, rather than estimating that departure's hidden off-block time. The
first-party pages do **not** establish a prize-compatible open dataset grant,
a publicly reproducible 2025/2026 historical archive, or immutable as-of
snapshots. This review did not retrieve a NOTAM record, archive body, challenge
row or label, prediction, model, or score; no API call, fit or upload occurred.

| Gate | First-party evidence and decision |
| --- | --- |
| Named observation and novelty | [EAD INO](https://www.eurocontrol.int/service/international-notam-operations) maintains and distributes original/processed NOTAM, SNOWTAM and ASHTAM; the service advertises NOTAM history and pre-publication visualization. [EAD](https://www.eurocontrol.int/service/european-ais-database) covers aerodrome information and notices from ECAC and beyond. A candidate would count already-published *airport/runway restriction notices* active at an allowed query time, using notice type and location, not any flight's off-block event. This is a source class distinct from OPDI per-flight events, VDL IQ messages, ADSB.lol ground tracks, operator IFR counts, CAMS PM2.5 and the prior daily ATFM lead. **Pass only for a distinct named mechanism; no incremental value or event schema was measured.** |
| Rights and prize use | [EAD](https://www.eurocontrol.int/service/european-ais-database) says data providers retain intellectual-property rights over what they input; client access requires a signed EAD agreement, and services can carry charges and Member State royalty fees. Its free EAD Basic application is limited public consultation for general purposes. No EAD NOTAM dataset-specific open license or grant permitting republication/reproducible prize use is identified on these service pages. The [organizer's rule](https://prc-data-challenge-2026.netlify.app/eligibility.html#eligibility-for-prize) requires external datasets openly accessible/usable and documented, with all additional datasets openly available under an open-source license. **Fail current acquisition/prize admission.** A software license or free viewing of notices does not license the underlying data. The result does not rule out a separate national AIS source or a later explicit grant. |
| Required calendar | [EAD](https://www.eurocontrol.int/service/european-ais-database) describes a long-running service (almost 20 years as of April 2024) and ECAC-area source participation; [INO](https://www.eurocontrol.int/service/international-notam-operations) advertises a history function. Neither page supplies an openly licensed, downloadable manifest of original notices and cancellations for **January and July 2025 and January and July 2026**, let alone all 2025, for these airports. **Unknown for exact historical airport-month records and fail the documented archive gate.** Long-running real-time service is not a dated historical release inventory. |
| Historical first-publication/as-of | [INO](https://www.eurocontrol.int/service/international-notam-operations) mentions creation, distribution, NOTAM history and immediate notification in briefing services; [EAD](https://www.eurocontrol.int/service/european-ais-database) promises real-time access to current aeronautical information. The pages do not expose an immutable public record of when each original notice/correction/cancellation first became accessible, the archived revision bytes, or a retrieval SLA for historical queries. A notice's effective period or later historical presence cannot establish that its precise version was public before a past query takeoff. **Unknown/fail as-of evidence** for the campaign's stricter historical observation policy. This is separate from the organizer's published prize rule; no claim is made that the rule expressly forbids retrospective NOTAMs. |
| Ten-airport join | The [organizer's dictionary](https://prc-data-challenge-2026.netlify.app/data.html) lists EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LTFM and LSZH; `ADEP_mvt` is the departure ICAO, `RUNWAY_mvt` a movement runway, and `MVT_TIME_UTC_mvt` the departure takeoff time. A potential *many-to-one contextual* join would map an aerodrome/runway notice's fixed location identifier to the departure's ICAO/runway and use only versions first published by a frozen pre-takeoff cutoff. EAD's overview says the service covers aerodrome information but does not publish the actual notice key schema, one-notice-per-key rule, UTC normalization or four-cell per-airport counts. **No unique per-flight join is claimed; no ten-airport support has been measured.** |

The source gate is already closed on rights and historical reproducibility, so
there is no reason to acquire messages or probe challenge rows now. Reopen only
for a specific first-party NOTAM archive with a dataset-specific grant the
organizer accepts for prize use, public immutable original/revision publication
receipts covering the four January/July cells and full 2025, and a documented
airport/runway identifier and UTC convention. Before any value read, freeze a
prior-published-only rule, a ten-airport input-only coverage and ambiguity
screen, and an equal-information matched control. Do not use a final historical
state as though it were visible at the flight's query time.
