# Historically issued TAFs: documentation-only source gate

25 September 2026. **STOP before acquiring TAF bodies, challenge rows, fitting,
scoring or uploading.** A Terminal Aerodrome Forecast (TAF) is a plausible new
input: unlike observed IEM METARs, it describes expected local wind, visibility,
clouds and weather over a future validity interval. A prior-issued amendment
could signal changing conditions that the latest observation misses. This is a
distinct hypothesis, not a demonstrated taxi-out improvement. The [AWC product
guide](https://aviationweather.gov/help/data/#taf) explains issue/valid periods,
scheduled forecasts and amendments; its US-specific issue cadence cannot be
assumed for the ten European airports.

| Gate | First-party evidence and decision |
| --- | --- |
| Historical archive | The [AWC Data API](https://aviationweather.gov/data/api/) advertises worldwide TAFs but says its database reaches only the previous **30 days**; its current TAF cache refreshes every ten minutes. The [AWC Archive View](https://aviationweather.gov/tools/archive/) likewise offers only recent views and points older requests to [NCEI AIRS](https://www.ncdc.noaa.gov/has/HAS.DsSelect). NCEI's SRRS Text and NOAAPort Text (NEW) catalogue periods cover 2001/2000 to September 2026 at the *dataset* level, hence encompass full 2025 and January/July 2026 in principle. They do not certify historical TAFs at each airport. The public [SRRS station selector](https://www.ncdc.noaa.gov/has/HAS.StationYearSelect?datasetname=9957ANX&subqueryby=STATION&applname=SRRSBTNSEL&outdest=APPS&dtypesort=dtypeord&stationsort=id) has no exact EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LTFM or LSZH station-code entry. The [NOAAPort Text (NEW) selector](https://www.ncdc.noaa.gov/has/HAS.FileAppRouter?datasetname=6431_02&subqueryby=STATION&applname=&outdest=FILE) has LTFM listed since 2020 and EHAM only through 2002, with the other eight codes absent. Selector absence does **not** prove a raw archive contains no embedded bulletin for these airports; it means no documented, ten-airport, four-cell TAF retrieval path is established. **Fail exact support for now.** |
| As-of issuance and revision | The [AWC TAF schema](https://aviationweather.gov/data/schema/openapi.yaml) distinguishes `issueTime`, forecast `validTimeFrom`/`validTimeTo`, `mostRecent`, and `dbPopTime`, described as the time the product was received. This could support an AWC-reception cutoff **if the same historical versions and receipt values were available**. AWC's 30-day retention does not reach the campaign cohorts. The checked [NCEI AIRS catalogue](https://www.ncdc.noaa.gov/has/HAS.DsSelect) advertises native archived files and periods of record, not per-bulletin original delivery or immutable amendment/revision history for the ten airports. A TAF issue time or valid time alone is not proof that the retrieved historical text had been public at the prediction anchor. **Fail the campaign's historical as-of gate.** |
| Prize-compatible rights | The [NWS terms](https://www.weather.gov/disclaimer) say NWS web-page information is public domain unless otherwise noted, with attribution/endorsement safeguards; this does not itself establish rights for foreign-origin airport TAF bulletins redistributed by AWC or NCEI. The [organizer's prize rule](https://prc-data-challenge-2026.netlify.app/eligibility.html#eligibility-for-prize) requires additional datasets openly available under an open-source license and reproducible documentation. No dataset-specific grant covering the exact historical international TAF archive, or organizer interpretation accepting a public-domain assertion as the literal license condition, was established. **Hold prize use; this is not a claim that all TAF use is prohibited.** |
| Join contract | The [challenge dictionary](https://prc-data-challenge-2026.netlify.app/data.html) defines `ADEP_mvt` as departure ICAO and `MVT_TIME_UTC_mvt` as the recorded departure/takeoff time. A prospective airport-time join would map one original DEP ID to the identical ICAO TAF station, select only forecast versions with independently proven first receipt before a fixed `T`, resolve amendments by the last eligible receipt and a deterministic tie, then use the forecast group valid at `T` (or a predeclared missing value). It must avoid reissued text from the current archive and retain all ten-airport cohorts. `T` is retrospective takeoff, not proof that the same clock was known before off-block. No join or airport-month support was run. |

Reopen only on an explicit reuse grant covering the international source,
documented access to original TAF versions and dated first-receipt/revision
records throughout 2025 and January/July 2026, and evidence for all ten
airport/month cells. Freeze identity, version precedence and the as-of/valid
time anchor before an input-only support audit. Any later supervised test needs
an unchanged IEM-weather matched control, chronological F1/F3 and full-season
gates and independent release review. No TAF body, challenge record, label,
prediction or RMSE was used for this review.
