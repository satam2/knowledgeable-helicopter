# NOAA GFS prior-issued forecast: source gate

25 September 2026. **STOP at documentation; no data acquisition or fit.** The
NOAA/NCEI Global Forecast System (GFS) is a distinct *prospective weather
forecast* source: a prior-issued gridded forecast of precipitation, wind or
temperature at a departure airport could contain information beyond the
campaign's already tested IEM observations. This is not the stopped CAMS PM2.5
forecast, an aviation event feed, or a rerun of the observed-METAR arm. No
forecast file or event body, challenge row, label, prediction, model or score
was read or produced. The documentation pages below were checked on this date.

| Gate | First-party evidence and decision |
| --- | --- |
| Named observation and plausible signal | [NCEI's GFS product page](https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast) describes forecast atmospheric and land variables including winds, temperature and precipitation. [NCEP's producer page](https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gfs.php) says GFS runs four times daily and predicts globally. A *forecast valid near departure*, selected from an earlier issued run, could flag approaching precipitation/wind disruptions unseen by a prior station report. **PASS for a distinct mechanism only; no incremental taxi-out information is measured.** |
| Prize-compatible reuse | [NOAA's NODD page](https://www.noaa.gov/nodd) calls its data open and publicly accessible. The [GFS NODD listing in the AWS Open Data Registry](https://registry.opendata.aws/noaa-gfs-bdp-pds/) says NOAA NODD data can be used as desired, requests attribution for unaltered data, and forbids implying NOAA endorsement or calling modified data unaltered. The [organizer's eligibility rule](https://prc-data-challenge-2026.netlify.app/eligibility.html#eligibility-for-prize) separately requires *additional datasets openly available under an open source license*. The checked GFS pages do not name a dataset-specific open-source license or organizer interpretation of NOAA's open-public-data grant. **HOLD for literal prize-eligibility confirmation**, not a claim that NOAA forbids this use. The Apache license in NCEI's web explorer HTML is for the explorer page, not a grant for GFS files. |
| Historical product and ranking access | [NCEI](https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast) advertises **0.5-degree GFS forecast files** from 10 October 2006 to present, approximately two years to present *online*, with 00/06/12/18 UTC cycles and three-hourly lead times through +192 hours. At 25 September 2026 this advertises a plausible route to all 2025 and January/July 2026. It does **not** establish that every specific run/field required at ten airports remains accessible. Its separate 1-degree archive advertises only about six months online. NCEI explicitly says the cloud 0.25/0.5-degree feed is a **trailing 30-day window**; the [NOAA-managed cloud registry](https://registry.opendata.aws/noaa-gfs-bdp-pds/) cannot by itself supply this campaign's old runs. **PASS at product-period level for NCEI 0.5-degree only; exact four-cell file/field support unknown.** No archive objects were opened. |
| Publication-as-of and revision | [NCEI's archive page](https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast) gives cycle and forecast-valid times, not original public delivery times for each 2025/2026 file or immutable revisions; it warns that its cloud migration may temporarily delay access. The [GFS NODD registry listing](https://registry.opendata.aws/noaa-gfs-bdp-pds/) advertises live new-object notifications, but the linked cloud dataset retains only the trailing 30 days per NCEI. A 00 UTC *model cycle* is not evidence the corresponding 2025 archived bytes were public at 00 UTC, or that those bytes were never replaced. **FAIL the campaign's documented historical as-of gate.** A conservative lag cannot be chosen from these pages and asserted to prove release without dated archive/version evidence. This is a campaign provenance requirement, not an express claim that the organizer forbids all retrospective forecasts. |
| Deterministic airport join | The [organizer's dictionary](https://prc-data-challenge-2026.netlify.app/data.html) names ten airport ICAOs and `ADEP_mvt`/`MVT_TIME_UTC_mvt` for a departure's airport/takeoff clock. GFS has a geographic grid, forecast *cycle* and *valid hour*, not a shared `MVT_ID_mvt` or flight key. A potential many-to-one join would pin licensed airport coordinates, select a grid cell with a fixed spatial tie rule, then select a forecast valid near `T` from a run proven public before `T`. The challenge dictionary itself supplies ICAOs, **not** a licensed coordinate table; no cell mapping, exact archived field, airport-month completeness or input support has been verified. The takeoff clock is a retrospective challenge anchor, not proof a live pre-takeoff system knew that clock. |

Do not request NCEI GRIB bodies, build a cache, access challenge rows, fit, score
or upload from this note. Reopen only when an organizer-compatible interpretation
of the NOAA data grant and a publisher-backed immutable per-run first-publication
and revision record are available, together with exact archived forecast files
for January/July 2025 and 2026 and the rest of 2025. Then freeze an airport
coordinate version, spatial/valid-time ties, conservative publication cutoff,
field selection and ten-airport four-cell *input-only* support screen before
reading file values or outcomes. A later supervised test must use a matched
control retaining the already tested IEM weather inputs and the campaign's
unchanged F1/F3/full-season qualification gates; source availability alone
does not predict an RMSE improvement.
