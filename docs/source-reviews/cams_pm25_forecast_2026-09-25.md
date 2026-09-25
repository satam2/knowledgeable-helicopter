# CAMS PM2.5 forecast: source gate

25 September 2026. **STOP before data acquisition or feature fitting.** The
Copernicus Atmosphere Monitoring Service (CAMS) global PM2.5 forecast is a
distinct historical modeled field, but its exact as-of-query
availability and ten-airport coordinate join are not established. This is a
documentation review only; it reads no challenge row, forecast value, label,
model, score or ranking prediction.

## First-party evidence

| Requirement | Established fact | Limit |
| --- | --- | --- |
| Historical source | The [CAMS product catalogue](https://ads.atmosphere.copernicus.eu/datasets/cams-global-atmospheric-composition-forecasts?tab=overview) describes a global 0.4-degree grid, surface fields hourly, archive **2015 to present**, and new 00/12 UTC forecasts daily. Its [machine-readable collection](https://ads.atmosphere.copernicus.eu/api/catalogue/v1/collections/cams-global-atmospheric-composition-forecasts) has a global bounding box and a temporal extent spanning the required January/July 2025 and 2026 periods. The [ECMWF dataset documentation](https://confluence.ecmwf.int/x/5cqpD) lists surface `particulate_matter_2.5um` (`pm2p5`, parameter 210073) in kg/m3. | This proves product-level date and geographic coverage, not successful retrieval or nonmissing PM2.5 at all ten airport cells in all four months. No archive values or month-cell support were sampled. |
| Reuse rights | The collection specifies `CC-BY-4.0`, linked to its [product licence text](https://object-store.os-api.cci2.ecmwf.int:443/cci2-prod-catalogue/licences/cc-by/cc-byv1_2b61eb0b42e053566cb9447c1d2847a69a275c095e00fca00bad1bf5326a9432.md). The [Creative Commons deed](https://creativecommons.org/licenses/by/4.0/) expressly permits sharing and adaptation even commercially, subject to attribution. | The [challenge eligibility page](https://prc-data-challenge-2026.netlify.app/eligibility.html) says additional datasets must be openly available **under an open source license** and results reproducible. CC-BY-4.0 is an explicit open *data/content* licence, but the organizer has not been shown to accept that wording as satisfying its literal condition. Do not claim prize-use clearance from the catalogue alone. |
| Observation and publication time | The [ECMWF documentation](https://confluence.ecmwf.int/x/5cqpD) distinguishes 00/12 UTC forecast base times from hourly surface valid times, and says 00 UTC forecasts are guaranteed available by 10:00 UTC and 12 UTC forecasts by 22:00 UTC. | The same page warns ADS delivery can vary because it is non-operational and directs time-critical users to separate dissemination services. The catalogue says there is only one dataset version, with occasional model upgrades; it does not supply per-run first-publication timestamps, an immutable release log, or a guarantee that a currently archived value equals what ADS served at historical departure time. A base time or valid time is not a publication receipt. |
| Exact challenge join | The [challenge data dictionary](https://prc-data-challenge-2026.netlify.app/data.html) defines `MVT_ID_mvt` as movement ID, `ADEP_mvt` as departure ICAO airport, and `MVT_TIME_UTC_mvt` as DEP takeoff. Ranking covers January/July 2026 and blanks only DEP block/taxi targets. A possible deterministic feature would retain every `PHASE_mvt=DEP` ID, map `ADEP_mvt` to a frozen WGS84 airport point, choose one 0.4-degree grid cell with a frozen tie rule, then use an eligible 00/12 run and a surface forecast valid hour no later than that DEP's `T`. | The challenge data dictionary supplies ICAO codes, not airport coordinates. No ten-airport point table with an applicable licence and pinned version was verified here. The forecast-run eligibility rule also needs a defensible release cutoff; selecting by base time alone would leak a run published after `T`. This would be a spatial/time join, not a matching external flight ID. |

The forecast candidate is PM2.5 concentration, not a METAR report, flight
event, or another unfiltered IEM observation. The global grid may characterize
regional haze but has no demonstrated taxi-out effect. The supplied takeoff
clock is a retrospective query anchor; this note does not assert that a future
takeoff clock is available to a live pre-takeoff predictor.

## Reopening condition

Do not download bulk CAMS rows, write a feature cache, or fit/score a model on
this lead now. Reopen only after (1) the organizer confirms that CC-BY-4.0
meets the additional-dataset prize rule, (2) first-party evidence or archived
release receipts establish which exact forecast values were accessible at each
historical `T`, including ADS delays/revisions, (3) an openly licensed,
versioned ten-airport coordinate table and deterministic grid/time tie rule
are frozen, and (4) a bounded, input-only sample proves all four requested
airport-month support cells before a full feature run. Keep any future
source/input review separate from target access, fitting, ranking and upload.
