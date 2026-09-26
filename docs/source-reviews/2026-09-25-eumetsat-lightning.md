# EUMETSAT Lightning Imager: conditional source review

25 September 2026. **Distinct lead; STOP before observation-body acquisition,
model fitting, ranking inference, or upload.** EUMETSAT's MTG-I1 Lightning
Imager (LI) Level-2 **Accumulated Flashes**, collection `EO:EUM:DAT:0686`,
measures electrical-storm activity near airports. Its [exact catalogue
entry](https://user.eumetsat.int/catalogue/EO:EUM:DAT:0686) marks the product
operational/NRT, 2-km resolution, 3 July 2024 onward, and displays **"Free and
unrestricted - CC-BY-4.0"**. The [machine-readable collection record](https://user.eumetsat.int/catalogue/es/csw/_doc/EO:EUM:DAT:0686)
confirms `policy=Free and unrestricted - CC-BY-4.0` and maps
`policyId=NoConditions` to that same grant. The older product-navigator's
`NoConditions` is therefore a policy identifier, not a contradictory license;
the OpenSearch feed's generic `rights: Copyright` does not supersede the exact
collection policy. EUMETSAT's
[website terms](https://www.eumetsat.int/about-us/terms-use) expressly exclude
satellite products from the separate website/Learning Zone Creative Commons
grant. The [publisher's Core-data licensing guide](https://user.eumetsat.int/resources/user-guides/data-registration-and-licensing#ID-Core-data-and-products-free-access-and-unrestricted-use)
describes CC BY 4.0 Core reuse: copying, redistribution, adaptation, and
commercial use with attribution. The exact product grant is well evidenced,
but organizer interpretation remains unresolved. Data
Store access requires user-portal registration.

| Gate | Primary-source evidence | Decision |
| --- | --- | --- |
| Four calendar cells | The official [OpenSearch description](https://api.eumetsat.int/data/search-products/1.0.0/osdd?pi=EO:EUM:DAT:0686) identifies type `MTILI2AF`, platform `MTI1`, and 0-degree full-disc coverage. Public catalogue `totalResults` for overlapping month-window queries were [4,444 Jan 2025](https://api.eumetsat.int/data/search-products/1.0.0/os?pi=EO:EUM:DAT:0686&dtstart=2025-01-01T00:00:00Z&dtend=2025-02-01T00:00:00Z&c=1&format=json), [4,441 Jul 2025](https://api.eumetsat.int/data/search-products/1.0.0/os?pi=EO:EUM:DAT:0686&dtstart=2025-07-01T00:00:00Z&dtend=2025-08-01T00:00:00Z&c=1&format=json), [4,459 Jan 2026](https://api.eumetsat.int/data/search-products/1.0.0/os?pi=EO:EUM:DAT:0686&dtstart=2026-01-01T00:00:00Z&dtend=2026-02-01T00:00:00Z&c=1&format=json), and [4,426 Jul 2026](https://api.eumetsat.int/data/search-products/1.0.0/os?pi=EO:EUM:DAT:0686&dtstart=2026-07-01T00:00:00Z&dtend=2026-08-01T00:00:00Z&c=1&format=json). | **Product-level temporal presence, not completeness.** A 31-day ten-minute series has 4,464 nominal slots; overlap-query hits may include boundary records and do not establish unique slots, all-2025 continuity, or valid pixels. |
| Rights for prizes | The [exact catalogue display](https://user.eumetsat.int/catalogue/EO:EUM:DAT:0686) says CC BY 4.0, and its [structured collection metadata](https://user.eumetsat.int/catalogue/es/csw/_doc/EO:EUM:DAT:0686) records `dataPolicies[0].policy=Free and unrestricted - CC-BY-4.0` with `policyId=NoConditions`. Thus [older product-navigator](https://api.eumetsat.int/product-navigator/2.0.0/csw/_doc/EO:EUM:DAT:0686) `NoConditions` is a policy ID rather than a competing grant. [OpenSearch](https://api.eumetsat.int/data/search-products/1.0.0/os?pi=EO:EUM:DAT:0686&c=1) gives generic feed `rights: Copyright`, not an alternative product license. The [Core licensing guide](https://user.eumetsat.int/resources/user-guides/data-registration-and-licensing#ID-Data-Licensing) permits commercial reuse with attribution. The [organizer](https://prc-data-challenge-2026.netlify.app/eligibility.html#eligibility-for-prize) requires additional datasets available under an "open source license" and separately requires GPLv3 for produced code. | **Exact product CC BY 4.0: strongly evidenced; organizer prize interpretation: HOLD.** CC BY 4.0 is an open data/content grant, but no organizer determination that the challenge's literal "open source license" clause accepts it was obtained. Registration is an access step, not a competing license. |
| Historical as-of and revisions | The [OpenSearch description](https://api.eumetsat.int/data/search-products/1.0.0/osdd?pi=EO:EUM:DAT:0686) labels its `publicationDate` sort key "ingestion time". An indexed [15 Jan 2025 product](https://api.eumetsat.int/data/search-products/1.0.0/os?pi=EO:EUM:DAT:0686&dtstart=2025-01-15T12:00:00Z&dtend=2025-01-15T12:10:00Z&c=10&format=json) senses 12:00-12:10 UTC and has search `updated=12:10:40.287Z` and MD5; the [product metadata](https://api.eumetsat.int/data/download/1.0.0/collections/EO%3AEUM%3ADAT%3A0686/products/W_XX-EUMETSAT-Darmstadt%2CIMG%2BSAT%2CMTI1%2BLI-2-AF--FD--x-x--ARC-x_C_EUMT_20250115121023_L2PF_OPE_20250115120000_20250115121000_N__O_0073_0000/metadata?format=json) says `productVersion=original`, `processingDate=12:10:23Z`, but its own `updated=12:10:00Z`. A [15 Jul 2026 product](https://api.eumetsat.int/data/search-products/1.0.0/os?pi=EO:EUM:DAT:0686&dtstart=2026-07-15T12:00:00Z&dtend=2026-07-15T12:10:00Z&c=10&format=json) has search `updated=12:10:44.430Z`. | **Unresolved for strict historical as-of.** Different metadata clocks, current MD5s, and `original` version flags do not prove immutable first-publication times, replacement history, or what a past flight could have known. The project's publication-by-takeoff policy is stricter than an express organizer rule for its retrospective inputs. |
| Identity, airport support, novelty | The [challenge dictionary](https://prc-data-challenge-2026.netlify.app/data.html) supplies departure airport (`ADEP_mvt`) and takeoff (`MVT_TIME_UTC_mvt`), while ranking departure block/taxi targets are hidden. Satellite airport-time context needs a fixed airport-reference-coordinate and UTC-window join, not a flight callsign or NM ID. The catalogue reports nominal full-disc/2-km scope, but the search geometry is only a point at `(0,0)` and no ten-airport valid-pixel counts were measured. The released 387-feature set already has 18 METAR-derived `weather_T_*` fields including binary rain/thunder ([feature receipt](../../../private_runs/tail240_20260916/final_ordinary/v2/lgb387/feature_receipt.json)); [IMERG Early](2026-09-25-nasa-imerg-early-precipitation.md) is satellite precipitation, not lightning. The catalogue warns that an individual LI-AF pixel value is an event/detection variation proxy, **not** a physical flash count. | **Distinct physical input, uncertain incremental value.** Geographical footprint is plausible, but EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LTFM, and LSZH have no measured valid-pixel or query support in the four months. Existing METAR and the failed extra-IEM-weather F1 arm lower confidence in a large global gain; no LI score was measured. |

Resolve the organizer's interpretation of the documented product-specific
CC BY 4.0 grant and obtain a first-party
historical publication/replacement policy binding time and hash before opening
any observation body. Only then freeze a regional LI aggregation, airport
coordinates, strictly prior observation/publication window, resource cap, and
an input-only sample across all 40 airport-month cells, separately counting
ordinary and missing-NM departures, valid pixels, missing/revised products,
and variation beyond METAR thunder/rain. This would establish input support,
not predictive gain; complete matched F1/F3 and independent release gates
would remain separate. No LI observation file, challenge row/label, prediction,
model, or official result was read or produced for this source review.

## Follow-up gates

The [first-party LI L2 guide and AF BODY format](https://user.eumetsat.int/resources/user-guides/mtg-li-level-2-data-guide#ID-Accumulated-gridded-data)
describe **sparse** contributing lightning pixels, not a dense observed-zero
image. The [independent decoder audit](../../../review_work/lead235_20260925/li_zero_semantics_v1/REPORT.md)
found no regional valid-zero/pixel mask or first-party rule that an unlisted
pixel means a valid no-lightning observation. Positive projected-cell presence
may be usable; treating every empty airport disk as zero is not admitted.
This blocks the proposed 96-product value screen pending a documented zero,
quality and coverage rule for the actual baseline.

The [archive API audit](../../../review_work/lead235_20260925/li_archive_versions_v1/REPORT.md)
found current ingestion-like clocks and `original` version labels, but no
public immutable first-availability/hash or replacement history for strict
takeoff-time use. A separate, explicitly retrospective
[protocol](../../../review_work/lead235_20260925/li_retrospective_protocol_v1/PROTOCOL.md)
was independently [reviewed](../../../review_work/lead235_20260925/li_protocol_review_v2/REPORT.md)
as a conditional design; it still requires organizer rulings on CC BY 4.0 and
post-takeoff publication and does not admit observation bodies or fitting.
An unsent [clarification draft](../../../review_work/lead235_20260925/li_organizer_query_v1/DRAFT.md)
asks those two questions separately.

An [input-only monthly catalogue screen](../../../review_work/lead235_20260925/li_month_counts_v1/REPORT.md)
found records in all twelve 2025 months, but March and October overlap hits
were 295 and 502 below nominal ten-minute slot counts. Search hits are not
unique-slot, airport-pixel or outage receipts; exact interval enumeration is
needed before any full-year matrix. A pinned, pre-2025
[OurAirports coordinate source](../../../review_work/lead235_20260925/licensed_airport_points_v1/REPORT.md)
covers all ten ICAOs under public-domain/Unlicense terms, but its point datum,
within-airport definition and numerical accuracy are unspecified. It can
anchor a coarse input audit, not certify an individual 2-km pixel.
