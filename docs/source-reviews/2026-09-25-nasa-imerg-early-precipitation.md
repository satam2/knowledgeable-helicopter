# NASA IMERG Early precipitation: conditional source review

25 September 2026. **Distinct, provisionally reusable, but held before data-value
acquisition or fitting.** The [GPM IMERG product page](https://gpm.nasa.gov/data/imerg)
describes satellite-derived precipitation estimates, updated every half hour;
its Early Run has nominal four-hour latency. The specific [NASA CMR collection](https://cmr.earthdata.nasa.gov/search/collections.umm_json?short_name=GPM_3IMERGHHE&version=07)
is `GPM_3IMERGHHE` V07 (DOI `10.5067/GPM/IMERG/3B-HH-E/07`), a global
half-hourly 0.1-degree by 0.1-degree precipitation field. This measures
rainfall intensity near an airport, not the supplied airport METAR's binary
rain/thunder report or an airport movement's hidden off-block time. V3's
387-field receipt already includes **18 METAR-derived weather fields**, among
them rain, snow, thunder, visibility, cloud, wind, humidity, report age and
availability. Incremental prediction value remains unmeasured.

| Gate | First-party evidence | Decision |
| --- | --- | --- |
| Product calendar and geography | CMR advertises global coordinates (-180..180 longitude, -90..90 latitude) and a 1998 starting date. Metadata-only [granule-index searches](https://cmr.earthdata.nasa.gov/search/granules.json?short_name=GPM_3IMERGHHE&version=07&temporal=2026-07-01T00%3A00%3A00Z%2C2026-07-31T23%3A59%3A59Z&page_size=1) returned `CMR-Hits: 1488` for **each** January/July 2025 and 2026, equal to 31 days times 48 intervals. Sample identifiers began on the first day of each month; 2025 samples were V07B and July 2026 was V07C. | **Product-level pass only.** No actual precipitation pixel, ten-airport nonmissing rate, scan chronology, or query join was measured. The public index is not an airport-cell availability receipt. |
| Reuse rights | [NASA Earthdata data-use guidance](https://www.earthdata.nasa.gov/learn/use-data/data-use-policy) says **data provided from a NASA-led mission**, unless marked with a restriction or different license, are CC0; non-NASA data available through ESDIS retain the sponsoring organization's license. The CMR collection `UseConstraints` refers to NASA use/citation guidance, and [GPM's own data policy](https://gpm.nasa.gov/data/policy) calls GPM products freely available and requests citation. IMERG combines observations from international constellation sensors, so the NASA-led derived-product classification and any item-level exceptions need confirmation. The [challenge eligibility page](https://prc-data-challenge-2026.netlify.app/eligibility.html#eligibility-for-prize) requires external datasets to be openly accessible/usable and additional datasets under an open-source license. | **Provisionally compatible, not a release ruling.** Record exact granule-level restriction flags, derived-product CC0 applicability and any third-party terms before acquisition; resolve whether the organizer accepts CC0 under its data-license wording before official release. No attribution should imply NASA endorsement. |
| Identity and clock | The collection grid permits a deterministic airport-coordinate to gridded-pixel and UTC half-hour join. No flight ID is present or needed for an airport precipitation context. [GPM's Early/Late/Final description](https://gpm.nasa.gov/data/imerg) says Early is the low-latency run; Late uses subsequent information and Final is a delayed research product. | **Distinct retrospective candidate; deployable as-of use remains unproved.** Nominal four-hour latency is not a per-granule original-publication or immutable-version receipt. A current CMR entry may reflect later reprocessing. Do not treat later Final/Late data as available to an earlier departure. |
| Potential gain | Intense local rain can change departure flow and tail delays; 0.1-degree, half-hour estimates carry magnitude and spatial context absent from V3's binary METAR flags. | **Low-to-uncertain five-second case.** V3 already has 18 METAR weather fields plus runway/peer context, and the separately tested unfiltered extra-time IEM weather arm **lost 0.2119 seconds** versus its matched routine-only control in complete F1 July. That negative test concerned report-selection freshness, not satellite precipitation intensity; it does constrain confidence in a large weather-only gain. No IMERG model or RMSE has been measured. |

Next dependency: freeze the retrospective versus strict as-of modeling choice,
the version/publication policy and an airport-location source before an
input-only, one-use body read. Independently count finite pixels, temporal
coverage, duplicate/version conflicts, and airport-coordinate joins in all
40 airport-month cells (ten airports, January/July in 2025/2026). Quantify
precipitation variation *beyond* same-cohort binary METAR rain/thunder on
original ordered departure IDs, without opening taxi targets. Protect queries
against post-query windows, and use Early-only data and a documented conservative
publication lag for a claimed real-time feature. Freeze the same-input matched
control and whole-cohort stop rule before supervised work. An input support pass
alone does not admit F1, F3, ranking inference, or upload.

The indexed V07B 2025 examples and V07C July 2026 example show that V07 is
not a single immutable historical version. CMR `updated` metadata can reflect
later revisions; the index count alone cannot reconstruct original
publication-time values. **No observation acquisition** is admitted until the
prize-use/CC0 interpretation and the frozen, independently reviewable input-only
audit are settled. The retrospective question may be evaluated separately from
the stricter query-time publication requirement; neither is an organizer ruling.

Only public product, policy, collection, and granule-index metadata were read.
No precipitation granule body, challenge row, target, prediction, model, or
submission was accessed or produced for this review.
