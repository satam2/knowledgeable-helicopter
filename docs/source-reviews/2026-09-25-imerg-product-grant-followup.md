# IMERG Early V07 product grant and granule-history follow-up

25 September 2026. This is a public-documentation and metadata check of the
[source screen](2026-09-25-nasa-imerg-early-precipitation.md) and the
[prior rights/metadata report](../../../review_work/lead235_20260925/imerg_rights_followup_v1/REPORT.md).
**Decision: HOLD the new input-only granule-body audit.** Neither a restriction
on this processed product nor a product-specific CC0 declaration was found in
the checked NASA records. The sampled current granule metadata cannot prove
the first-publication time or immutable historical precipitation bytes. The
organizer's prize interpretation and the separate airport-pixel audit remain
outside this public NASA metadata check.

## Rights: what the NASA records actually grant

* [NASA Earthdata's data-use policy](https://www.earthdata.nasa.gov/learn/use-data/data-use-policy)
  says data from a **NASA-led mission** are CC0 unless marked with a use
  restriction or license. It separately says non-NASA data available through
  ESDIS are governed by their sponsoring organization's license. The policy
  urges citation and prohibits implying NASA endorsement. This general rule
  does not name the exact IMERG Early collection or certify how a NASA/JAXA,
  multi-sensor derived product is classified for that rule.
* [NASA GPM's product policy](https://gpm.nasa.gov/data/policy) says GPM/TRMM
  data are "freely available" at levels processed by GPM, including partner
  constellation satellites at "Levels 1c through 3" (as applicable), and
  requests acknowledgment. This covers the stated processing level of the
  [GES DISC collection](https://cmr.earthdata.nasa.gov/search/collections.umm_json?short_name=GPM_3IMERGHHE&version=07),
  whose entry title is **GPM IMERG Early Precipitation L3 Half Hourly 0.1 degree
  x 0.1 degree V07**, concept `C2723758340-GES_DISC`, DOI
  `10.5067/GPM/IMERG/3B-HH-E/07`. Free availability is not itself the text of
  an explicit CC0 dedication.
* The current [CMR collection UMM-C](https://cmr.earthdata.nasa.gov/search/collections.umm_json?short_name=GPM_3IMERGHHE&version=07)
  says `AccessConstraints.Description: None`. Its `UseConstraints.Description`
  links generic EOSDIS use/citation guidance and NASA GES DISC citation guidance;
  it does **not** name CC0, a separate collection-specific grant, an
  international-sensor exception, or a flow-through restriction. The four
  UMM-G records linked in the table below have no granule-level
  `UseConstraints` property. Absence of a restriction field is not proof that
  upstream rights have been extinguished.

Thus the public NASA policy strongly favors permitted reuse of processed GPM
L3 data, and these records identify **no affirmative international-sensor
restriction on `GPM_3IMERGHHE` V07**. They do not explicitly declare this
specific derived collection CC0/public domain or resolve whether any upstream
term applies. The exact dataset-specific grant and the organizer's acceptance
of it remain unresolved; this is not a finding that NASA prohibits use.

## Publication and version evidence

The [CMR collection](https://cmr.earthdata.nasa.gov/search/collections.umm_json?short_name=GPM_3IMERGHHE&version=07)
cites a **7 June 2024** collection release, not a publication receipt for an
individual half-hour or its bytes. [NASA's V07 release notes](https://gpm.nasa.gov/resources/documents/imerg-v07-release-notes)
describe retrospective processing of the historical record and Early/Late
initial processing effective from 1 June 2024. [NASA's IMERG explanation](https://gpm.nasa.gov/data/imerg)
calls Early nominally four-hour latency but distinguishes initial from
retrospective processing; retrospective runs can use eventual inputs that
arrived too late for an initial run. A run label and observation interval do
not certify historical query-time availability.

The following **current metadata for one 00:00-00:29:59 UTC interval per target
month** was checked on 25 September 2026. All times are UTC. The table links
each exact public CMR UMM-G query and, for the first three records, the tested
revision-1 endpoint.

| Interval and metadata | Current producer suffix / CMR concept | Production | Provider `Insert` | Current CMR revision and date | Prior revision check |
| --- | --- | --- | --- | --- | --- |
| [2025-01-01](https://cmr.earthdata.nasa.gov/search/granules.umm_json?short_name=GPM_3IMERGHHE&version=07&temporal=2025-01-01T00%3A00%3A00Z%2C2025-01-01T00%3A29%3A59Z&page_size=1) | `V07B`, `G3355219957-GES_DISC` | 2025-01-01 05:04:44 | 2025-01-01 08:02:00 | 2, 2026-05-13 10:00:29 | [Revision 1](https://cmr.earthdata.nasa.gov/search/concepts/G3355219957-GES_DISC/1.umm_json): 404 |
| [2025-07-01](https://cmr.earthdata.nasa.gov/search/granules.umm_json?short_name=GPM_3IMERGHHE&version=07&temporal=2025-07-01T00%3A00%3A00Z%2C2025-07-01T00%3A29%3A59Z&page_size=1) | `V07B`, `G3587998303-GES_DISC` | 2025-07-01 05:04:23 | 2025-07-01 08:06:07 | 2, 2026-05-13 10:14:29 | [Revision 1](https://cmr.earthdata.nasa.gov/search/concepts/G3587998303-GES_DISC/1.umm_json): 404 |
| [2026-01-01](https://cmr.earthdata.nasa.gov/search/granules.umm_json?short_name=GPM_3IMERGHHE&version=07&temporal=2026-01-01T00%3A00%3A00Z%2C2026-01-01T00%3A29%3A59Z&page_size=1) | `V07B`, `G3942161168-GES_DISC` | 2026-01-01 05:04:48 | 2026-01-01 08:00:58 | 2, 2026-05-13 10:25:29 | [Revision 1](https://cmr.earthdata.nasa.gov/search/concepts/G3942161168-GES_DISC/1.umm_json): 404 |
| [2026-07-01](https://cmr.earthdata.nasa.gov/search/granules.umm_json?short_name=GPM_3IMERGHHE&version=07&temporal=2026-07-01T00%3A00%3A00Z%2C2026-07-01T00%3A29%3A59Z&page_size=1) | `V07C`, `G4232066385-GES_DISC` | 2026-07-01 05:04:31 | 2026-07-01 08:05:15 | 1, 2026-07-01 08:21:23 | Current first CMR revision only |

The [CMR search API](https://cmr.earthdata.nasa.gov/search/site/docs/search/api.html)
documents retrieval by concept and revision ID; its `all_revisions=true`
example applies to **collections**, and the granule search rejected that
parameter. The three older revision-1 URLs above returned `404` / "does not
exist" while their [current revision-2](https://cmr.earthdata.nasa.gov/search/concepts/G3355219957-GES_DISC/2.umm_json)
records were available. This says only that those earlier metadata revisions
were not retrievable through the tested public endpoint at review time; it
does not prove they never existed. The July 2026 record is currently revision
1. CMR `revision-date` dates a **metadata revision**; `ProviderDates.Insert`
is an archive-provider field and `ProductionDateTime` dates production, not
first public accessibility of immutable precipitation bytes. No sampled
record supplies a content hash plus preserved initial/revised byte versions
and public release receipts. The filename's `V07B`/`V07C` also does not bind
all 2025 and January/July 2026 granules to a single immutable value history.

These are four examples, not a query-time proof for every granule in the full
2025 or January/July 2026 calendar. With no target departure times or granule
bodies in scope, the exact airport/time joins, finite pixels, and as-of value
versions have not been tested. The campaign's stricter as-of policy is distinct
from the organizer's prize wording; neither may be inferred from today's CMR
index. **The new input-only body-read gate remains closed** pending a
publisher-level product grant/exception answer, organizer interpretation,
and a separately frozen input-only coverage/version protocol appropriate to
retrospective or as-of use. No granule body, challenge row or label,
prediction, fit, score, upload, or contact was performed here.
