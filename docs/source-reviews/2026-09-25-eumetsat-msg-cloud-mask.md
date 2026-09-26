# EUMETSAT MSG Cloud Mask: conditional source review

25 September 2026. **Distinct dense observation; HOLD before product bodies,
model fitting, ranking inference or upload.** EUMETSAT's [MSG 0-degree Cloud
Mask](https://user.eumetsat.int/catalogue/EO:EUM:DAT:MSG:CLM), collection
`EO:EUM:DAT:MSG:CLM`, is a 15-minute, nominal 3-km regional cloud/clear-sky
classification. The [product guide](https://user.eumetsat.int/s3/eup-strapi-media/pdf_clm_pg_6b00d53e6b.pdf)
and [GRIB code table](https://codes.ecmwf.int/grib/format/grib2/ctables/4/217/)
define 0 clear water, 1 clear land, 2 cloud, 3 no data and 255 missing. A
future airport-region feature must keep valid clear separate from code 3,
missing GRIB scans and off-disc cells. The actual file's GRIB grid navigation
and quality still need inspection before airport-pixel use.

The [structured collection policy](https://user.eumetsat.int/catalogue/es/csw/_doc/EO:EUM:DAT:MSG:CLM)
lists `Free and unrestricted - CC-BY-4.0`; `NoConditions` is the policy ID.
EUMETSAT's grant permits attributed reuse, but the [organizer prize
rule](https://prc-data-challenge-2026.netlify.app/eligibility.html#eligibility-for-prize)
does not say whether this data license satisfies its phrase "open source
license" for additional datasets. An organizer ruling is needed before
calling it a prize-eligible input.

Public [OpenSearch metadata](https://api.eumetsat.int/data/search-products/1.0.0/osdd?pi=EO:EUM:DAT:MSG:CLM)
shows all 2025 months, with **34,998 hits versus 35,040 nominal** 15-minute
slots. January/July 2025 and 2026 have **2,976 / 2,976 / 2,974 / 2,976**
hits versus 2,976 nominal in each month. A later exact-slot census found
the 12 January **18:15 and 18:30 UTC** scans absent; the earlier 10:15/10:30
UTC claim was incorrect. Exact [10:15](https://api.eumetsat.int/data/search-products/1.0.0/os?pi=EO%3AEUM%3ADAT%3AMSG%3ACLM&dtstart=2026-01-12T10:15:00Z&dtend=2026-01-12T10:15:00Z&c=10&format=json)
and [10:30](https://api.eumetsat.int/data/search-products/1.0.0/os?pi=EO%3AEUM%3ADAT%3AMSG%3ACLM&dtstart=2026-01-12T10:30:00Z&dtend=2026-01-12T10:30:00Z&c=10&format=json)
queries each return one scan; [18:15](https://api.eumetsat.int/data/search-products/1.0.0/os?pi=EO%3AEUM%3ADAT%3AMSG%3ACLM&dtstart=2026-01-12T18:15:00Z&dtend=2026-01-12T18:15:00Z&c=10&format=json)
and [18:30](https://api.eumetsat.int/data/search-products/1.0.0/os?pi=EO%3AEUM%3ADAT%3AMSG%3ACLM&dtstart=2026-01-12T18:30:00Z&dtend=2026-01-12T18:30:00Z&c=10&format=json)
return zero. Across 2025 there are 34,998 hits for 34,997 unique
15-minute starts: **43 missing starts**, offset in the hit count by one excess
hit where MSG3 and MSG4 share 12 November 13:45 UTC. Across all 14 enumerated
calendar months there are 40,948 hits, 40,947 unique starts, 45 missing
starts and one conflicting start out of 40,992 nominal slots. The
[independently reviewed census](../../../review_work/lead235_20260925/msg_cloud_census_review_v1/REPORT.md)
repeated these exact-slot results; neither run preserves a complete item-
version inventory, file integrity or valid-airport-pixel evidence. Next-month
midnight is an inclusive boundary and inflates a broad query by one. The
collection's full-disc envelope and airport-area `bbox` search do not show per-pixel
validity; even an off-disc box returns the same indexed product.

A later [full-JSON metadata audit](../../../review_work/lead235_20260925/msg_full_manifest_review_v1/REPORT.md)
covered 41,002 nominal 2025 and January/July 2026 slots including ten
month-boundary starts. Two current-index inventories retained 40,958 IDs and
MD5s at 40,957 distinct starts, 45 gaps and one November conflict, with the
same ordered ID/MD5 hash in both. Twelve consecutive 6 February 2025 items
have a non-`NA` suffix whose meaning is not documented by the inspected
product metadata; preserve the exact IDs without assigning revision or
publication semantics. Across the current items, 878 catalogue `updated`
timestamps lag sensing by more than a day, with a maximum near 26 days.
Different raw response hashes and a matching present-day MD5 inventory do
not establish immutable original versions or historical first availability.

Current item `updated` values in four sampled months follow 12:00 sensing
by about 34-39 minutes, while detail `processingDate` says 12:15 and
`productVersion=1`. The inspected public APIs do not supply an immutable
first-availability/hash and replacement history for strict historical
takeoff-time use. The competition is retrospective, but a separately frozen
retrospective-only policy and organizer interpretation would be needed to
use a version pinned before submission without claiming it was public at
takeoff. Metadata estimate about 0.6 MB per compressed Data Store file;
the full-year-plus-target catalogue exceeds 40,000 files. This is a cost
indicator, not a measured transfer or approved resource ticket.

The released V3 representation already includes METAR cloud, visibility,
rain/thunder and report-age fields. It lacks this 15-minute regional cloud
mask, so there is a distinct input hypothesis, but no measured conditional
variation or RMSE gain. A five-second local gain requires a roughly 3.69%
reduction in V3's separate, seasonally weighted 2025 complete-cohort squared
error, or 917,839,120 fewer units in its weighted numerator; this source
has not been scored. The earlier extra-time METAR candidate failed F1 July,
and is not a test of this satellite product.

**Next gate:** settle the product-license and retrospective/as-of questions;
freeze licensed airport points, source version selection, actual decoded GRIB
navigation, one region/scan cutoff, code-3 abstention, resource budget and a
bounded input-only sample. Then independently check valid clear/cloud and
missing support in all ten airports across January/July 2025 and 2026,
including variation beyond the existing METAR fields, before designing a
matched F1/F3 model test. The [source audit](../../../review_work/lead235_20260925/dense_mtg_source_v1/REPORT.md),
[rights/metadata cross-check](../../../review_work/lead235_20260925/msg_cloud_rights_v1/REPORT.md)
and [format/materiality review](../../../review_work/lead235_20260925/msg_cloud_semantics_v1/REPORT.md)
contain the source queries and stop rationale. No product body, challenge
row, target, model, score or official submission was read or produced.
