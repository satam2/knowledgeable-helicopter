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
hits versus 2,976 nominal in each month. The January 2026 catalogue lacks
the 12 January 10:15 and 10:30 UTC scans. Search hits are not a unique-ID,
file-integrity or valid-airport-pixel audit; next-month midnight is an
inclusive boundary and inflates a broad query by one. The collection's
full-disc envelope and airport-area `bbox` search do not show per-pixel
validity; even an off-disc box returns the same indexed product.

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
reduction in V3's separate 2025 complete-cohort squared error; this source
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
