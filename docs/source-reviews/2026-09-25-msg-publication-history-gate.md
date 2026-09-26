# MSG Cloud Mask: historical publication and checksum gate

25 September 2026 Pacific. **Strict historical as-of gate remains HOLD** for
`EO:EUM:DAT:MSG:CLM` (2025 and January/July 2026). The inspected public
EUMETSAT Search, Browse, item metadata, and advertised collection links expose
a current catalogue state. They do not furnish an immutable original public
dissemination timestamp paired with the exact bytes, or an enumerable chain of
prior checksums, replacements and withdrawals. This is a finding about these
public interfaces, not proof that EUMETSAT has no internal audit history.

## Endpoint evidence

| Surface | Observed contract and one delayed-item check | Historical limit |
| --- | --- | --- |
| [Collection OpenSearch description](https://api.eumetsat.int/data/search-products/1.0.0/osdd?pi=EO:EUM:DAT:MSG:CLM) | Advertises UTC sensing start/end, a `publication` parameter, and `publicationDate` sorting explicitly labelled **ingestion time**. The templates list ordinary Search parameters; no historical-as-of or revision selector is advertised. | An ingestion-time sort/filter is not an immutable first-publication-and-content ledger. |
| [Search result](https://api.eumetsat.int/data/search-products/1.0.0/os?pi=EO%3AEUM%3ADAT%3AMSG%3ACLM&dtstart=2025-02-06T12:30:00Z&dtend=2025-02-06T12:30:00Z&c=10&format=json) | The 2025-02-06 12:30 UTC item has one current hit, `updated=2025-02-10T20:41:18.63Z`, and its current MD5 is `3981db62731d2fff441656d486531dda`. | The hit does not say whether it was originally disseminated earlier, whether the same bytes existed then, or whether another version was removed. |
| [Browse item](https://api.eumetsat.int/data/browse/1.0.0/collections/EO%3AEUM%3ADAT%3AMSG%3ACLM/products/MSG3-SEVI-MSGCLMK-0100-0100-20250206123000.000000000Z-20250206124330-4883702?format=json) and [Browse collection](https://api.eumetsat.int/data/browse/1.0.0/collections/EO%3AEUM%3ADAT%3AMSG%3ACLM?format=json) | The item repeats the current `updated` and MD5; its links point to download, current JSON/XML metadata, SIP entries, and the collection. The collection advertises `dates` and the OpenSearch description. | Neither link list advertises prior item states or a predecessor checksum/revision collection. SIP entry links are product bodies and were not opened. |
| [Item `/metadata?format=json`](https://api.eumetsat.int/data/download/1.0.0/collections/EO%3AEUM%3ADAT%3AMSG%3ACLM/products/MSG3-SEVI-MSGCLMK-0100-0100-20250206123000.000000000Z-20250206124330-4883702/metadata?format=json) | `processingDate=2025-02-06T12:45:00Z`, `productVersion=1`, `methodVersion=0100`, and empty `updated`; there is no MD5 or previous-version array in this response. | Processing and version fields do not timestamp public dissemination or certify that version 1's bytes were never replaced. The numeric ID suffix is opaque absent a documented definition. |

Two attempts to pass a `start/end` interval as `publication` for this exact
item returned `InvalidParameterValue`. The [description](https://api.eumetsat.int/data/search-products/1.0.0/osdd?pi=EO:EUM:DAT:MSG:CLM)
does not specify interval syntax for that parameter; these errors establish
only that the attempted syntax is invalid. They do not show that publication
filtering, a different documented syntax, or an internal history service is
impossible. No as-of read was demonstrated.

The independently checked [full current-index manifest](../../../review_work/lead235_20260925/msg_full_manifest_review_v1/REPORT.md)
retains 40,958 distinct IDs/MD5s across the specified windows. Its two passes
have identical parsed ID/MD5 inventories but differing raw page hashes, and
878 current `updated` clocks lag sensing by more than a day. Those are useful
current-inventory checks, not predecessor hashes or first-publication proof.
The prior [rights audit](../../../review_work/lead235_20260925/msg_cloud_rights_v1/REPORT.md)
and [input protocol](../../../review_work/lead235_20260925/msg_input_protocol_v1/PROTOCOL.md)
set the stricter requirement: exact item/version bytes must have been public
by the historical takeoff, with replacement/withdrawal history. The
[source review](2026-09-25-eumetsat-msg-cloud-mask.md) records the separate
retrospective-snapshot alternative without claiming historical availability.

**Self-service decision:** Repeating current Search/Browse requests or pinning
today's MD5 inventory cannot clear strict as-of for the past. It would take a
first-party immutable item-level first-dissemination and version/hash chronology,
or genuinely contemporaneous preserved records of exact bytes and availability;
none is exposed by the inspected public routes. A separately authorized
retrospective-only policy could use a pinned current archive without an as-of
claim, but does not clear this strict gate. The organizer's CC BY 4.0 acceptance
is an independent unresolved ruling, unaffected by this API finding.

Scope: ten bounded HTTP requests, approximately 42 KB of successful response
bodies plus two short invalid-parameter responses. Only public API/documentation
metadata was read. No OAuth login, product body, challenge row/label, model,
fit, score, contact, or upload.
