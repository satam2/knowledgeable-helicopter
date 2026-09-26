# MSG Cloud Mask: independent history mechanisms

26 September 2026 Pacific. **Strict historical as-of gate remains HOLD** for
`EO:EUM:DAT:MSG:CLM` in 2025 and January/July 2026. This follow-up examined
the publisher's notification service and legacy archive policy, distinct from
the current Search/Browse/item routes checked on [25 September](2026-09-25-msg-publication-history-gate.md).

- EUMETSAT's [Data Store detailed guide, V7.3 (2 September 2026)](https://user.eumetsat.int/resources/user-guides/data-store-detailed-guide)
  describes Data Store Notifications (DSN) as an MQTT feed of **newly
  published** products. Its `--notifications-only` example exposes a product
  ID and `properties.pubtime`. The guide describes active subscriptions and
  deferred local downloads, but does not document a public replay/export of
  2025/2026 publication events, predecessor hashes, replacement events or
  withdrawals. The [EUMDAC guide](https://user.eumetsat.int/resources/user-guides/eumetsat-data-access-client-eumdac-guide)
  lists publication-time filters for subscriptions and locally managed order
  files; neither is stated to be a publisher-certified historical manifest.
  A `pubtime` observed on a live event could timestamp that publication event,
  but alone would not certify first publication, exact bytes or later changes.
- EUMETSAT's [Data Centre transition notice, dated 25 June 2024](https://user.eumetsat.int/news-events/news/transition-from-the-data-centre-to-the-data-store)
  specifically places MSG High Rate SEVIRI and **Cloud Mask** in the
  25 June 2024 stage of removing Data Centre *user access* for collections
  available on Data Store. It says the online Data Store generally offers only
  the best-quality product for an orbit/slot, unlike the tape-based Data Centre;
  reprocessed data are offered by default, with other timeliness classes filling
  gaps. The [Data Centre introductory guide](https://user.eumetsat.int/resources/user-guides/data-centre-introductory-guide)
  describes long-term preservation, but warns its ordering instructions may be
  outdated and points to that transition notice. This does not prove that
  EUMETSAT discarded underlying tape data or lacks internal event logs. It does
  mean the published independent self-service archive does not establish a
  retrievable original CLM version for these windows.
- The [Data Store detailed guide](https://user.eumetsat.int/resources/user-guides/data-store-detailed-guide)
  allows unauthenticated catalogue browsing/searching. Downloads require an
  EUMETSAT account and time-limited OAuth token even for unrestricted products.
  The DSN service is described as a **pilot** requiring separate subscription
  credentials requested from EUMETSAT Helpdesk. Its NRT-licence example names
  MSG SEVIRI/MTG FCI1C; it does not establish a CLM-specific NRT permission.
  The separate [CLM collection policy](https://user.eumetsat.int/catalogue/es/csw/_doc/EO:EUM:DAT:MSG:CLM)
  says `Free and unrestricted - CC-BY-4.0`, subject to the campaign's independent
  prize-use interpretation.

**Next source gate:** a first-party immutable historical DSN/dissemination
export or genuinely contemporaneous preserved records for each chosen CLM
item must tie first public availability to an exact item/version checksum and
enumerate later replacements/withdrawals, with authorized access to the same
bytes. Neither published live-feed instructions nor the migrated Data Centre
provide that proof. A separately approved retrospective-only snapshot policy
would be a different experiment and would not clear strict as-of or prize-use
rights.

Scope: public publisher documentation and metadata only; no product bodies,
login, credentials, challenge data, fit, score, organizer contact or upload.
