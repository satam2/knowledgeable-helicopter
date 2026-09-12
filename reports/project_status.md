# Project Status

Current phase: local technical completion achieved; GPLv3 source and reproduction
documentation included in this repository; competition delivery pending.

- Phases 0-1 tested: rules verified; complete input hashes; 4,167,797 movements and 2,085,047 departures audited; all departure target identities exact.
- One unlabelled arrival and six file-boundary spillovers documented; raw files unchanged.
- Phase 2 tested: exact legacy benchmark reproduced; 344,841-row anytime artifact validated.
- Phases 3-4 tested: direct/residual/fallback seasonal comparisons, G1, paired airport-day stability, anomaly-policy screens and refitted traffic ablation complete.
- Thirteen contract/model tests pass. Failed full-data and Newton-Huber attempts retained.
- Selected: residual_long_proxy_specialist. F1/F2/F3 RMSE 369.370/269.313/275.712s; seasonal score 331.176s; G1 285.575s.
- Frozen settings/source: reports/freeze.json and configs/release.yaml. Tail limitations and selection: reports/selection_decision.md.
- Resource exception: 250k direct/residual training cap after full-data timeout; missing-proxy specialist uses all eligible cohort rows. Every score row is retained.
- C1 passed once: RMSE 266.406s on 165,677 December rows; missing-proxy RMSE 1222.068s. This remains reused confirmation.
- Final fit complete: release-20260911T182332345949-6c34f3be; direct 250,000 rows, residual 247,127 rows, specialist all 22,470 missing-proxy rows.
- Final Parquet passed: 344,841 exact template IDs, correct double/int32 schema and finite values. Repeated inference maximum difference 0.0 seconds.
- Submission SHA-256: 823408382b15b41a5b353b219ef7e5ebbfb07e81e8c3c833f3ebf66295c88e13.
- Cold reconstruction passed in data/interim/clean-reproduction-20260911T183003, with empty initial caches and a freshly installed wheel: 13 tests passed; serialized values and Parquet bytes exactly match the reference hash.
- Publication verification on 2026-09-12 passed from an isolated Git checkout and fresh wheel: all 41 frozen file hashes matched, 13 tests passed, and cold reproduction matched the exact prediction hash. See reports/publication_verification.json.
- Local source and model/prediction archives passed CRC checks; no raw challenge files included. See reports/package_validation.json.
- Model card, selection report, configuration registry, validation figure and paper outline are complete.
- Local technical completion: complete. Accepted competition upload: not performed. Prize eligibility confirmation: pending.
- GPLv3 source and reproduction documentation are prepared for https://github.com/satam2/knowledgeable-helicopter; raw files and individual predictions are excluded.
- External submission acceptance remains pending verified team/account details, bucket access and the next version.

Reproduce: `.venv/Scripts/python -m taxiout.cli reproduce --config configs/release.yaml --offline`.

Next competition action requires verified team/account details and bucket access;
use the official team/version filename and verify upload acceptance separately.
