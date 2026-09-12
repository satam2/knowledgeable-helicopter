# Implementation Decisions

- Six movements fall 2-4 seconds before the month named by their file. Files and
  original row order are preserved. Folds use actual UTC take-off times, and
  production traffic context is grouped by actual UTC month across files.
- One November arrival lacks both block time and taxi-in label. It remains a
  movement event. All departure targets are present and their identities pass.
- No arrival taxi-in labels or block times are needed by retained features, so
  the sanitizer removes those columns for both phases. This is a stricter
  contract than the maximum ranking-visible information the rules permit.
- Custom hierarchical means are standalone baselines. They are not passed as
  in-sample target features to CatBoost; an inner-OOF API is tested for future use.
- Invalid NM proxies are negative or above 7200 seconds under a fixed declared
  routing rule. Residual candidates use the full direct model for those rows.
  Raw values are retained; predictions and score labels are not capped.
- The separately logged all-finite residual candidate permits correction of
  flagged long/negative clocks. Its routing status is recorded separately from
  the fixed 0-7200-second quality slices, preserving comparable diagnostic cohorts.
- Raw categorical values receive collision-safe, missing-distinct tokens. New
  categories are handled by CatBoost's categorical hashing and fitted priors;
  training-only vocabulary determines the new-category evaluation flag.
- A profiler located repeated year-wide airport string scans in traffic
  preparation. Grouping event arrays once preserved every January feature and
  the Parquet SHA-256 exactly. No feature was changed by this optimization.
- Month metadata is derived from integer year/month values before formatting,
  avoiding millions of unnecessary per-row string conversions.
- Audit/feature caches, ID-keyed errors, fitted models and submissions remain
  local/ignored. Public source packaging excludes raw challenge files.
- No external weather/layout source or learned ensemble is required for local
  completion. Optional additions must earn a promotion through the declared gates.
- Multiple multi-hour host/tool stalls were observed. Windows CLI jobs temporarily
  request system execution state while active, restoring it at process exit;
  display state and persistent power settings are unchanged. Manifest wall runtime
  can include host suspension and should be read beside component fitting times.
