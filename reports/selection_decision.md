# Frozen Selection Decision

Decision made before the C1 confirmation run: select
`residual_long_proxy_specialist` with the original 30 production features.

The direct/residual components use the bounded 250,000-row deterministic sample;
the missing-proxy component uses every eligible missing-proxy row. Raw labels are
unchanged. There is no output clipping or learned blend. The residual model is
fit on 0-7200-second proxies, and its correction is also used for long positive
proxies. Negative proxies use direct prediction; missing proxies use the specialist.

| Candidate | F1 July RMSE | F2 October RMSE | F3 November RMSE | Seasonal score | G1 RMSE |
|---|---:|---:|---:|---:|---:|
| Shared direct | 583.742 | 276.681 | 341.204 | 491.329 | 305.358 |
| Residual + direct fallback | 579.440 | 274.438 | 333.856 | 486.230 | Not shortlisted |
| Direct + missing specialist | 423.666 | 272.518 | 284.459 | 368.560 | Not shortlisted |
| Residual + missing specialist | 417.718 | 270.240 | 275.603 | 361.735 | 286.413 |
| All-finite residual + specialist | 371.489 | 268.397 | 277.561 | 333.175 | 286.541 |
| Selected long-positive route + specialist | 369.370 | 269.313 | 275.712 | 331.176 | 285.575 |

The selected recipe improves the stricter combined reference's seasonal score
by 30.559 seconds. Its F3 regression is 0.109 seconds, its missing-proxy predictions
are identical, and G1 improves 0.838 seconds. It passes the declared numerical
promotion gates. It is simpler to justify than refitting the residual learner on
all flagged clocks because it preserves the well-supported correction estimator.

The July improvement is concentrated in an extreme day-long observation on 15
July. With any one July day removed, the improvement remains 0.308-50.502 seconds,
but the airport-day bootstrap interval spans a small regression. October also
improves under every leave-one-day-out check; November changes only slightly.
G1's small gain is not statistically conclusive. These are stability diagnostics,
not evidence that extreme-event relationships will transfer to 2026. Retaining
the supplied long clock is consistent with the retrospective observation policy;
no label or timestamp is repaired. This documented tail sensitivity remains a
material limitation of the release.

## Rejected Additions

- Quality/source/precision/local-calendar features: F1 585.548s, F3 330.534s;
  seasonal score 489.292s versus 491.329s for matched direct. The 2.037s gain is
  below the declared 2.457s added-complexity threshold. Not promoted; no F2 search.
- Huber with default Newton leaves: numerically unstable one/three-tree models;
  retained as failed configurations. Gradient leaf estimation produced stable
  fits but F1 667.908s and F3 424.990s, worse than raw squared loss. Not promoted.
- Selected-architecture traffic ablation: F1 372.834s and F3 278.355s, both worse
  than the retained recipe. Keep traffic counts and month-edge coverage flags.
- Complete clock-family evidence comes from the exact refitted legacy diagnostic:
  443.973s without clocks versus 339.050s with clocks on 328,009 Nov/Dec departures.
  That uses a different training sample and continuous context; it is not a new
  independent seasonal score for the selected architecture.
- Weather, layout, learned blending, rotations and additional tail architectures
  were not attempted. The no-external-data recipe is qualified without them.

The initial development budget contains twelve distinct configurations including
the failed full-data fit and failed Newton-Huber settings, excluding reproduction
of the already specified diagnostic and the simple anytime mean artifact.
Every attempted configuration and incomplete run is retained in the registry.

## Freeze And Confirmation

Freeze the complete configuration, seed, feature policy and source hashes before
scoring December. Final component iterations are the medians of F1/F2/F3:
direct 178, residual 180, missing specialist 131. C1 may tune its own tree counts
on November under the frozen procedure but cannot change those final counts.

C1 is reused confirmation, previously exposed in the broad legacy diagnostic.
Its complete predictions/reload checks must pass, RMSE must beat the best of the
refit airport/hierarchical/raw-proxy baselines, and missing-proxy RMSE must not
exceed the airport-mean fallback. A failed gate stops final release for an explicit
investigation; no automatic architecture search or repeated December tuning.
