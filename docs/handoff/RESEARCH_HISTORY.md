# Research History and Current Evidence

This document contains aggregate research findings for the team handoff. Raw data, feature caches, fitted models, individual predictions, private manifests, and original data-derived reports remain outside this Git repository. References prefixed `PRIVATE_PROJECT/` are paths in the separate authorized project workspace, not files supplied by this branch.

The latest decision is to **retain V3**. A newer chronological robustness workflow completed and passed implementation verification, but its tested model replacements failed their predictive comparisons. Keep that workflow and its negative results; do not treat the most recently trained model as the accepted model.

## Task and Current Result

The challenge estimates departure taxi-out time from supplied retrospective observations. With takeoff T, hidden airport off-block B, and observed NM off-block N, the target is Y = T-B and the proxy is P = T-N. The fitted residual Y-P = N-B represents disagreement between off-block sources. Physical taxi delay and clock/source reliability are related but different modeling questions.

The authorized dataset contains twelve 2025 training months, with 2,085,047 departures. Ranking contains 344,841 departures from January and July 2026. Ranking departure targets and airport off-block values are hidden. Available arrival fields have a distinct phase-specific contract; they must not be confused with hidden departure labels. No 2024 labels are available in this project.

| Recipe | Local Seasonal Development RMSE | Official RMSE |
| --- | ---: | ---: |
| Original release | 331.175989 seconds | A historical earlier submission scored 348.0775; inspected evidence did not bind its uploaded bytes tightly enough to treat this as another calibrated recipe pair |
| V2 | 292.813346 seconds | 294.626000 seconds |
| V3, retained | 268.662991 seconds | 278.014600 seconds |

V3 was accepted on all 344,841 required ranking pairs and improved official RMSE by 16.611400 seconds over V2. No later robustness candidate was submitted. The local and official values concern different years, months, and fitted model instances. Local 268.662991 is not an official-score forecast.

Reaching official 240 from 278.0146 requires approximately 25.5% less MSE. The earlier 230-second objective and later 240-range aim remain ambitions, not established attainable outcomes. A label-aware oracle, perfect-subgroup scenario, generic model benchmark, or leaderboard observation does not prove an achievable score for this project.

## Retained V3 Model

The ordinary route is finite proxy in [0,7200], inclusive. Its active mixture is:

| Expert | Input Fields | Final Mixture Weight |
| --- | ---: | ---: |
| LightGBM union context | 387 | 0.438586999 |
| PLE8 neural ensemble | 387 | 0.266155730 |
| CatBoost | 225 | 0.153580442 |
| Standard TabM | 225 | 0.106213044 |
| LightGBM longer ordered context | 449 | 0.035463785 |

Other finite proxies use a separate 225-field LightGBM. Missing proxies use the fixed nested composition `0.5625 * V2 + 0.1875 * ExtraTrees + 0.25 * normalized`. The inherited V2 missing route includes its global missing specialist and three fixed Rome mixture members. It does not require V2's ordinary residual/direct/gate predictions as V3 inputs.

New finite-clock final models fitted all 2,062,577 eligible 2025 rows. Missing-clock experts used all 22,470 eligible rows. Preprocessing was fitted on permitted training data, labels were retained without clipping, and final output used exact template order/schema with nearest-even int32 rounding. Final coefficients combine earlier tune-fitted season coefficients; training lengths follow a frozen integer-median rule. They were not optimized against ranking outcomes.

The current model is supported by independent source/preprocessing reconstruction, native inference replay, route/composition checks, and accepted complete-cohort output. These checks establish implementation integrity. They do not prove absence of overfitting or stable transfer to every later period.

## What Produced the Local Progress

The complete V4 development composition reached 272.263967 before the final normalized-missing and PLE387 changes. These were the subsequent complete-cohort steps:

| Composition | Seasonal RMSE | Incremental Gain |
| --- | ---: | ---: |
| Frozen V4 reference | 272.263967 | - |
| Add fixed 25% normalized missing-clock blend | 269.863486 | 2.400480 seconds |
| Replace PLE225 component with PLE387 | 268.662991 | 1.200495 seconds |

All 353,045 original July/November score rows and raw labels were retained. Seasonal RMSE is the square root of weighted monthly MSE, using summer weight 192122/344841 and winter weight 152719/344841. It is not an average of monthly RMSEs or an unweighted pooling of the local rows.

Several matched comparisons clarify the source of progress:

- Expanding LightGBM from 30 to 225 observed fields improved seasonal RMSE by about 10.87 seconds at the smaller capacity setting and 11.38 at the larger setting. Changing the capacity package improved the corresponding settings by about 2.31 and 2.82 seconds. The packages change several hyperparameters together, so this is not an isolated leaf-count effect.
- Ordered neighboring-event fields supplied useful signal. Removing 24 peer departure-clock channels eliminated about 98.7% of the observed improvement in one matched 600-tree context experiment. This concerns that feature contrast, not 98.7% of the final model's total gain or proof of a physical queue mechanism.
- Matched PLE225-to-PLE387 information expansion improved both earlier tune months and later fixed-epoch score refits. Its signal partly overlaps other experts, leaving a smaller ensemble increment than standalone gain.
- A fixed-capacity two-point learning curve found 816,060 fitting rows better than a deterministic half sample on the same June finite-clock cohort by 6.118942 seconds. This supports benefit from more data for that model; it is not a scaling law or a prediction of all-year transfer.
- Some ordinary model gains replicated under an alternative seed, but this was component replication, not a complete independently seeded pipeline study.

## Tested Directions and Limits

The following is a selected evidence index, not a count of independent discoveries. Candidate variants share data, experts and development decisions. In one bounded earlier campaign there were 30 completed model/fold records and 38 correlated seasonal candidate/variant comparisons; the project-wide search was broader, and no effective independent-trial count was established.

| Direction | Evidence and Disposition |
| --- | --- |
| Matched boosting families | LightGBM, CatBoost and XGBoost were evaluated. LightGBM/CatBoost contributed to the retained ordinary mixture; XGBoost's global coefficient was effectively zero. A model-family name alone did not predict usefulness. |
| Modern neural/tabular models | Standard TabM and PLE supplied complementary signal. Bounded RealMLP, recurrent-context and foundation-model studies did not justify replacing the retained ensemble. Sampled/reference-limited results were not represented as full-data training. |
| Larger PLE ensemble | A 32-member PLE experiment lost about 0.392 seconds on its primary June ensemble endpoint and did not advance to the second fold. More members were not automatically better. |
| Neural relation attention | The tested June candidate lost about 1.017 seconds against the current PLE387 ensemble replacement endpoint. No second-fold advancement followed. |
| Richer CatBoost inputs | Expanding to 387 fields improved the matched-family tune comparison by about 4.755 seasonal seconds, but only about 0.463 inside the current ensemble, below its declared materiality requirement. This is distinct from the retained CatBoost225 component. |
| Nonlinear expert gating | Earlier small-expert gates were negative or mixed. A later fixed current-nine nonlinear selector lost about 1.119 seconds on its late-June comparison. These predictions retained upstream stopping exposure. |
| Additive calibration | A current-expert residual correction produced only about 0.157 seasonal late-tune seconds; removing influential rows reversed its gains. It was not promoted. |
| Per-airport capacity and linear leaves | Some matched standalone gains failed the current-ensemble endpoint or stability/materiality checks. No blanket conclusion about all airport experts or linear leaves follows. |
| Explicit physical, surface and arrival information | Availability audits, physical-reference models, arrival features and surface-data pilots were run. Some richer observed information helped, but individual physical/surface pilots did not establish the remaining large gain. Throughput counts are not queue lengths and straight-line geometry is not taxi-route distance. |
| Clock-source and retrospective context | Source conventions and peer context were useful ingredients. Particular following-event, clock-anchor, exact-flight arrival, weather/ATFM and clock-innovation follow-ups were small, unstable or negative against their declared controls. Check the historical result before proposing the same idea again. |
| Historical state and missing-source transfer | Chronological state/prior machinery and several transfer hypotheses were tested. Tested simple state additions did not establish a stable ensemble improvement. Sparse tail support limited claims about known-clock-to-missing-clock transfer. |
| Normalized targets | The missing-clock observed-scale formulation preserves the raw-MSE objective through reconstruction and scale-squared weights. Its fixed blend helped local RMSE, whereas full replacement lost. An analogous finite-clock formulation was negative. |
| Forest missing specialists | ExtraTrees and RandomForest were tested alongside historical-template controls. ExtraTrees contributed to V3, but narrow rankings among specialist variants and much of the gain were tail-sensitive. |
| External/public method review | Public implementations and primary sources informed hypotheses. Their reported metrics, cohorts, licenses and availability assumptions were examined separately; another team's reported score is not a matched result for this data and pipeline. |

Historical status documents may say that an experiment was pending or that no V3 submission existed. Those statements refer to their snapshot. The final V3 and completed robustness reports supersede status, while the original evidence remains preserved.

## Official/Local Gap: Supported Risks

V3's official RMSE is 9.351609 seconds above local; its official MSE is 7.08275% higher. Local improvement over V2 was 24.150355 seconds, versus 16.611400 officially. None of these aggregate differences isolates a causal overfit penalty.

Concrete risks are:

- June/October labels were reused for expert stopping and mixture fitting; July/November repeatedly informed model and feature decisions. Predictions outside gradient fitting are not automatically independent of model selection. A later declaration cannot make previously inspected months untouched.
- November is a proxy for January. Main original folds forecast the next calendar month, while actual July 2026 follows six intervening months after the final 2025 training cutoff. This is a training-label recency difference, not a requirement to forecast supplied retrospective features six months ahead.
- An older model improved both F2 and G1 October checks, but those share the same outcome rows. They support some history/transfer robustness for that older model, not independent validation of the current complete recipe.
- Missing clocks account for about 1.52% of season-weighted local rows but 31.42% of remaining MSE. Rare-record severity and seasonal composition therefore matter disproportionately. Counts or observable shifts alone do not identify hidden ranking errors.
- Substituting ranking month/airport/clock-status frequencies while keeping local within-group MSE fixed increased the calculated local score by about 1.742 seconds. This is a conditional sensitivity calculation, not a measured attribution of the official gap.

On the original local score rows, mechanically replacing fold coefficients with the final averaged coefficients changed RMSE by only 0.077074 seconds; rounding added about 0.000127. Those local sensitivities do not estimate the actual 2026 effects. Aggregate official feedback supplies no row, month, route, or component errors with which to identify the remaining causes.

The newer PLE increment is more stable than the normalized-missing increment: both monthly ensemble gains survive day removal and removal of ten beneficial rows. The normalized blend supplies about 66.81% of the last improvement when decomposed in MSE, but its monthly gain can reverse after removing a few highly influential records. Bootstrap intervals are conditional on exposed data and selected predictions, not selection-adjusted guarantees or 2026 predictive intervals.

## Completed Chronological Robustness Experiment

The new workflow separated all supervised roles and kept the model frozen across its evaluation panels:

| Origin | Gradient Fit | Stopping | Fixed-Length Refit | Mixture Calibration | Evaluation |
| --- | --- | --- | --- | --- | --- |
| F1 | January-March | April | January-April | May | June, July, December |
| F3 | January-July | August | January-August | September | October, November, December |

Earlier stages were purged against declared later flight entities. Preprocessing used only the appropriate training/refit prefix. The three fixed experts were LGB387, PLE387 and CatBoost387, rather than the complete V3 ordinary set. Equal mixture was the comparator. The primary candidate was fixed 50% shrinkage of a calibration-fitted simplex toward equal weights; unshrunk simplex and individual experts were diagnostics. No nonlinear gate or shrinkage-factor search was added.

| Evaluation Panel | Primary Ordinary-RMSE Gain vs Equal |
| --- | ---: |
| F1 June | +0.043104 seconds |
| F1 July | +0.025628 seconds |
| F1 December | -0.129143 seconds |
| F3 October | +0.004575 seconds |
| F3 November | -0.021415 seconds |
| F3 December | -0.018106 seconds |

Positive means the primary mixture improved RMSE. **Both origins failed their frozen advancement gates.** The small July gain failed uncertainty/influence checks, November lost, and both December panels lost. All predeclared panels were reported; the second origin was completed unchanged after the first failed.

A historical full-cohort hybrid replacing ordinary predictions while retaining original exceptional predictions scored **275.356259**, versus V3 **268.662991**, a **6.693268-second loss**. It changes training history, expert membership, stopping and mixture calibration, and retains exceptional components with later historical cutoffs. It is not a matched estimate of the original procedure's optimism or a chronology-clean complete-pipeline result. Full-year V3 models were not used to predict their own 2025 training rows.

### Missing-Clock Results

Fixed ablations on the original complete score cohorts gave:

| Arm | Seasonal Complete-Cohort RMSE |
| --- | ---: |
| Current V3 | 268.662991 |
| Remove normalized correction | 271.074103 |
| Remove ExtraTrees correction | 271.184594 |
| Remove both corrections | 275.704079 |

Both components help the retained development result, but their benefit is concentrated. Removing five component-supporting rows per fold leaves only about 0.011 seconds of normalized benefit and reverses the ExtraTrees benefit. These are influence diagnostics, not a proposal to delete score rows or route on unknown errors.

A fresh chronological missing-clock experiment held inputs, earlier history and tree count fixed while comparing ExtraTrees minimum leaf 1 with leaf 20. **Leaf 20 lost on all six panels**, including every single-day removal. The normalized expert beat leaf1 in June, July and October but lost in November and both December panels. No new specialist blend was fitted. The results reject the tested stronger leaf regularization; they do not prove that all regularization is harmful or that existing choices are unbiased.

### Verification and Reproducibility Limits

The workflow passed 27 focused tests. All 12 ordinary selection/refit stages and six missing final models passed independent contract checks. Ordinary models had bounded independent raw-feature native replay; missing models had complete query replay. Producer same-device reloads were exact. Neural cross-device differences stayed below 0.000367 seconds under the declared 0.01-second tolerance.

There were 20 fitting calls: 12 ordinary fits/refits, four fixed forests, and two normalized selection/refit pairs. Peak ordinary process memory was 12.42 GiB and peak missing-process memory 1.53 GiB. Jobs used two CPU threads, one heavy GPU job at a time, and memory watchdogs. A startup guard rejected one attempt before fitting; it later succeeded unchanged. A scorer metadata-index failure was preserved and repaired in a separate version without changing predictions, labels, model settings or acceptance gates.

This establishes a checked within-run chronology harness and rejects the tested candidates. It does not erase historical data exposure, cure every overfitting risk, or prove retained V3 free of overfitting.

## Agreed Next Work

1. **Observable source reliability with matched controls.** Define a substantive new observable distinction or error mechanism, then test its incremental contribution inside the current ensemble and complete routes. Review previous clock, context, state, exact-flight, surface and calibration failures first. Do not assume that another architecture or deeper tree supplies the missing information.
2. **Matched V3 chronology and transfer.** Apply the verified role-separation machinery to the actual V3 expert and exceptional-route recipe, matching information, capacity and history closely enough to interpret the contrast. Report short-gap and long-gap checks with explicit exposure caveats. The completed three-expert experiment did not perform this matched comparison.
3. **Seasonal weighting later.** January-through-July emphasis and January-plus-July emphasis are different untested hypotheses. Predeclare one interpretation and one bounded training-weight rule against uniform weighting, retaining all rows and raw unweighted stopping/evaluation objectives. Do not optimize weights against two official scores. One year cannot directly validate same-season next-year benefit: January lacks earlier-year history, and a July holdout cannot also be a July training sample.

The current summer refit contains only January-April, so first-seven-month emphasis would be uniform. The autumn refit principally tests August downweighting. A later-origin design can provide a limited additional contrast, but it remains exposed development evidence. Competition season aggregation weights are not automatically appropriate training weights.

No new upload follows automatically from any of these plans. Freeze the candidate set, information contract, supervised roles, routes, metrics and decision gates before fitting. Preserve failures and distinguish engineering acceptance from predictive improvement.

## Private Evidence Map

These paths require the separate authorized private workspace. They are not included in this Git branch merely because they are named here.

| Private Reference | Purpose |
| --- | --- |
| `PRIVATE_PROJECT/output/tail240_20260916/FINAL_SUBMISSION_RESULT.md` | Accepted V3 result and final fitting summary |
| `PRIVATE_PROJECT/private_runs/tail240_20260916/final_submission_v3_v2/` | Frozen V3 release and independent acceptance/composition evidence |
| `PRIVATE_PROJECT/private_runs/tail240_20260916/final_ordinary/v2/` | Full-year tree and standard TabM models/encoders |
| `PRIVATE_PROJECT/private_runs/tail240_20260916/state/final_ple387/model_v2/` | Full-year PLE387 model and protocol |
| `PRIVATE_PROJECT/private_runs/tail240_20260916/forensics/final_missing/v1/` | Full-year missing-clock fitted components and preparation |
| `PRIVATE_PROJECT/output/tail240_20260916/NEURAL_INTEGRATION_RESULT.md` | Local current composition and incremental PLE evidence |
| `PRIVATE_PROJECT/output/tail240_20260916/SCORE_GAP_ANALYSIS.md` | Official/local gap diagnosis and conditional sensitivity calculations |
| `PRIVATE_PROJECT/output/tail240_20260916/robustness/RESULTS.md` | Completed ordinary and missing-clock robustness results |
| `PRIVATE_PROJECT/private_runs/tail240_20260916/robustness/validation/` | Independent cohort, expert, calibration, evaluation and tail receipts |
| `PRIVATE_PROJECT/output/tail240_20260916/robustness/seasonal_design.md` | Seasonal hypotheses, chronological matrix and limits |
| `PRIVATE_PROJECT/output/tail240_20260916/PIPELINE_REVIEW.md` | Later feature/model tests, negative evidence and implementation audits |
| `PRIVATE_PROJECT/output/breakthrough_20260916/REPRODUCE_CLOSING.md` | Broader campaign inventory and historical reproduction index |
| `PRIVATE_PROJECT/output/breakthrough_20260916/research_gap/model_shortlist_final_1402.md` | Model/source shortlist and tested versus untested distinctions |
| `PRIVATE_PROJECT/private_runs/campaign_20260916/experiment_registry.json` | Bounded earlier campaign records and preserved failures |

Private fitted models and cached matrices are not distributed through this branch. Existing native replay scripts may require those private assets, matching runtime versions, and original source-path reconstruction. Preserved historical manifests describe the executed workspace; do not rewrite their hashes or provenance to conceal missing artifacts. Consult the branch handoff instructions for source layout and dependency setup before attempting reproduction.
