# Methods Outline

Working title: Reproducible Retrospective Taxi-Out Estimation with Observable NM Clocks.

1. Define total departure taxi-out and the retrospective information boundary.
   Distinguish airport and NM clocks, arrivals, and additional taxi-out indicators.
2. Describe supplied movements, ten reporting airports, two ranking months,
   missing NM matches, extreme raw labels and the single unlabelled arrival.
3. Explain the sanitizer, ID integrity, UTC duration/window tests and independent
   label table. Arrival context uses movement times, never hidden departure blocks.
4. Describe mean/hierarchical/proxy baselines and direct versus residual models.
   Residual sign is target minus takeoff-to-NM-AOBT proxy, with a direct fallback.
5. Document temporal fit/tune/refit/score intervals, month-isolated context,
   related-flight purges, reused December confirmation and frozen final iterations.
6. Report measured complete-cohort RMSE, SSE shares, missingness/airport/tail slices,
   key feature ablations, airport-day stability and failed complexity experiments.
7. Discuss one labelled year, absent surface trajectories and causes, source
   inconsistencies, rare tails and January/July 2026 transfer uncertainty.
8. Provide GPLv3 source, pinned environment, raw-input manifest and offline
   reconstruction recipe. Identify private raw data rather than redistributing it.

Populate numerical results only from completed run manifests and the final model
card. No causal congestion, avoidable delay, fuel or CO2 savings claims are made.
