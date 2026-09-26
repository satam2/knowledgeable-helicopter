# Leading-team public method disclosures: bounded check

25 September 2026 Pacific (26 September UTC). **No new method admitted.** This
was a public-source check, not a fit, score, submission, or inference about
hidden labels. Fifteen completed HTTP requests covered the challenge pages,
their embedded leaderboard display, and eight first-party team profiles. The
score-band selection came from the display's [public mirror](https://prc-challenge-2026.vercel.app/)
and its linked data feed, not a separate verification against the organizer's
leaderboard API. Scores below identify profiles for this search only; they do
not attribute performance to any method.

| Profile | Displayed best RMSE | What the first-party profile actually discloses |
| --- | ---: | --- |
| [gentle-tractor](https://ansperformance.eu/study/data-challenge/dc2026/teams/gentle-tractor.html) | 235.92 | Interest in learning about airport A-CDM; no implemented input or algorithm stated. |
| [strong-crane](https://ansperformance.eu/study/data-challenge/dc2026/teams/strong-crane.html) | 235.94 | Aviation research interest; no method stated. |
| [gentle-igloo](https://ansperformance.eu/study/data-challenge/dc2026/teams/gentle-igloo.html) | 236.32 | Describes an unrelated aviation-occurrence corpus and views airport-state reconstruction as data engineering; no challenge-specific implementation disclosed. |
| [jovial-uniform](https://ansperformance.eu/study/data-challenge/dc2026/teams/jovial-uniform.html) | 237.55 | General ML and reproducibility aspiration; no specific feature, target-safe procedure, or validation design. |
| [upstanding-firefly](https://ansperformance.eu/study/data-challenge/dc2026/teams/upstanding-firefly.html) | 240.55 | Mentions past OpenSky/noise work, gradient boosting, and an intention to publish GitHub code and a JOAS write-up; no challenge code or write-up linked in its profile. |
| [enthusiastic-daisy](https://ansperformance.eu/study/data-challenge/dc2026/teams/enthusiastic-daisy.html) | 243.01 | Trajectory/deep-learning research background and intent to explore airport-surface factors; no implemented challenge method in the full rationale. |
| [gentle-lemon](https://ansperformance.eu/study/data-challenge/dc2026/teams/gentle-lemon.html) | 243.87 | General interest in experimenting and comparing approaches; no implemented challenge method in the full rationale. |
| [youthful-giraffe](https://ansperformance.eu/study/data-challenge/dc2026/teams/youthful-giraffe.html) | 244.12 | Brief participation rationale; no method stated. |

The team pages' main-content links were checked and supplied no external
repository, paper, or write-up URL. The initial extraction printed only the
first 950 characters; a subsequent read through the closing `main` tag checked
the two longer [enthusiastic-daisy](https://prc-data-challenge-2026.netlify.app/teams/enthusiastic-daisy.html)
and [gentle-lemon](https://prc-data-challenge-2026.netlify.app/teams/gentle-lemon.html)
rationales in full. Neither gave a challenge-specific method or linked code.
The live [ranking page](https://prc-data-challenge-2026.netlify.app/ranking.html)
embeds a separate leaderboard display; a team's aggregate result is not a
disclosure of its model or evidence of an isolated feature gain. No competitor
source was downloaded or executed, and no private row, label, prediction,
credential, or organizer truth was accessed.

The earlier [public-method refresh](../../../review_work/lead235_20260924/public_method_refresh_v1/REPORT.md)
already examined actual public code for Kind-Mango and Victor Alcadi, including
route/type-centered duration, CatBoost blending, and OPDI event possibilities.
The subsequent [method delta](../../../review_work/lead235_20260925/public_method_delta_v1/REPORT.md)
and [gap screen](../../../review_work/lead235_20260925/public_method_gap_v1/REPORT.md)
identify the arrival-grid idea as unadmitted and the centered-duration/blend
work as already locally evaluated; daily ATFM remains held on source rights
and historical timing. None of the eight profiles changes those dispositions.

**Decision:** stop this team-profile method lead. Do not infer a training
recipe from ranks or team research interests, and do not fit or upload a model
from these descriptions. Reopen only on a dated, first-party release of
challenge-specific code or a method write-up with a distinct, lawful input or
testable target-safe contrast; assess it against the campaign ledger before
any supervised work.
