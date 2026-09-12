# Rules Review

Retrieved all five official overview, data, ranking, eligibility and rationale
pages by direct HTTPS on 2026-09-10 after the browser tool transport failed.
HTTP status 200 and raw HTML SHA-256 hashes are in `rules/snapshot.json`.
No leaderboard scores/API were queried or used for model selection.

Pre-release recheck: 2026-09-11 UTC (2026-09-10 local). All five pages returned
HTTP 200. Their HTML hashes changed; the overview, data, ranking and eligibility
sections were reread and the substantive rules below still match. Current hashes
and exact check times are in `rules/snapshot.json`.

- Deadline remains 11 October 2026, 23:59:59 CET. Internal delivery target: 9 October.
- Submission: `<team-name>_v<incremental integer>.parquet`, exact template IDs and complete rows.
- Limits: five per day and 1 GB per bucket. Reset timezone is not specified on the checked page.
- Ranking: best submitted RMSE. Attempts to exploit ranking feedback are forbidden.
- Prize: public GitHub source under GPLv3, original solution, adequate reproduction documentation and open licensed additional datasets.
- JOAS publication is encouraged; eligibility page does not make it mandatory.
- Official prose still conflicts on 10/11 airports; actual table and supplied data have ten.
- Both departure block time and target are hidden in ranking. Actual take-off/NM clocks remain visible.
- Team registration, eligibility/account identity, bucket and next version are unverified.
- No upload or public publication is part of this local execution. Raw data remains excluded.

Source pages: [overview](https://prc-data-challenge-2026.netlify.app/),
[data](https://prc-data-challenge-2026.netlify.app/data.html),
[ranking](https://prc-data-challenge-2026.netlify.app/ranking.html),
[eligibility](https://prc-data-challenge-2026.netlify.app/eligibility.html),
[rationale](https://prc-data-challenge-2026.netlify.app/rationale.html).
