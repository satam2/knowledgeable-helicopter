# Dependency And Reuse Attribution

Project source is licensed under the existing GNU GPLv3 LICENSE.

The original feature diagnostic supplied with the user's execution plan was
ported into explicit tested functions in `benchmark.py` and `features/`.
Its original source SHA-256 is recorded in `reports/prior_evidence.json`.
The production observation contract, temporal training/refit protocol,
residual/fallback routing, validation, artifact packaging and release workflow
were implemented in this repository. No prior team's competition solution was
adapted. CatBoost and the following packages are standard software dependencies.

| Dependency | Upstream | License |
|---|---|---|
| CatBoost | https://github.com/catboost/catboost | Apache-2.0 |
| NumPy | https://github.com/numpy/numpy | BSD-3-Clause |
| pandas | https://github.com/pandas-dev/pandas | BSD-3-Clause |
| Apache Arrow / PyArrow | https://github.com/apache/arrow | Apache-2.0 |
| scikit-learn | https://github.com/scikit-learn/scikit-learn | BSD-3-Clause |
| PyYAML | https://github.com/yaml/pyyaml | MIT |
| psutil | https://github.com/giampaolo/psutil | BSD-3-Clause |
| tzdata | https://github.com/python/tzdata | Apache-2.0; IANA database public domain |
| pytest | https://github.com/pytest-dev/pytest | MIT |

Transitive dependencies are pinned in `requirements.lock.txt`. Installed package
metadata and bundled license files remain authoritative for their distributions.
No challenge-data redistribution license is assumed. No external weather,
airport layout, tracking or other dataset is used. The IANA timezone database is
a software dependency for historical local calendar conversion.
