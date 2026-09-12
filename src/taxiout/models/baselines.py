import numpy as np
import pandas as pd

from taxiout.schema import ID, TARGET, align


class HierarchicalMean:
    def __init__(self, alpha=30):
        self.alpha = alpha
        self.levels = [["ADEP_mvt"], ["ADEP_mvt", "RUNWAY_mvt"], ["ADEP_mvt", "RUNWAY_mvt", "STAND_mvt"]]

    def fit(self, x, labels):
        frame = align(x.reset_index(), labels, [TARGET])
        self.mean = float(frame[TARGET].mean())
        self.tables = []
        for cols in self.levels:
            grouped = frame.groupby(cols, observed=True)[TARGET].agg(["sum", "count"])
            self.tables.append({(key,) if not isinstance(key, tuple) else key: (float(row["sum"]), int(row["count"])) for key, row in grouped.iterrows()})
        return self

    def predict(self, x, depth=3):
        result = np.full(len(x), self.mean)
        for cols, table in zip(self.levels[:depth], self.tables[:depth]):
            values = [table.get(key, (0.0, 0)) for key in x[cols].itertuples(index=False, name=None)]
            sums = np.array([v[0] for v in values])
            counts = np.array([v[1] for v in values])
            alpha = 0 if len(cols) == 1 else self.alpha
            supported = counts > 0
            result[supported] = (sums[supported] + alpha * result[supported]) / (counts[supported] + alpha)
        return result

    def fit_transform_oof(self, x, labels, folds=5, seed=20260910):
        # This API is only needed if priors become model inputs. Standalone means do not use it.
        rng = np.random.default_rng(seed)
        assignment = np.empty(len(x), dtype=int)
        assignment[rng.permutation(len(x))] = np.arange(len(x)) % folds
        result = np.empty(len(x))
        aligned = align(x.reset_index()[[ID]], labels, [TARGET])
        for fold in range(folds):
            score = assignment == fold
            estimator = HierarchicalMean(self.alpha).fit(x.loc[~score], aligned.loc[~score])
            result[score] = estimator.predict(x.loc[score])
        self.fit(x, labels)
        return result
