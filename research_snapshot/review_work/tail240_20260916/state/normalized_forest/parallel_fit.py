"""ExtraTrees parallel fit with single-thread deterministic prediction sums."""
from sklearn.ensemble import ExtraTreesRegressor


class ParallelFitExtraTrees(ExtraTreesRegressor):
    def fit(self, X, y, sample_weight=None):
        self.set_params(n_jobs=4)
        try:
            result = super().fit(X, y, sample_weight=sample_weight)
        finally:
            self.set_params(n_jobs=1)
        return result
