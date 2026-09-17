import numpy as np
import pandas as pd

from taxiout.availability import POLICY_VERSION, assert_observations
from taxiout.features.calendar import calendar_features
from taxiout.features.clocks import clock_features
from taxiout.features.traffic import traffic_features
from taxiout.schema import CATEGORIES, ID, MOVEMENT, PHASE


def token(series):
    text = series.astype("string")
    return ("s" + text.str.len().astype("string") + ":" + text).fillna("m:")


class FeaturePipeline:
    """Label-free transforms; category support is fitted only for cohort reporting.

    CatBoost receives raw categorical tokens and learns its own ordered statistics.
    No upstream target-encoded feature or ranking distribution fitting is used.
    """

    def __init__(self, config):
        self.config = config
        self.policy = POLICY_VERSION
        self.columns = None
        self.vocabulary = {}

    def transform(self, observations, context=None):
        assert_observations(observations)
        context = observations if context is None else context
        assert_observations(context)
        dep = observations.loc[observations[PHASE].eq("DEP")].copy()
        dep = dep.set_index(ID, drop=False)
        settings = self.config["features"]
        legacy = settings["legacy"]
        x = pd.DataFrame(index=dep.index)
        for col in CATEGORIES:
            x[col] = dep[col].astype("string").fillna("MISSING") if legacy else token(dep[col])
            x[col] = x[col].astype("category")
        for location in ["stand", "runway"]:
            col = location.upper() + "_mvt"
            x["airport_" + location] = ((dep["ADEP_mvt"].astype(str) + "|" + dep[col].astype(str)) if legacy else token(dep["ADEP_mvt"]) + token(dep[col])).astype("category")
        if settings.get("flight_prefix", False):
            prefix = dep["FLIGHT_mvt"].astype("string").str.extract(r"^([A-Z]{2,3})(?=[0-9])", expand=False)
            x["flight_prefix"] = token(prefix).astype("category")
            x["flight_prefix_unparsed"] = prefix.isna().astype(float)
        x = pd.concat([x, calendar_features(dep[MOVEMENT], dep["ADEP_mvt"], settings["local_calendar"])], axis=1)
        if settings["clocks"]:
            clocks = clock_features(dep, settings["quality"])
            # Match original 27-column ordering, including the last proxy flag.
            x = pd.concat([x, clocks.drop(columns="proxy_missing")], axis=1)
        if settings["traffic"]:
            x = pd.concat([x, traffic_features(dep, context, self.config["coverage"] == "month-isolated")], axis=1)
        if settings["clocks"]:
            x["proxy_missing"] = clocks["proxy_missing"]
        for col in x.select_dtypes("number"):
            x[col] = x[col].replace([np.inf, -np.inf], np.nan).fillna(-999999).astype("float32")
        if self.columns is not None and list(x) != self.columns:
            raise ValueError("Feature schema differs from fitted pipeline")
        return x

    def fit_features(self, x):
        self.columns = list(x)
        self.dtypes = {c: str(x[c].dtype) for c in x}
        self.vocabulary = {c: set(x[c].astype(str)) for c in x.select_dtypes("category")}
        return self

    def fit(self, observations, fit_labels=None):
        return self.fit_features(self.transform(observations))

    def new_category(self, x):
        unknown = np.zeros(len(x), dtype=bool)
        for col, known in self.vocabulary.items():
            unknown |= ~x[col].astype(str).isin(known).to_numpy()
        return unknown
