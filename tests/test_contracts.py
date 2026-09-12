import copy

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from taxiout.availability import assert_observations, make_observations
from taxiout.features.calendar import calendar_features
from taxiout.features.pipeline import FeaturePipeline, token
from taxiout.features.traffic import prior_counts, traffic_features
from taxiout.metrics import evaluate, pooled, scores
from taxiout.schema import BLOCK, FLIGHT_ID, ID, MOVEMENT, PHASE, TARGET, align, duration, target_identity
from taxiout.splits import interval_mask, make_fold
from taxiout.submission import build_submission, serialized_values, validate_submission


def test_phase_identity_and_sanitizing(movements):
    movements.loc[0, PHASE] = "ARR"
    movements.loc[0, BLOCK] = movements.loc[0, MOVEMENT] + pd.Timedelta(seconds=900)
    target_identity(movements)
    obs, labels, _ = make_observations(movements)
    assert BLOCK not in obs and TARGET not in obs and len(labels) == len(obs) - 1
    assert_observations(obs)
    movements.loc[0, BLOCK] = pd.NaT
    movements.loc[0, TARGET] = np.nan
    assert target_identity(movements)["unlabelled_arrival_context"] == 1
    movements.loc[1, TARGET] = np.nan
    with pytest.raises(ValueError, match="identity"):
        target_identity(movements)


def test_frozen_feature_hidden_neighbor_mutations(movements, config):
    obs, _, _ = make_observations(movements)
    pipeline = FeaturePipeline(config).fit(obs)
    before = pipeline.transform(obs.iloc[10:], context=obs)
    movements[TARGET] = np.random.default_rng(42).normal(size=len(movements)) * 1e9
    movements[BLOCK] = pd.Timestamp("1900-01-01", tz="UTC")
    mutated, _, _ = make_observations(movements)
    pd.testing.assert_frame_equal(before, pipeline.transform(mutated.iloc[10:], context=mutated))
    with pytest.raises(ValueError, match="hidden"):
        pipeline.transform(movements)


def test_duration_midnight_multiday_units_and_missing():
    start = pd.Series(pd.to_datetime(["2025-01-01T23:50Z", "2025-01-01T00:00Z", None], utc=True))
    end = pd.Series(pd.to_datetime(["2025-01-02T00:10Z", "2025-01-02T12:00Z", "2025-01-01T01:00Z"], utc=True))
    result = duration(end, start)
    assert result.iloc[:2].tolist() == [1200, 129600]
    assert pd.isna(result.iloc[2])
    pd.testing.assert_series_equal(result, duration(end.dt.as_unit("us"), start.dt.as_unit("us")))


def test_window_boundaries_ties_and_units():
    events = pd.Series(pd.to_datetime(["2025-01-01T09:00Z", "2025-01-01T09:45Z", "2025-01-01T09:59Z", "2025-01-01T10:00Z", "2025-01-01T10:00Z"], utc=True))
    query = events.iloc[-1:]
    assert prior_counts(events, query, 15).tolist() == [2]
    assert prior_counts(events, query, 60).tolist() == [3]
    assert prior_counts(events.dt.as_unit("us"), query.dt.as_unit("ns"), 15).tolist() == [2]
    for minutes in [15, 60]:
        brute = [sum((events >= t - pd.Timedelta(minutes=minutes)) & (events < t)) for t in events]
        assert prior_counts(events, events, minutes).tolist() == brute


def test_traffic_airports_phases_month_edges(movements):
    movements = movements.iloc[:4].copy()
    movements[MOVEMENT] = pd.to_datetime(["2025-01-31T23:59Z", "2025-02-01T00:01Z", "2025-02-01T00:02Z", "2025-02-01T00:03Z"], utc=True)
    movements.loc[2, PHASE] = "ARR"
    movements.loc[2, "ADES_mvt"] = "EDDF"
    obs, _, _ = make_observations(movements)
    dep = obs.loc[obs[PHASE].eq("DEP")]
    isolated = traffic_features(dep, obs)
    assert isolated.dep_prior_15m.tolist() == [0, 0, 1]
    assert isolated.arr_prior_15m.tolist() == [0, 0, 1]
    assert isolated.observed_window_15m_sec.tolist() == [900, 60, 180]
    continuous = traffic_features(dep, obs, False)
    assert continuous.dep_prior_15m.tolist() == [0, 1, 2]


def test_dst_and_collision_safe_tokens():
    ts = pd.Series(pd.to_datetime(["2025-03-30T00:30Z", "2025-03-30T01:30Z"], utc=True))
    result = calendar_features(ts, pd.Series(["EDDF", "EDDF"]), True)
    assert result.local_hour.tolist() == [1, 3]
    assert result.utc_offset_hours.tolist() == [1, 2]
    assert result.dst.tolist() == [0, 1]
    a = token(pd.Series(["a|b", "a", None, "m:"]))
    b = token(pd.Series(["c", "b|c", "", ""]))
    assert (a + b).is_unique


def test_split_boundaries_and_flight_purge():
    meta = pd.DataFrame({ID: [1., 2., 3., 4.], FLIGHT_ID: [100., 200., 100., 400.], MOVEMENT: pd.to_datetime(["2025-01-10", "2025-02-10", "2025-03-10", "2025-01-20"], utc=True)})
    spec = {"fit": ["2025-01-01", "2025-02-01"], "tune": ["2025-02-01", "2025-03-01"], "refit": ["2025-01-01", "2025-03-01"], "score": ["2025-03-01", "2025-04-01"]}
    idx, evidence = make_fold(meta, spec)
    assert idx["fit"].tolist() == [3]
    assert evidence["purged_related_departures"]["refit_against_score"] == 1
    boundary = pd.Series(pd.to_datetime(["2025-02-01", "2025-03-01"], utc=True))
    assert interval_mask(boundary, spec["tune"]).tolist() == [True, False]
    bad = copy.deepcopy(spec)
    bad["score"] = bad["fit"]
    with pytest.raises(ValueError, match="overlap"):
        make_fold(meta, bad)


def test_metrics_alignment_and_sse_slices():
    labels = pd.DataFrame({ID: [1., 2., 3.], TARGET: [0, 2, 5]})
    predictions = pd.DataFrame({ID: [3., 1., 2.], "prediction_sec": [5., 3., 6.], "proxy_status": ["present", "missing", "present"]})
    metric, rows = evaluate(predictions, labels)
    assert metric["overall"]["sse"] == 25
    assert metric["overall"]["rmse_sec"] == pytest.approx(np.sqrt(25 / 3))
    assert metric["slices"]["proxy_status"]["missing"]["sse_share"] == 9 / 25
    assert pooled([scores([0], [3]), scores([2, 5], [6, 5])])["rmse_sec"] == metric["overall"]["rmse_sec"]
    with pytest.raises(ValueError, match="cohorts"):
        evaluate(predictions.iloc[:2], labels)
    with pytest.raises(ValueError, match="unique"):
        align(labels, pd.concat([predictions, predictions.iloc[:1]]))


def test_submission_shuffled_exact_roundtrip_and_failures(tmp_path):
    template = tmp_path / "template.parquet"
    pq.write_table(pa.table({ID: pa.array([2., 1.], type=pa.float64()), TARGET: pa.array([None, None], type=pa.int32())}), template)
    predictions = pd.DataFrame({ID: [1., 2.], "prediction_sec": [2.5, -1.6]})
    file = tmp_path / "prediction.parquet"
    result = build_submission(template, predictions, file)
    assert result["passed"] and pd.read_parquet(file)[TARGET].tolist() == [-2, 2]
    for invalid in [[np.inf], [np.nan], [2 ** 31], [-2 ** 31 - 1]]:
        with pytest.raises(ValueError):
            serialized_values(invalid)
    for invalid in [predictions.iloc[:1], pd.concat([predictions, predictions.iloc[:1]]), predictions.assign(MVT_ID_mvt=[1., 99.])]:
        with pytest.raises(ValueError):
            build_submission(template, invalid, file)
    altered = pd.read_parquet(file).iloc[::-1]
    altered.to_parquet(tmp_path / "wrong.parquet", index=False)
    with pytest.raises(ValueError):
        validate_submission(tmp_path / "wrong.parquet", template)
