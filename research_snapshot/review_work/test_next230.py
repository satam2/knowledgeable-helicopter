import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_split_comparison_accepts_json_dates_and_rejects_changed_cohorts():
    import datetime
    import next230_common as common
    assert hasattr(common, 'same_split'), 'Serialization-safe split comparison missing'
    a = {'spec': {'start': datetime.date(2025, 1, 1)}, 'stages': {'n': 100}}
    b = {'spec': {'start': '2025-01-01'}, 'stages': {'n': 100}}
    assert common.same_split(a, b)
    b['stages']['n'] = 101
    assert not common.same_split(a, b)


def test_schedule_mixture_is_a_soft_expectation_without_label_routing():
    from next230_schedule_mixture import predict_mixture
    np.testing.assert_allclose(predict_mixture(np.array([100.,100.,100.]),
        np.array([0.,.5,1.]), 2., np.array([-50.,-50.,-50.])), [50.,76.,102.])
    with pytest.raises(ValueError, match='probability'):
        predict_mixture(np.array([100.]), np.array([1.1]), 2., np.array([-50.]))


def extension():
    path = Path(__file__).with_name('next230_features.py')
    assert path.exists(), 'Next-batch feature and calibration module is not implemented'
    spec = importlib.util.spec_from_file_location('next230_features', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def observations():
    return pd.DataFrame({
        'MVT_ID_mvt': range(8),
        'PHASE_mvt': ['DEP', 'DEP', 'ARR', 'DEP', 'DEP', 'DEP', 'DEP', 'DEP'],
        'MVT_TIME_UTC_mvt': pd.to_datetime([
            '2025-01-31 23:59Z', '2025-02-01 00:01Z', '2025-02-01 00:02Z',
            '2025-02-01 00:03Z', '2025-02-01 00:03Z', '2025-02-01 00:02Z',
            '2025-02-01 00:02Z', '2025-02-01 00:04Z']),
        'ADEP_mvt': ['AAA', 'AAA', 'XXX', 'AAA', 'AAA', 'BBB', 'AAA', 'AAA'],
        'ADES_mvt': ['XXX', 'XXX', 'AAA', 'XXX', 'XXX', 'XXX', 'XXX', 'XXX'],
        'RUNWAY_mvt': ['01', '01', '01', '01', '01', '01', '02', None],
    })


def test_runway_excludes_self_ties_other_airports_and_other_runways():
    obs = observations()
    dep = obs.loc[obs.PHASE_mvt.eq('DEP')].set_index('MVT_ID_mvt', drop=False)
    out = extension().runway_features(dep, obs)
    assert out.loc[3, 'rw_dep_prior_5m'] == 1
    assert out.loc[4, 'rw_dep_prior_5m'] == 1
    assert out.loc[3, 'rw_arr_prior_5m'] == 1
    assert out.loc[3, 'rw_dep_age_sec'] == 120
    assert out.loc[3, 'rw_arr_age_sec'] == 60
    assert out.loc[3, 'rw_dep_share_5m'] == .5


def test_runway_month_boundary_and_missing_have_explicit_states():
    obs = observations()
    dep = obs.loc[obs.PHASE_mvt.eq('DEP')].set_index('MVT_ID_mvt', drop=False)
    out = extension().runway_features(dep, obs)
    assert out.loc[1, 'rw_dep_prior_60m'] == 0
    assert out.loc[1, 'rw_dep_age_sec'] == -1
    assert out.loc[1, 'rw_observed_5m_sec'] == 60
    assert out.loc[7, 'rw_missing'] == 1
    assert out.loc[7, 'rw_dep_prior_5m'] == -1
    assert out.loc[7, 'rw_dep_share_5m'] == -1


def test_runway_rejects_hidden_columns():
    obs = observations()
    obs['TAXITIME_SEC_mvt'] = 1
    with pytest.raises(ValueError, match='hidden'):
        extension().runway_features(obs, obs)


def test_clock_blend_learns_on_tune_and_shrinks_sparse_airports():
    mod = extension()
    residual = np.array([0., 0., 0., 0.])
    direct = np.array([10., 10., 10., 10.])
    y = np.array([0., 0., 10., 10.])
    airports = np.array(['A', 'A', 'B', 'B'])
    gate = mod.fit_blend(y, residual, direct, airports, shrinkage=2)
    assert gate['global'] == .5
    assert gate['airports']['A'] == .25
    assert gate['airports']['B'] == .75
    np.testing.assert_allclose(mod.blend_weights(gate, ['A', 'B', 'NEW']), [.25, .75, .5])
    assert mod.fit_blend(np.ones(4) * 50, residual, direct, airports)['global'] == 1
    assert mod.fit_blend(y, residual, residual, airports)['global'] == 0


def test_clock_gate_protects_long_missing_and_negative_routes():
    mod = extension()
    original = np.array([3., 8000., 700., 500.])
    out = mod.apply_blend(original, [100., 100., 100., 100.],
                          ['residual', 'residual_long_proxy', 'specialist_missing', 'direct_invalid'],
                          [.5, .5, .5, .5])
    np.testing.assert_array_equal(out, [51.5, 8000., 700., 500.])


def test_schedule_context_is_observed_only_and_unknown_fallback_is_explicit():
    obs = observations().iloc[[1, 3, 4]].copy()
    obs['SCHED_TIME_UTC_mvt'] = pd.to_datetime(['2025-01-31 23:00Z', '2025-02-01 00:00Z', None])
    obs['STAND_mvt'] = ['S1', 'S2', None]
    obs['FLIGHT_mvt'] = ['AB123', 'AB123', None]
    obs = obs.set_index('MVT_ID_mvt', drop=False)
    out = extension().schedule_features(obs)
    assert out.loc[1, 'schedule_day_offset'] == 1
    assert out.loc[1, 'schedule_hour'] == 23
    assert out.loc[4, 'schedule_missing'] == 1
    assert str(out.flight_designator.dtype) == 'category'
    assert out.loc[4, 'flight_designator'] == 'm:'
    support = extension().designator_support(out.iloc[:1])
    np.testing.assert_array_equal(extension().known_designator(out, support), [True, True, False])
