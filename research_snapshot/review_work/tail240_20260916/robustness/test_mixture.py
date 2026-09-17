import numpy as np
import pytest
from mixture import fit_mixture, predict_mixture


def test_shrinkage_limits_single_expert_and_stays_in_hull():
    p = np.array([[0., 10., 20.], [1., 11., 21.], [2., 12., 22.]])
    state = fit_mixture(p, p[:, 0])
    assert np.allclose(state['simplex'], [1., 0., 0.], atol=1e-6)
    assert np.allclose(state['shrunk'], [2/3, 1/6, 1/6], atol=1e-6)
    result = predict_mixture(state, p, 'shrunk')
    assert np.all(result >= p.min(axis=1))
    assert np.all(result <= p.max(axis=1))


def test_equal_predictions_are_stable_and_bad_input_rejected():
    p = np.ones((12, 3))
    state = fit_mixture(p, np.arange(12.))
    assert np.allclose(predict_mixture(state, p, 'shrunk'), 1.)
    with pytest.raises(ValueError):
        fit_mixture(p, np.full(12, np.nan))
    with pytest.raises(ValueError):
        predict_mixture(state, np.ones((4, 2)), 'shrunk')


def test_evaluation_values_cannot_change_fitted_state():
    rng = np.random.default_rng(9)
    calibration = rng.normal(size=(100, 3))
    state = fit_mixture(calibration, calibration @ np.array([.2, .3, .5]))
    before = repr(state)
    predict_mixture(state, rng.normal(size=(15, 3)), 'shrunk')
    predict_mixture(state, rng.normal(size=(15, 3)) * 1e9, 'shrunk')
    assert repr(state) == before
