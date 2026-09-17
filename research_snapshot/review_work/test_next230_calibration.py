import importlib.util
from pathlib import Path

import numpy as np


def test_probability_calibration_recovers_an_observable_shift():
    path = Path(__file__).with_name('next230_calibrate_mixture.py')
    assert path.exists(), 'Tuning-only probability calibration is not implemented'
    spec = importlib.util.spec_from_file_location('next230_calibrate_mixture', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    p = np.linspace(.1, .7, 500)
    offset = np.full(500, 4000.)
    bad = np.full(500, -3000.)
    y = offset + (1-(p+.15))*bad
    params = mod.fit_probability_calibration(p, offset, 0., bad, y)
    calibrated = mod.calibrate_probability(p, params)
    # Ridge deliberately leaves a small bias; require >99% error reduction.
    assert np.mean((calibrated-(p+.15))**2) < .01 * np.mean((p-(p+.15))**2)
    assert np.all((calibrated >= 0) & (calibrated <= 1))
    assert mod.fit_probability_calibration(p[:5], offset[:5], 0., bad[:5], y[:5])['insufficient_support']
