from pathlib import Path
import importlib.util
import numpy as np


def test_weighted_gate_loss_equals_prediction_loss_on_eligible_rows():
    path = Path(__file__).with_name('next230_clock_gate.py')
    assert path.exists(), 'Learned clock gate is not implemented'
    spec=importlib.util.spec_from_file_location('next230_clock_gate',path)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    y=np.array([5.,7.,12.,20.])
    residual=np.array([0.,10.,10.,10.])
    direct=np.array([10.,0.,10.1,10.])
    use,target,weight,scale=m.gate_targets(y,residual,direct)
    np.testing.assert_array_equal(use,[True,True,False,False])
    gate=np.array([.2,.8])
    loss=weight*(gate-target)**2*scale
    prediction=residual[use]+gate*(direct[use]-residual[use])
    np.testing.assert_allclose(loss,(prediction-y[use])**2)
