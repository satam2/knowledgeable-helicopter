import numpy as np
from compose import release_weights,route


def test_routes_preserve_boundaries_and_missing_semantics():
    proxy=np.array([0,7200,-1,7201,np.nan,np.inf,-np.inf])
    p,regular,unusual,missing=route(proxy,np.arange(7)+10.,np.arange(7)+20.,np.arange(7)+30.)
    np.testing.assert_array_equal(p,[10,11,22,23,34,35,36])
    assert (regular.sum(),unusual.sum(),missing.sum())==(2,2,3)


def test_season_weights_and_explicit_numerical_zeros():
    old,new=release_weights([1e-17,.8,.2],[0,.1,.9])
    assert new[0]==0 and new.sum()==1.
    np.testing.assert_allclose(new[1],(192122*.8+152719*.1)/344841,atol=1e-15)


def test_missing_selected_route_cannot_pass():
    import pytest
    with pytest.raises(ValueError):route([np.nan],[1.],[2.],[np.nan])
