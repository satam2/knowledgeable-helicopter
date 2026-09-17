"""Three-expert convex optimum via exhaustive simplex faces, not SLSQP."""
import itertools
import numpy as np


def simplex_oracle(predictions, labels):
    p,y=np.asarray(predictions,float),np.asarray(labels,float)
    if p.ndim!=2 or p.shape[1]!=3 or y.shape!=(len(p),) or not len(p):
        raise ValueError('Expected nonempty aligned three-expert arrays')
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite mixture input')
    residual=p-y[:,None]
    gram=residual.T@residual/len(y)
    scale=max(float(np.trace(gram)),1.)
    gram/=scale
    candidates=[]
    for count in (1,2,3):
        for subset in itertools.combinations(range(3),count):
            index=list(subset)
            g=gram[np.ix_(index,index)]
            system=np.zeros((count+1,count+1))
            system[:count,:count]=g
            system[:count,-1]=1
            system[-1,:count]=1
            target=np.zeros(count+1);target[-1]=1
            solution=np.linalg.lstsq(system,target,rcond=None)[0][:count]
            if np.any(solution < -1e-10) or not np.isclose(solution.sum(),1,atol=1e-9):continue
            weights=np.zeros(3);weights[index]=np.maximum(solution,0);weights/=weights.sum()
            candidates.append((float(weights@gram@weights)*scale,weights))
    mse,weights=min(candidates,key=lambda value:value[0])
    return dict(mse=mse,weights=weights)


def validate_state(state,predictions,labels):
    oracle=simplex_oracle(predictions,labels)
    equal=np.array(state['equal'],float);simplex=np.array(state['simplex'],float);shrunk=np.array(state['shrunk'],float)
    for weights in (equal,simplex,shrunk):
        if weights.shape!=(3,) or not np.isfinite(weights).all() or np.any(weights<0) or not np.isclose(weights.sum(),1):
            raise ValueError('Invalid weights')
    np.testing.assert_allclose(equal,np.full(3,1/3),atol=1e-15,rtol=0)
    np.testing.assert_allclose(shrunk,.5*simplex+.5*equal,atol=1e-15,rtol=0)
    actual=float(np.mean((np.asarray(predictions)@simplex-np.asarray(labels))**2))
    np.testing.assert_allclose(actual,oracle['mse'],rtol=1e-7,atol=1e-7)
    assert state['calibration_rows']==len(labels)
    return dict(actual_mse=actual,oracle_mse=oracle['mse'],oracle_weights=oracle['weights'].tolist())
