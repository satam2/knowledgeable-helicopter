"""One100k synthetic linear-tree runtime canary; no private flight rows."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import lightgbm as lgb
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'private_runs/tail240_20260916/forensics/linear387_canary/v1'
DECLARED=ROOT/'private_runs/tail240_20260916/models/linear_finite_tune_v1/protocol.json'
SEED=20260916
ROWS=100000
NUMERIC=372
CATEGORICAL=15
CAP=3*1024**3


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def check_memory(current,peak,available):
    assert max(current,peak)<CAP, f'3GiB process ceiling: current={current},peak={peak}'
    assert available>=8*1024**3,'Host reserve below8GiB'


def guard(_env=None):
    info=psutil.Process().memory_info()
    peak=getattr(info,'peak_wset',info.rss)
    check_memory(info.rss,peak,psutil.virtual_memory().available)
    return dict(rss=info.rss,peak=peak,available=psutil.virtual_memory().available)


def inspect_dump(dump,categories):
    leaves=0
    coefficients=0
    feature_indices=set()
    category_splits=0
    def visit(node):
        nonlocal leaves,coefficients,category_splits
        if 'split_index' in node:
            category_splits+=int(node['split_feature'] in categories)
            visit(node['left_child']);visit(node['right_child'])
            return
        leaves+=1
        for key in ['leaf_value','leaf_const']:
            if key in node:assert np.isfinite(float(node[key]))
        fields=node.get('leaf_features',[])
        values=node.get('leaf_coeff',[])
        assert len(fields)==len(values)
        assert not set(fields).intersection(categories), 'Categorical index in linear leaf features'
        assert all(np.isfinite(float(v)) for v in values), 'Nonfinite leaf coefficient'
        feature_indices.update(fields)
        coefficients+=len(values)
    for tree in dump['tree_info']:visit(tree['tree_structure'])
    return dict(trees=len(dump['tree_info']),leaves=leaves,linear_coefficients=coefficients,
        numeric_linear_feature_indices=sorted(feature_indices),categorical_split_nodes=category_splits,
        category_indices_absent_from_linear_leaf_features=True,all_leaf_constants_and_coefficients_finite=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--declare-only',action='store_true')
    args=parser.parse_args()
    declared=json.loads(DECLARED.read_text())
    params=dict(declared['params']['linear'])
    assert params['linear_tree'] is True and params['linear_lambda']==5 and params['num_leaves']==63
    assert params['n_estimators']==2500 and params['n_jobs']==2 and params['device_type']=='cpu'
    params['n_estimators']=30
    protocol=dict(source_sha256=sha(__file__),declared_protocol_sha256=sha(DECLARED),params=params,
        rows=ROWS,numeric=NUMERIC,categorical=CATEGORICAL,seed=SEED,
        synthetic='Fixedseedstandardnormal372numeric; first8latentcolumns generate relatedobservedclocklike columns8..23 plusnear/exactcollinearity; independent2pctnumericNaNs, knownsyntheticcategorycodes0/2..65. Numericfitmean/populationstdscalingwithNaNpreserved. Targetfromlatentnumerics+categoricaleffect+noise beforemissing. All100kfit, replayall100k; noaccuracyclaim.',
        scope='Onecaseonly. No privateflightrowfiles opened, no maincode mutation. Protocolparameter metadata only. NativeJSONleaf_features/leaf_coeff inspections, savednativepredictionexact.',
        resources='OneprocessCPU2, currentandOShistoricalpeak<3GiB,hostreserve8GiB,start11GiB. Guardbeforelaunchandboostcallbacks; sampledguard notOShardlimit.',
        projection='Measuredarraybytes/row exact; nativefitRSSincrement linear-per-row projection to816060F1fit only heuristic, plusfull996700rowmatrix. No guarantee2500roundmemory/runtime scaleslinearly orraredata degeneracyabsent.')
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:write(path,protocol)
    if args.declare_only:return
    assert not (OUT/'receipt.json').exists() and not (OUT/'model.txt').exists()
    assert psutil.virtual_memory().available>=11*1024**3
    initial=guard()
    rng=np.random.default_rng(SEED)
    began=time.perf_counter()
    x=np.empty((ROWS,NUMERIC+CATEGORICAL),dtype=np.float32,order='C')
    x[:,:NUMERIC]=rng.standard_normal((ROWS,NUMERIC),dtype=np.float32)
    latent=x[:,:8].copy()
    for j in range(8,24):
        x[:,j]=(j%5+1)*latent[:,j%8]+.02*rng.standard_normal(ROWS,dtype=np.float32)
    x[:,24]=x[:,8]-x[:,9]
    x[:,25]=2*x[:,24]
    x[:,26]=1.
    x[:,27]=np.nan
    for j in range(NUMERIC,NUMERIC+CATEGORICAL):
        x[:,j]=rng.integers(2,66,size=ROWS).astype(np.float32)
        x[rng.random(ROWS)<.01,j]=0.
    y=(50*latent[:,0]-30*latent[:,1]+20*latent[:,2]+5*(x[:,NUMERIC]%5)+rng.normal(0,5,ROWS)).astype(np.float64)
    nan_count=0
    scaler=[]
    for j in range(NUMERIC):
        column=x[:,j].astype(np.float64)
        column[rng.random(ROWS)<.02]=np.nan
        good=np.isfinite(column)
        mean=float(column[good].mean()) if good.any() else 0.
        sd=float(column[good].std()) if good.any() else 1.
        sd=sd if sd>1e-12 else 1.
        x[:,j]=((column-mean)/sd).astype(np.float32)
        nan_count+=int((~good).sum())
        scaler.append(dict(mean=mean,scale=sd,finite=int(good.sum())))
    del latent,column,good
    before_fit=guard()
    data_seconds=time.perf_counter()-began
    features=[f'synthetic_numeric_{j:03d}' for j in range(NUMERIC)]+[f'synthetic_category_{j:02d}' for j in range(CATEGORICAL)]
    cats=list(range(NUMERIC,NUMERIC+CATEGORICAL))
    fit_start=time.perf_counter()
    model=lgb.LGBMRegressor(**params)
    model.fit(x,y,categorical_feature=cats,feature_name=features,callbacks=[guard,lgb.log_evaluation(10)])
    fit_seconds=time.perf_counter()-fit_start
    after_fit=guard()
    prediction=model.predict(x,num_iteration=30,num_threads=2)
    assert np.isfinite(prediction).all()
    model.booster_.save_model(str(OUT/'model.txt'),num_iteration=30)
    native=lgb.Booster(model_file=str(OUT/'model.txt'))
    replay=native.predict(x,num_iteration=30,num_threads=2)
    np.testing.assert_array_equal(prediction,replay)
    dump=native.dump_model(num_iteration=30)
    inspection=inspect_dump(dump,set(cats))
    assert inspection['trees']==30 and inspection['linear_coefficients']>0 and inspection['categorical_split_nodes']>0
    write(OUT/'native_dump.json',dump)
    write(OUT/'scaler.json',scaler)
    np.save(OUT/'synthetic_predictions.npy',prediction)
    final=guard()
    array_bytes=x.nbytes+y.nbytes
    marginal=max(0,after_fit['peak']-before_fit['rss'])/ROWS
    f1_matrix=996700*387*4
    conservative=initial['rss']+f1_matrix+marginal*816060
    record=dict(status='passed',source_sha256=sha(__file__),protocol_sha256=sha(path),params=params,
        inspection=inspection,native_replay_max_abs_delta=0.,all_predictions_finite=True,
        rows=ROWS,numeric_nan_cells=nan_count,seed=SEED,data_build_seconds=data_seconds,fit_seconds=fit_seconds,
        total_seconds=time.perf_counter()-began,memory=dict(initial=initial,before_fit=before_fit,after_fit=after_fit,final=final),
        projection=dict(synthetic_array_bytes=array_bytes,array_bytes_per_row=array_bytes/ROWS,
            approximate_native_fit_increment_bytes_per_fit_row=marginal,fullF1_matrix_bytes=f1_matrix,
            fullF1_projected_bytes_with_observed_native_increment=conservative,
            fullF1_projected_GiB=conservative/1024**3,F1_cap_GiB=12,
            linear_30round_fit_seconds_projection=fit_seconds*816060/ROWS,
            round2500_times_rowcount_naive_seconds=fit_seconds*816060/ROWS*2500/30,
            limitation='Heuristicresourceestimateonly; modelroundgrowth, allocatorretention,realvocab/collinearity andtrainingdata can differ. Measured30roundsuccess doesnotcertify2500roundfullrun.'),
        no_private_flight_rows=True,no_model_grid=True,
        outputs={p.name:sha(p) for p in OUT.iterdir() if p.name in ['model.txt','native_dump.json','scaler.json','synthetic_predictions.npy']})
    write(OUT/'receipt.json',record)
    print('CANARY',record,flush=True)


if __name__=='__main__':main()
