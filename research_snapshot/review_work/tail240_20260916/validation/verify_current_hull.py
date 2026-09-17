"""Independent aggregate-only audit of label-aware scalar expert bounds."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import psutil
from threadpoolctl import threadpool_limits
import validate_candidate as v

ROOT,ID,TARGET,TIME=v.ROOT,v.ID,v.TARGET,v.common.MOVEMENT
BASE=ROOT/'private_runs/tail240_20260916/models/current_hull_diagnostic_v1'
CURRENT=ROOT/'private_runs/tail240_20260916/models/neural_missing_integration_v1/replacement'
ENSEMBLE=ROOT/'private_runs/breakthrough_20260916/models/context_gate/final_simplex9_v1'
NEURAL=ROOT/'private_runs/tail240_20260916/state/neural_context/refit_score_v3'
OUT=ROOT/'private_runs/tail240_20260916/validation/current_hull_v1'


def guard():
    info=psutil.Process().memory_info();peak=max(info.rss,info.peak_wset)
    assert peak<2*1024**3 and psutil.virtual_memory().available>=8*1024**3
    return peak


def close(a,b):np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-7)


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    protocol=v.read_json(BASE/'protocol.json');record=v.read_json(BASE/'receipt.json')
    assert record['protocol_sha256']==v.sha256(BASE/'protocol.json')
    assert protocol['source_sha256']==v.sha256(ROOT/'review_work/tail240_20260916/models/current_hull_diagnostic.py')
    assert protocol['preparation_sha256']==v.sha256(ENSEMBLE/'preparation.json')
    assert protocol['current_manifest_sha256']==v.sha256(CURRENT/'manifest.json')
    assert protocol['binding_sha256']==v.sha256(ROOT/'private_runs/tail240_20260916/validation/baseline_binding.json')
    preparation=v.read_json(ENSEMBLE/'preparation.json');currentmarker=v.read_json(CURRENT/'manifest.json')
    binding=v.read_json(ROOT/'private_runs/tail240_20260916/validation/baseline_binding.json')
    metadata=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert v.sha256(metadata)==v.read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][metadata.name]
    meta=pd.read_parquet(metadata,columns=[ID,'FLIGHT_ID_mvt',TIME,TARGET,'proxy_sec'])
    metrics={}
    for fold in ['F1','F3']:
        idx,split,_=v.common.fold_data(meta,fold,full=True)
        raw=meta.iloc[idx['score']]
        bound=binding['folds'][fold];reported=record['folds'][fold]
        assert v.object_hash(split)==v.object_hash(bound['split'])
        assert v.object_hash(raw[ID].tolist())==bound['score_id_hash']==reported['id_hash']
        assert v.object_hash(raw[TARGET].to_numpy(float).tolist())==bound['score_target_hash']==reported['label_hash']
        assert v.sha256(bound['prediction_path'])==bound['prediction_sha256']
        reference=pd.read_parquet(bound['prediction_path'],columns=[ID,TARGET,'proxy_sec',TIME])
        for name in [ID,TARGET,'proxy_sec',TIME]:np.testing.assert_array_equal(reference[name],raw[name])
        assert len(raw)==bound['score_rows']==reported['rows'] and raw[ID].is_unique
        ordinary=raw.proxy_sec.between(0,7200).to_numpy();y=raw[TARGET].to_numpy(float)
        cf=currentmarker['folds'][fold];currentpath=CURRENT/cf['prediction']['path']
        assert v.sha256(currentpath)==cf['prediction']['sha256']
        current=pd.read_parquet(currentpath)
        np.testing.assert_array_equal(current[ID],raw[ID]);values=current.prediction_sec.to_numpy(float)
        assert np.isfinite(values).all()
        protected_path=Path(cf['matched_control']['path']);assert v.sha256(protected_path)==cf['matched_control']['sha256']
        protected=pd.read_parquet(protected_path);np.testing.assert_array_equal(protected[ID],raw[ID])
        np.testing.assert_array_equal(values[~ordinary],protected.prediction_sec.to_numpy()[~ordinary])
        weights=preparation['folds'][fold]['weights'];experts=weights['experts'];vectors=[]
        assert len(experts)==9 and len(set(experts))==9
        for expert in experts:
            source=preparation['sources'][fold][expert];directory=Path(source['directory'])
            assert v.sha256(directory/'manifest.json')==source['manifest_sha256']
            assert v.sha256(directory/'candidate.parquet')==source['outputs']['candidate.parquet']
            frame=pd.read_parquet(directory/'candidate.parquet',columns=[ID,TARGET,'prediction_sec'])
            np.testing.assert_array_equal(frame[ID],raw[ID]);np.testing.assert_array_equal(frame[TARGET],y)
            vectors.append(frame.prediction_sec.to_numpy(float)[ordinary]);guard()
        assert v.sha256(NEURAL/fold/'manifest.json')==protocol['neural_manifests'][fold]
        neuralmarker=v.read_json(NEURAL/fold/'manifest.json');neuralpath=NEURAL/fold/'finite_score_predictions.parquet'
        assert v.sha256(neuralpath)==neuralmarker['outputs'][neuralpath.name]
        neural=pd.read_parquet(neuralpath).set_index(ID);assert neural.index.is_unique
        neural=neural.loc[raw.loc[ordinary,ID]];np.testing.assert_array_equal(neural[TARGET],y[ordinary])
        vectors[experts.index('tabm_ple8')]=neural.prediction_sec.to_numpy(float)
        matrix=np.column_stack(vectors);assert np.isfinite(matrix).all()
        coefficients=np.asarray(weights['global']);assert (coefficients>=0).all();close(coefficients.sum(),1.)
        np.testing.assert_allclose(matrix@coefficients,values[ordinary],rtol=0,atol=1e-7)
        current_mse=float(np.square(values-y).mean());close(current_mse,reported['current_mse'])
        fixed_sse=float(np.square(values[~ordinary]-y[~ordinary]).sum());cases={}
        for name,selected in [('all9',np.ones(9,bool)),('positive_weight',coefficients>1e-8)]:
            candidate=matrix[:,selected];truth=y[ordinary]
            left=candidate.min(axis=1);right=candidate.max(axis=1)
            distance=np.maximum(left-truth,0.)+np.maximum(truth-right,0.)
            loss=distance*distance
            nearest=np.abs(candidate-truth[:,None]).min(axis=1);single_loss=nearest*nearest
            assert (loss<=single_loss+1e-7).all()
            outside=(truth<left)|(truth>right)
            base_loss=(values[ordinary]-truth)**2
            oracle=(fixed_sse+float(loss.sum()))/len(y)
            single=(fixed_sse+float(single_loss.sum()))/len(y)
            observed=reported['cases'][name]
            assert observed['experts']==np.asarray(experts)[selected].tolist()
            assert observed['all_experts_wrong_side_rows']==int(outside.sum())
            for key,value in dict(oracle_mse=oracle,best_single_mse=single,protected_sse=fixed_sse,
                oracle_ordinary_sse=float(loss.sum()),all_experts_wrong_side_fraction=float(outside.mean()),
                current_ordinary_sse_on_wrong_side_fraction=float(base_loss[outside].sum()/base_loss.sum())).items():close(value,observed[key])
            assert oracle<=single<=current_mse
            cases[name]=dict(oracle_mse=oracle,best_single_mse=single,experts=observed['experts'])
        metrics[fold]=dict(rows=len(y),ordinary_rows=int(ordinary.sum()),current_mse=current_mse,cases=cases)
        assert metrics[fold]['ordinary_rows']==reported['ordinary_rows']
        del matrix,vectors,candidate,frame,reference,current,protected,neural,raw;gc.collect();guard()
    weights={'F1':192122/344841,'F3':152719/344841}
    current_mse=sum(weights[f]*metrics[f]['current_mse'] for f in weights)
    close(np.sqrt(current_mse),record['seasonal']['current'])
    required=current_mse-240**2;fractions={}
    for name in ['all9','positive_weight']:
        oracle=sum(weights[f]*metrics[f]['cases'][name]['oracle_mse'] for f in weights)
        single=sum(weights[f]*metrics[f]['cases'][name]['best_single_mse'] for f in weights)
        close(np.sqrt(oracle),record['seasonal'][name+'_oracle_rmse'])
        close(np.sqrt(single),record['seasonal'][name+'_best_single_rmse'])
        fractions[name]=dict(oracle_rmse=float(np.sqrt(oracle)),best_single_rmse=float(np.sqrt(single)),
            oracle_removable_weighted_mse=current_mse-oracle,required_fraction_of_oracle_removable_mse=required/(current_mse-oracle),
            best_single_removable_weighted_mse=current_mse-single,required_fraction_of_best_single_removable_mse=required/(current_mse-single))
    assert sum(vv['rows'] for vv in metrics.values())==353045
    assert record['inference_usable'] is False and record['model_trained'] is False and record['row_level_oracle_predictions_exported'] is False
    result=dict(status='passed',source_sha256=v.sha256(__file__),diagnostic_receipt_sha256=v.sha256(BASE/'receipt.json'),
        protocol_sha256=v.sha256(BASE/'protocol.json'),raw_cohorts_labels_and_every_expert_binding_exact=True,current_composition_exact=True,
        nonordinary_routes_preserved_exact=True,scalar_interval_and_best_individual_math_exact=True,weighted_mse_exact=True,
        score_rows=353045,current_rmse=float(np.sqrt(current_mse)),current_weighted_mse=current_mse,target_rmse=240.,
        required_weighted_mse_reduction=required,required_current_mse_fraction=required/current_mse,cases=fractions,folds=metrics,
        no_training=True,no_gpu=True,no_oracle_predictions_exported=True,peak_bytes=guard(),
        limitation='Outcome-aware counterfactual using fixed saved experts and exposed score labels. Required fractions are arithmetic, not proof that an inference-time gate can identify the right expert or attain240.')
    v.write_json(OUT/'receipt.json',result)
    print('VERIFIED_HULL',result['current_rmse'],fractions,'peak',result['peak_bytes'],flush=True)


if __name__=='__main__':
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    with threadpool_limits(1):main()
