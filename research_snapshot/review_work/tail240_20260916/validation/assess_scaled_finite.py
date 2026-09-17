"""Independent seasonal primary and fixed component-replacement assessment."""
import numpy as np
import pandas as pd
import validate_candidate as v

ROOT=v.ROOT
OUT=ROOT/'private_runs/tail240_20260916/validation/scaled_finite_seasonal_v1'


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    reports={};weights={'F1':192122/344841,'F3':152719/344841}
    base=ROOT/'private_runs/breakthrough_20260916/models/context_gate/final_simplex9_v1'
    for fold in weights:
        native=ROOT/'private_runs/tail240_20260916/validation'/('scaled_finite_'+fold)/'receipt.json'
        receipt=v.read_json(native);assert receipt['status']=='passed'
        folder=ROOT/'private_runs/tail240_20260916/models/scaled_finite_tune_v1'/fold
        manifest=v.read_json(folder/'manifest.json')
        assert v.sha256(folder/'manifest.json')==receipt['manifest_sha256']
        assert v.sha256(folder/'tune.parquet')==manifest['outputs']['tune.parquet']
        prediction=pd.read_parquet(folder/'tune.parquet').set_index(v.ID)
        alignedpath=base/f'{fold}_aligned_tune.parquet'
        preparation=v.read_json(base/'preparation.json')['folds'][fold]
        assert v.sha256(alignedpath)==preparation['aligned_tune_sha256']
        aligned=pd.read_parquet(alignedpath)
        w=v.read_json(base/f'{fold}_weights.json');assert w==preparation['weights']
        reference=aligned[w['experts']].to_numpy()@np.asarray(w['global'])
        coef=w['global'][w['experts'].index('lgb63_union')]
        candidate=reference+coef*(prediction.loc[aligned[v.ID]].prediction_sec.to_numpy()-aligned.lgb63_union.to_numpy())
        paired=v.paired(aligned[v.TARGET].to_numpy(float),reference,candidate,aligned[v.common.MOVEMENT].dt.floor('D').to_numpy())
        reports[fold]=dict(native_receipt_sha256=v.sha256(native),matched=receipt['matched'],fixed25=receipt['ordinary_fixed25'],replacement_diagnostic=paired)
    seasonal={}
    for name in ['fixed25','replacement_diagnostic']:
        candidate=float(np.sqrt(sum(weights[f]*reports[f][name]['candidate']['mse'] for f in weights)))
        reference=float(np.sqrt(sum(weights[f]*reports[f][name]['control']['mse'] for f in weights)))
        seasonal[name]=dict(rmse=candidate,reference_rmse=reference,gain=reference-candidate)
    gate=seasonal['fixed25']['gain']>=2 and all(reports[f][name]['delta_rmse']<0 and reports[f][name]['all_day_removals_improve'] for f in weights for name in ['matched','fixed25'])
    producer=ROOT/'private_runs/tail240_20260916/models/scaled_finite_assessment_v1/summary.json'
    expected=v.read_json(producer)
    assert gate==expected['declared_gate_passed']
    for name,values in seasonal.items():
        for key,value in values.items():np.testing.assert_allclose(value,expected['seasonal'][name][key],rtol=1e-12,atol=1e-10)
    receipt=dict(status='passed',source_sha256=v.sha256(__file__),producer_summary_sha256=v.sha256(producer),seasonal=seasonal,gate_passed=gate,folds=reports,no_score_or_refit=True)
    v.write_json(OUT/'receipt.json',receipt)
    print(seasonal,'GATE',gate,flush=True)


if __name__=='__main__':main()
