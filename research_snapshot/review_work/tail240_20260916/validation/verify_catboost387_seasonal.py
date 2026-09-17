"""Independent two-month CatBoost387 primary and secondary assessment."""
from pathlib import Path
import numpy as np
from audit_union387_sources import ROOT,read,sha,write


def main():
    base=ROOT/'private_runs/tail240_20260916/models/catboost_union387_v2'
    protocol=read(base/'protocol.json')
    assessment_path=ROOT/'private_runs/tail240_20260916/models/catboost_union387_assessment_v1/receipt.json'
    assessment=read(assessment_path);assert assessment['protocol_sha256']==sha(base/'protocol.json')
    receipts={};bindings={}
    for fold in ('F1','F3'):
        path=ROOT/f'private_runs/tail240_20260916/validation/catboost387_result_{fold}_v2/receipt.json'
        receipt=read(path);assert receipt['status']=='passed' and receipt['protocol_sha256']==sha(base/'protocol.json')
        assert receipt['producer_manifest_sha256']==assessment['manifests'][fold]==sha(base/fold/'manifest.json')
        receipts[fold]=receipt;bindings[fold]=sha(path)
    seasonal={}
    for key,endpoint in [('matched','matched_all_finite'),('primary','primary_current387')]:
        baseline=float(np.sqrt(sum(weight*receipts[fold]['metrics'][key]['control']['mse'] for fold,weight in zip(['F1','F3'],protocol['weights']))))
        candidate=float(np.sqrt(sum(weight*receipts[fold]['metrics'][key]['candidate']['mse'] for fold,weight in zip(['F1','F3'],protocol['weights']))))
        passed=all(receipts[fold]['metrics'][key]['delta_rmse']<0 and receipts[fold]['metrics'][key]['all_day_removals_improve'] for fold in receipts)
        value=dict(reference_rmse=baseline,rmse=candidate,gain=baseline-candidate,both_months_and_days_positive=passed)
        expected=assessment['seasonal'][endpoint]
        for name,actual in value.items():
            if isinstance(actual,bool):assert actual==expected[name]
            else:np.testing.assert_allclose(actual,expected[name],rtol=0,atol=1e-10)
        seasonal[endpoint]=value
    passed=seasonal['primary_current387']['gain']>=2 and seasonal['primary_current387']['both_months_and_days_positive']
    assert passed==assessment['gate_passed']
    out=ROOT/'private_runs/tail240_20260916/validation/catboost387_seasonal_v1'
    out.mkdir(parents=True,exist_ok=False)
    result=dict(status='passed',source_sha256=sha(Path(__file__)),assessment_sha256=sha(assessment_path),
        independent_fold_receipt_hashes=bindings,seasonal=seasonal,primary_gate_passed=passed,
        no_score_or_fitting=True,limitation='Exposed June/October ordinary-tune arithmetic; no complete July/November score or promotion claim.')
    write(out/'receipt.json',result);print(result,flush=True)


if __name__=='__main__':main()
