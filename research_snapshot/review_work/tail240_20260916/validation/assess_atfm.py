"""Independent seasonal ATFM arithmetic from both native replay receipts."""
import numpy as np
import validate_candidate as v


def main():
    root=v.ROOT/'private_runs/tail240_20260916/validation'
    out=root/'atfm_seasonal_v1';out.mkdir(exist_ok=False)
    paths={'F1':root/'atfm_model_tune_f1_v3_F1/receipt.json','F3':root/'atfm_model_v2_tune_f3_v2_F3/receipt.json'}
    records={f:v.read_json(p) for f,p in paths.items()}
    assert all(r['status']=='passed' for r in records.values())
    weights={'F1':192122/344841,'F3':152719/344841}
    results={}
    for name in ['matched_all_finite','matched_ordinary','ordinary_fixed25','ordinary_union_replacement']:
        candidate=float(np.sqrt(sum(weights[f]*records[f]['reports'][name]['candidate']['mse'] for f in weights)))
        reference=float(np.sqrt(sum(weights[f]*records[f]['reports'][name]['control']['mse'] for f in weights)))
        results[name]=dict(reference_rmse=reference,candidate_rmse=candidate,improvement=reference-candidate,both_months_improve=all(records[f]['reports'][name]['gain']>0 for f in weights),both_month_day_removals_improve=all(records[f]['reports'][name]['paired']['all_day_removals_improve'] for f in weights))
    producer=v.ROOT/'private_runs/tail240_20260916/forensics/atfm/two_month_assessment_v1/receipt.json'
    expected=v.read_json(producer)
    for name,values in results.items():
        for key,value in values.items():np.testing.assert_allclose(value,expected['results'][name][key],rtol=1e-12,atol=1e-10)
    assert expected['gate_passes'] is False
    result=dict(status='passed',source_sha256=v.sha256(__file__),native_receipts={f:v.sha256(p) for f,p in paths.items()},producer_receipt_sha256=v.sha256(producer),results=results,gate_passes=False,no_score_or_refit=True)
    v.write_json(out/'receipt.json',result)
    print(results,flush=True)


if __name__=='__main__':main()
