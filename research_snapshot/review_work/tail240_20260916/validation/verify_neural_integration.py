"""Independent observable-route reconstruction after both native refit checks."""
from pathlib import Path
import numpy as np
import pandas as pd
import validate_candidate as v

ROOT=v.ROOT
BASE=ROOT/'private_runs/tail240_20260916/models/neural_missing_integration_v1'
NEURAL=ROOT/'private_runs/tail240_20260916/state/neural_context/refit_score_v3'
MISSING=ROOT/'private_runs/tail240_20260916/models/normalized_missing_refit_v2/observed_schedule_scale_blend25'
OUT=ROOT/'private_runs/tail240_20260916/validation/neural_missing_integration_v1'


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    protocol=v.read_json(BASE/'protocol.json');binding=v.read_json(v.BINDING)
    assert v.sha256(v.BINDING)==protocol['baseline_binding_sha256']
    assert v.sha256(NEURAL/'protocol.json')==protocol['neural_protocol_sha256']
    assert v.sha256(ROOT/'review_work/tail240_20260916/models/integrate_neural_missing.py')==protocol['source_sha256']
    assert protocol['variants']==['replacement','blend25']
    meta_path=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert v.sha256(meta_path)==binding['metadata_sha256']
    metadata=pd.read_parquet(meta_path,columns=[v.ID,'proxy_sec']).set_index(v.ID)
    assert metadata.index.is_unique
    reports={}
    for fold,bound in binding['folds'].items():
        native_path=ROOT/f'private_runs/tail240_20260916/validation/neural_refit_v3_{fold}/receipt.json'
        native=v.read_json(native_path);assert native['status']=='passed'
        assert native['old225_native_max_abs_delta_sec']==native['new_native_max_abs_delta_sec']==0.
        assert native['producer_manifest_sha256']==v.sha256(NEURAL/fold/'manifest.json')
        marker=v.read_json(NEURAL/fold/'manifest.json')
        assert v.sha256(bound['prediction_path'])==bound['prediction_sha256']
        baseline=pd.read_parquet(bound['prediction_path'],columns=[v.ID,'prediction_sec'])
        ids=baseline[v.ID];assert v.object_hash(ids.tolist())==bound['score_id_hash']
        proxy=metadata.loc[ids,'proxy_sec'].to_numpy(float)
        finite=np.isfinite(proxy);ordinary=finite&(proxy>=0)&(proxy<=7200);tail=finite&~ordinary
        missing_path=MISSING/(fold+'.parquet');assert v.sha256(missing_path)==protocol['missing_predictions'][fold]
        missing=pd.read_parquet(missing_path);np.testing.assert_array_equal(missing[v.ID],ids)
        np.testing.assert_array_equal(missing.prediction_sec.to_numpy()[finite],baseline.prediction_sec.to_numpy()[finite])
        checks={}
        for variant in protocol['variants']:
            neural_path=NEURAL/fold/(variant+'.parquet');assert v.sha256(neural_path)==marker['outputs'][neural_path.name]
            neural=pd.read_parquet(neural_path,columns=[v.ID,'prediction_sec']);np.testing.assert_array_equal(neural[v.ID],ids)
            np.testing.assert_array_equal(neural.prediction_sec.to_numpy()[~ordinary],baseline.prediction_sec.to_numpy()[~ordinary])
            expected=baseline.prediction_sec.to_numpy(copy=True)
            expected[ordinary]=neural.prediction_sec.to_numpy()[ordinary]
            expected[~finite]=missing.prediction_sec.to_numpy()[~finite]
            exchange=v.read_json(BASE/variant/'manifest.json');assert exchange['status']=='complete'
            assert exchange['neural_component_manifests'][fold]==v.sha256(NEURAL/fold/'manifest.json')
            path=v.file_receipt(BASE/variant,exchange['folds'][fold]['prediction']);actual=pd.read_parquet(path)
            np.testing.assert_array_equal(actual[v.ID],ids);np.testing.assert_array_equal(actual.prediction_sec,expected)
            np.testing.assert_array_equal(actual.prediction_sec.to_numpy()[tail],baseline.prediction_sec.to_numpy()[tail])
            evaluated=ROOT/f'private_runs/tail240_20260916/validation/neural_missing_integration_{variant}_v1/evaluation.json'
            evaluation=v.read_json(evaluated);assert evaluation['status']=='passed' and evaluation['manifest_sha256']==v.sha256(BASE/variant/'manifest.json')
            checks[variant]={'rows':len(ids),'ordinary':int(ordinary.sum()),'missing':int((~finite).sum()),'finite_nonordinary_protected':int(tail.sum()),
                'component_vectors_exact':True,'prediction_sha256':v.sha256(path),'evaluation_sha256':v.sha256(evaluated)}
        reports[fold]={'native_receipt_sha256':v.sha256(native_path),'variants':checks}
    v.write_json(OUT/'receipt.json',{'status':'passed','source_sha256':v.sha256(__file__),
        'integration_protocol_sha256':v.sha256(BASE/'protocol.json'),'folds':reports,
        'scope':'Both fixed compositions exactly rebuilt using ID/proxy-only private metadata. Both full original score cohort evaluations and independent saved-model refit replays bound. No additional fitting, routing masks, thresholds or variant selection.',
        'private_input_columns':[v.ID,'proxy_sec'],'peak_rss_bytes':v.guard()})
    print('BOTH_COMPOSITIONS_EXACT',reports,flush=True)


if __name__=='__main__':main()
