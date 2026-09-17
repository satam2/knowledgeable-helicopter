"""Independent full XGB cohort/source/schema/score audit without native model load."""
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('combined_family_frozen',Path(__file__).with_name('audit.py'))
frozen=importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)
import numpy as np
import pandas as pd


def main():
    out=frozen.OUT/'xgb/contracts.json'
    assert not out.exists()
    path=frozen.ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert frozen.common.sha256(path)==frozen.common.read_json(frozen.ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][path.name]
    meta=pd.read_parquet(path,columns=[frozen.ID,frozen.TARGET,frozen.TIME,'FLIGHT_ID_mvt','proxy_sec'])
    results={}
    for fold in ['F1','F3']:
        folder=frozen.BASE/'combined_xgb'/f'xgb_aobt_allfinite_{fold}_s20260916'
        record=frozen.checked(folder)
        controls={'leaf63':frozen.BASE/'deeper_lgb/combined'/f'lightgbm_leaf63_aobt_allfinite_{fold}_s20260916',
            'tabm':frozen.BASE/'full_neural'/f'tabm_combined_standard_aobt_allfinite_{fold}_s20260916'}
        for p in controls.values():
            control=frozen.common.read_json(p/'manifest.json')
            for field in ['feature_columns','fit_ids','split']:
                assert record[field]==control[field]
            assert record['anchor']['feature_receipts']==control['anchor']['feature_receipts']
        assert len(record['feature_columns'])==225
        assert record['seed']==20260916 and record['threads']==2
        indices,split,_=frozen.common.fold_data(meta,fold,full=True)
        assert frozen.common.object_hash(split)==frozen.common.object_hash(record['split'])
        finite=np.isfinite(meta.proxy_sec.to_numpy(float))
        for stage,pos in indices.items():
            rows=pos[finite[pos]]
            assert record['fit_ids'][stage]==dict(n=len(rows),hash=frozen.common.object_hash(meta.iloc[rows][frozen.ID].tolist()))
        reference,_=frozen.common.reference(fold)
        frame=pd.read_parquet(folder/'candidate.parquet')
        for key in [frozen.ID,frozen.TARGET]:
            np.testing.assert_array_equal(frame[key],reference[key])
            np.testing.assert_array_equal(frame[key],meta.iloc[indices['score']][key])
        missing=~np.isfinite(reference.proxy_sec.to_numpy(float))
        np.testing.assert_array_equal(frame.prediction_sec.to_numpy()[missing],reference.prediction_sec.to_numpy()[missing])
        tune=pd.read_parquet(folder/'tune_predictions.parquet')
        positions=indices['tune'][finite[indices['tune']]]
        np.testing.assert_array_equal(tune[frozen.ID],meta.iloc[positions][frozen.ID])
        comparisons={}
        y=reference[frozen.TARGET].to_numpy(float)
        for name,path in controls.items():
            control=frozen.common.read_json(path/'manifest.json')
            assert frozen.common.sha256(path/'candidate.parquet')==control['outputs']['candidate.parquet']
            predictions=pd.read_parquet(path/'candidate.parquet')
            np.testing.assert_array_equal(predictions[frozen.ID],reference[frozen.ID])
            comparisons[name]=frozen.comparison(y,predictions.prediction_sec.to_numpy(float),frame.prediction_sec.to_numpy(float),reference.day.to_numpy())
        results[fold]=dict(status='passed',manifest_sha256=frozen.common.sha256(folder/'manifest.json'),
            same225_schema_cohorts_receipts_as_leaf63_TabM=True,full_score_IDs_rawlabels_exact=True,
            tune_IDs_exact=True,missing_V2_exact=True,source_snapshot_protocol_output_hashes_match=True,
            saved_fullscore_producer_reload_delta=record['reload_max_abs_delta'],selected_steps=record['fit']['steps'],
            metrics=frozen.metric(y,frame.prediction_sec.to_numpy(float)),comparisons=comparisons)
    frozen.common.write_json(out,dict(status='contract_checks_passed',source_sha256=frozen.common.sha256(__file__),
        helper_source_sha256=frozen.common.sha256(Path(__file__).with_name('audit.py')),folds=results,
        seasonal_rmse=frozen.season_score(results['F1']['metrics'],results['F3']['metrics']),GPU_used=False,
        peak_rss_bytes=frozen.guard(),independent_CPU_parity='Separate canary receipts; strict F1 mismatch retained. This contract audit does not waive it.'))
    print('XGB_CONTRACTS_PASSED', {f:r['metrics'] for f,r in results.items()},flush=True)


if __name__=='__main__':
    main()
