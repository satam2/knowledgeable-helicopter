"""Preserve failed v1 score attempt; normalize nonsemantic DataFrame row indices."""
import run_v1 as run
import pandas as pd


def main():
    run.declare()
    release=run.common.read_json(run.OUT/'independent_evaluation_release.json')
    assert release['status']=='released'
    assert release['protocol_sha256']==run.common.sha256(run.OUT/'protocol.json')
    manifests={}
    for fold in run.PANELS:
        for arm in run.ARMS:
            check=release['independent_receipts'][fold+'/'+arm]
            assert run.common.sha256(check['path'])==check['sha256']
            reviewed=run.common.read_json(check['path'])
            folder=run.OUT/fold/arm
            marker=run.common.read_json(folder/'manifest.json')
            assert marker['status']=='complete_predictions_frozen'
            assert marker['protocol_sha256']==release['protocol_sha256']
            assert run.common.sha256(folder/'manifest.json')==reviewed['expert_manifest_sha256']
            for stage,record in marker['predictions'].items():
                assert run.common.sha256(folder/(stage+'.parquet'))==record['sha256']
            manifests[fold+'/'+arm]=run.common.sha256(folder/'manifest.json')
    dest=run.common.external_path(run.OUT/'evaluation_v2')
    dest.mkdir(exist_ok=False)
    run.common.write_json(dest/'evaluation_opened.json',dict(utc=run.common.utc_now(),manifests=manifests,
        protocol_sha256=release['protocol_sha256'],release_sha256=run.common.sha256(run.OUT/'independent_evaluation_release.json'),
        source_sha256=run.common.sha256(__file__),frozen_metric_source_sha256=run.common.sha256(run.__file__),
        correction='Reset metadata row index before frame equality; original ordered IDs and every value still checked. v1 stopped before first label read.',
        previous_attempt_sha256=run.common.sha256(run.OUT/'evaluation/evaluation_opened.json')))
    reports={}
    for fold,months in run.PANELS.items():
        reports[fold]={}
        for month in months:
            stage='evaluation_'+month
            reference=None
            predictions={}
            for arm in run.ARMS:
                frame=pd.read_parquet(run.OUT/fold/arm/(stage+'.parquet'))
                assert run.TARGET not in frame
                if reference is None:
                    reference=frame.drop(columns='prediction_sec')
                else:
                    pd.testing.assert_frame_equal(reference,frame.drop(columns='prediction_sec'))
                predictions[arm]=frame.prediction_sec.to_numpy(float)
            expected=run.read_stage(fold,stage).reset_index(drop=True)
            pd.testing.assert_frame_equal(reference,expected)
            y=run.maintrain.read_labels(reference)
            days=reference[run.TIME].dt.strftime('%Y-%m-%d').to_numpy()
            record=dict(rows=len(y),id_hash=run.common.object_hash(reference[run.ID].tolist()),
                label_hash=run.common.object_hash(y.tolist()),
                metrics={arm:run.metric(y,p) for arm,p in predictions.items()},
                et20_vs_et1=run.pair(y,predictions['et_leaf20'],predictions['et_leaf1'],days),airports={})
            for airport in sorted(reference.ADEP_mvt.astype(str).unique()):
                keep=reference.ADEP_mvt.astype(str).eq(airport).to_numpy()
                record['airports'][airport]={arm:run.metric(y[keep],p[keep]) for arm,p in predictions.items()}
            replay=reference.copy()
            replay[run.TARGET]=y
            for arm,p in predictions.items():
                replay[arm]=p
            path=dest/(fold+'_'+month+'.parquet')
            replay.to_parquet(path,index=False)
            record['replay_sha256']=run.common.sha256(path)
            reports[fold][month]=record
    run.common.write_json(dest/'summary.json',dict(status='complete',protocol_sha256=release['protocol_sha256'],
        source_sha256=run.common.sha256(__file__),metric_source_sha256=run.common.sha256(run.__file__),
        independent_release_sha256=run.common.sha256(run.OUT/'independent_evaluation_release.json'),
        model_manifest_hashes=manifests,folds=reports,
        primary='ETleaf20vsETleaf1 on all missing raw outcomes; normalized diagnostic only',
        no_blend_or_complete_pipeline_claim=True,calibration_labels_used=False,
        limitation='Allmonths exposed,December shared acrossorigins; no unseen2026claim or automatic promotion'))
    print('MISSING_EVALUATED', {f:{m:r['et20_vs_et1']['rmse_gain'] for m,r in v.items()} for f,v in reports.items()},flush=True)


if __name__=='__main__':
    main()
