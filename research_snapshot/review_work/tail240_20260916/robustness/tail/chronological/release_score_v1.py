"""Bind all six independent model checks before the declared evaluation release."""
import run_v1 as run


def main():
    run.declare()
    path=run.OUT/'independent_evaluation_release.json'
    assert not path.exists()
    receipts={}
    for fold in run.PANELS:
        for arm in run.ARMS:
            source=run.ROOT/f'private_runs/tail240_20260916/robustness/validation/missing/{fold}_{arm}.json'
            receipt=run.common.read_json(source)
            assert receipt['status']=='passed'
            assert receipt['protocol_sha256']==run.common.sha256(run.OUT/'protocol.json')
            assert receipt['expert_manifest_sha256']==run.common.sha256(run.OUT/fold/arm/'manifest.json')
            assert receipt['independent_allrow_native_replay'] and not receipt['calibration_or_evaluation_labels_read']
            receipts[fold+'/'+arm]=dict(path=str(source),sha256=run.common.sha256(source))
    run.common.write_json(path,dict(status='released',utc=run.common.utc_now(),source_sha256=run.common.sha256(__file__),
        protocol_sha256=run.common.sha256(run.OUT/'protocol.json'),independent_receipts=receipts,
        authorization='Parent explicitly authorized declared missing scoring after all six independently passed; no model or promotion changes'))
    run.score()


if __name__=='__main__':
    main()
