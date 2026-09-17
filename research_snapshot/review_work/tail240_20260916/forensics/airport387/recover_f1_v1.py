"""Reassemble immutable completed airport models without repeating training."""
import run_v2 as run
import run_v1 as old
import argparse
import shutil
import gc
import numpy as np

ORIGINAL_DECLARE=run.declare
SOURCE=old.OUT/'F1'


def declaration():
    protocol=ORIGINAL_DECLARE()
    assert run.common.sha256(old.__file__)==run.common.read_json(old.OUT/'protocol.json')['source_sha256']
    bank=run.common.read_json(SOURCE/'airport_progress.json')
    assert len(bank)==10
    for row in bank.values():
        assert run.common.sha256(SOURCE/row['model_file'])==row['model_sha256']
        assert row['native_replay_max_abs_delta']==0.
    record=dict(source_sha256=run.common.sha256(__file__),runner_sha256=run.common.sha256(run.__file__),
        original_runner_sha256=run.common.sha256(old.__file__),original_protocol_sha256=run.common.sha256(old.OUT/'protocol.json'),
        protocol_sha256=run.common.sha256(run.OUT/'protocol.json'),original_progress_sha256=run.common.sha256(SOURCE/'airport_progress.json'),
        original_encoder_sha256=run.common.sha256(SOURCE/'encoder.json'),
        original_discovery_sha256=run.common.sha256(SOURCE/'discovery.json'),
        original_models={a:row['model_sha256'] for a,row in bank.items()},
        original_training_peak_bytes=max(row['peak_bytes'] for row in bank.values()),
        recovery='Original10airportfits and individualnativeexactreplays completed. Processstopped in O(nfit*ntune)objectisin finalmembershipassertion beforepredictionserialization. Rebuildsame387matrixfromfrozeninputs,requireoriginalvocab,verifyfit/tuneairportIDhashes andmodelbytes,copyimmutablemodelstorecoveryoutput,assemble nativepredictions. No fit call. Unique-key membershipsame semantics.')
    path=run.OUT/'recovery_protocol.json'
    if path.exists():assert run.common.read_json(path)==record
    else:run.common.write_json(path,record)
    return protocol


def replay_bank(matrix,nfit,target_fit,target_tune,proxy_tune,fit_ids,tune_ids,
                fit_airports,tune_airports,fallback_prediction,columns,vocab,params,global_steps,dest):
    binding=run.common.read_json(run.OUT/'recovery_protocol.json')
    assert run.common.sha256(SOURCE/'airport_progress.json')==binding['original_progress_sha256']
    bank=run.common.read_json(SOURCE/'airport_progress.json')
    old_encoder=run.common.read_json(SOURCE/'encoder.json')
    assert run.common.sha256(SOURCE/'encoder.json')==binding['original_encoder_sha256']
    assert old_encoder==dict(columns=columns,vocab=vocab)
    assert set(bank)==set(fit_airports)
    prediction=np.asarray(fallback_prediction,float).copy()
    assigned=np.zeros(len(tune_ids),bool)
    for airport,fi,ti in run.partitions(fit_airports,tune_airports):
        row=bank[airport]
        assert row['fit_n']==len(fi) and row['tune_n']==len(ti)
        assert row['fit_id_hash']==run.common.object_hash(np.asarray(fit_ids)[fi].tolist())
        assert row['tune_id_hash']==run.common.object_hash(np.asarray(tune_ids)[ti].tolist())
        options=dict(params)
        if not len(ti):options['n_estimators']=global_steps
        assert row['params']==options and row['fixed_global_rounds_no_tune']==(not bool(len(ti)))
        assert run.common.sha256(SOURCE/row['model_file'])==row['model_sha256']==binding['original_models'][airport]
        shutil.copyfile(SOURCE/row['model_file'],dest/row['model_file'])
        native=run.lgb.Booster(model_file=str(dest/row['model_file']))
        assert native.feature_name()==columns and native.current_iteration()==row['steps']
        if len(ti):
            xt=np.asarray(matrix[nfit+ti],dtype=np.float32,order='C')
            prediction[ti]=native.predict(xt,num_threads=2)+proxy_tune[ti]
            assigned[ti]=True
            del xt
        del native
        gc.collect()
        run.guard()
        print('RECOVERED',airport,len(fi),len(ti),row['steps'],flush=True)
    expected=np.isin(tune_airports,np.unique(fit_airports))
    np.testing.assert_array_equal(assigned,expected)
    np.testing.assert_array_equal(prediction[~assigned],np.asarray(fallback_prediction)[~assigned])
    assert np.isfinite(prediction).all()
    run.common.write_json(dest/'airport_progress.json',bank)
    run.common.write_json(dest/'recovery_receipt.json',dict(status='complete',recovery_source_sha256=run.common.sha256(__file__),
        recovery_protocol_sha256=run.common.sha256(run.OUT/'recovery_protocol.json'),no_refit=True,all_airport_cohorts_exact=True,
        all_original_models_byte_identical=True,original_training_peak_bytes=binding['original_training_peak_bytes'],
        replay_peak_bytes=run.guard()))
    return prediction,bank,~assigned


run.declare=declaration
run.train_bank=replay_bank

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--declare-only',action='store_true')
    args=parser.parse_args()
    if args.declare_only:declaration()
    else:run.run('F1')
