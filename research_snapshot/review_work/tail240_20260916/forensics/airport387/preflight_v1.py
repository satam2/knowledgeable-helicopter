"""Read-only source discovery/control binding; no matrix or label reads."""
import run_v1 as run


def main():
    protocol=run.declare()
    path=run.OUT/'preflight.json'
    assert not path.exists()
    tuples=run.sources(protocol['columns'])
    checks={}
    for fold in ('F1','F3'):
        marker=run.common.read_json(run.CONTROL[fold]/'manifest.json')
        assert run.common.sha256(run.CONTROL[fold]/'manifest.json')==protocol['fresh_controls'][fold]
        expected=[{k:r[k] for k in ('path','sha256','columns')} for r in marker['feature_receipts']]
        actual=[dict(path=str(p),sha256=h,columns=columns) for p,h,fill,columns in tuples]
        assert actual==expected and marker['params']==protocol['params'] and marker['original_control_max_abs_delta']<=1e-7
        encoder=run.common.read_json(run.CONTROL[fold]/'encoder.json')
        assert encoder['columns']==protocol['columns']
        for name,h in marker['outputs'].items():assert run.common.sha256(run.CONTROL[fold]/name)==h
        checks[fold]=dict(control_manifest_sha256=protocol['fresh_controls'][fold],
            feature_tuple_equality=True,original_parity=marker['original_control_max_abs_delta'],steps=marker['steps'])
    base=run.ROOT/'private_runs/screening_230/data/interim/features/a2a101f52a0aa418'
    for p,h,fill,columns in tuples:
        assert fill==(p.parent!=base and 'sequence_flatten' not in str(p.parent))
    nm=run.common.read_json(run.NM_PROTOCOL)
    assert nm['primary']==protocol['primary'] and nm['neural_controls']==protocol['neural_controls']
    peak=run.guard()
    assert peak<2*1024**3
    run.common.write_json(path,dict(status='passed',source_sha256=run.common.sha256(__file__),
        runner_sha256=run.common.sha256(run.__file__),protocol_sha256=run.common.sha256(run.OUT/'protocol.json'),
        controls=checks,discovery=run.DISCOVERY,exact_primary_neural_binding=True,peak_bytes=peak,
        private_labels_read=False,model_fit=False,tuples=[dict(path=str(p),sha256=h,fill=fill,columns=c) for p,h,fill,c in tuples]))
    print('PREFLIGHT_PASS',checks,'peak',peak,flush=True)


if __name__=='__main__':main()
