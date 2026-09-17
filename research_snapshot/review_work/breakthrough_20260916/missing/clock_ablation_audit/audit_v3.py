"""Load frozen audit under a distinct name; allow matched tune early stopping."""
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('clock_ablation_frozen',Path(__file__).with_name('audit.py'))
frozen=importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)


def checked(folder):
    record=frozen.common.read_json(folder/'manifest.json')
    assert record['status']=='complete', str(folder)
    for name,digest in record['outputs'].items():
        assert frozen.common.sha256(folder/name)==digest
    assert record['reload_max_abs_delta']==0
    assert record['fit']['steps']==record['refit']['steps']
    assert 1<=record['fit']['steps']<=600
    if folder.parent!=frozen.NEW:
        assert record['fit']['steps']==600
    assert record['threads']==4 and record['seed']==20260916
    return record


if __name__=='__main__':
    frozen.checked=checked
    frozen.OUT=frozen.OUT/'attempt_v3'
    original=frozen.common.write_json
    def write(path,payload):
        payload['audit_wrapper_sha256']=frozen.common.sha256(__file__)
        payload['selected_steps']={fold:frozen.common.read_json(frozen.NEW/f'lightgbm_sequence_no_dep_clocks_aobt_allfinite_{fold}_s20260916'/'manifest.json').get('fit',{}).get('steps') for fold in ['F1','F3']}
        original(path,payload)
    frozen.common.write_json=write
    frozen.main()
