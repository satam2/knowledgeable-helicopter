"""Initialize the known-good native library order before the frozen refit runner."""
import normalized_missing_tune
import normalized_missing_refit as frozen
from pathlib import Path
import argparse

frozen.OUT = frozen.common.external_path(frozen.ROOT / 'private_runs/tail240_20260916/models/normalized_missing_refit_v2')


def declaration():
    common, prior, shared = frozen.common, frozen.prior, frozen.shared
    frozen.OUT.mkdir(parents=True, exist_ok=True)
    old = common.read_json(frozen.ROOT / 'private_runs/tail240_20260916/models/normalized_missing_refit_v1/protocol.json')
    assert old['source_sha256'] == common.sha256(frozen.__file__)
    assert old['prior_source_sha256'] == common.sha256(prior.__file__)
    value = dict(old)
    value['runtime_fix'] = dict(wrapper_sha256=common.sha256(__file__),
        prior_failed_protocol_sha256=common.sha256(frozen.ROOT / 'private_runs/tail240_20260916/models/normalized_missing_refit_v1/protocol.json'),
        cause='With exact F1 refit8819x76, numpy/joblib before nativeLightGBM reproduces accessviolation at set_label; prior-first env/native import passes one-tree probe. No data, model, stopping or evaluation changes.',
        probe_source_sha256=common.sha256(Path(__file__).with_name('probe_normalized_runtime.py')))
    value['source_hashes'] = dict(old['source_hashes'])
    value['source_hashes'][str(Path(__file__).relative_to(frozen.ROOT))] = common.sha256(__file__)
    path = frozen.OUT / 'protocol.json'
    if path.exists():
        assert common.read_json(path) == value
    else:
        common.write_json(path, value)
    return value


frozen.declaration = declaration

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--declare-only', action='store_true')
    parser.add_argument('--gate-receipt', type=Path)
    args = parser.parse_args()
    if args.declare_only:
        declaration()
    else:
        assert args.gate_receipt is not None
        frozen.run(args.gate_receipt)
