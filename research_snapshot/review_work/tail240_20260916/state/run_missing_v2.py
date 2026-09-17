"""Canonical JSON declaration plus exact old-control schema guard."""
import json
from pathlib import Path
import run_missing as frozen
from taxiout.artifacts import read_json, sha256, write_json

ROOT = frozen.state.ROOT
old = read_json(ROOT / 'private_runs/tail240_20260916/state/models_v1/protocol.json')
for name, expected in old['source_hashes'].items():
    assert sha256(ROOT / name) == expected
frozen.ARMS = list(frozen.ARMS)
frozen.OUT = frozen.external_path(ROOT / 'private_runs/tail240_20260916/state/models_v2')
frozen.OUT.mkdir(parents=True, exist_ok=True)
receipt = {'wrapper_sha256': sha256(__file__), 'frozen_runner_sha256': sha256(frozen.__file__),
           'old_declaration_sha256': sha256(ROOT/'private_runs/tail240_20260916/state/models_v1/protocol.json'),
           'repair': 'Declared ARMS tuple canonicalized to JSON list before repeated declaration equality; no model semantics or fit performed in v1',
           'original_control_manifest_sha256': sha256(ROOT/'private_runs/breakthrough_20260916/missing/id_context_v1/models/historical_template_F1_s20260916/manifest.json')}
path = frozen.OUT / 'execution_wrapper.json'
if path.exists():
    assert read_json(path) == receipt
else:
    write_json(path, receipt)
loader = frozen.load_missing


def checked_loader():
    x, meta = loader()
    original = read_json(ROOT/'private_runs/breakthrough_20260916/missing/id_context_v1/models/historical_template_F1_s20260916/manifest.json')
    assert list(x) == original['features_used'] and len(x.columns) == 76
    return x, meta


frozen.load_missing = checked_loader


if __name__ == '__main__':
    frozen.main()
    for name, expected in old['source_hashes'].items():
        assert sha256(ROOT / name) == expected
