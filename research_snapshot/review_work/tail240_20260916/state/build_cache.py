"""Build immutable original-fold state caches without fitting a predictor."""
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import airport_state as state
import pandas as pd
from taxiout.config import load_config
from taxiout.splits import make_fold
from taxiout.paths import external_path


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', nargs='+', default=['F1', 'F3'])
    args = parser.parse_args()
    root = state.ROOT
    protocol_path = root / 'private_runs/tail240_20260916/state/protocol.json'
    protocol = json.loads(protocol_path.read_text())
    assert sha(state.__file__) == protocol['source_sha256']
    source = root / 'private_runs/screening_230/data/interim/audit/departures.parquet'
    audit = json.loads((root / 'private_runs/screening_230/reports/data_audit.json').read_text())
    source_digest = sha(source)
    assert source_digest == audit['artifacts'][source.name]
    out = external_path(root / 'private_runs/tail240_20260916/state/cache_v1')
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    meta = pd.read_parquet(source, columns=[state.ID, state.FID, state.TIME, state.AIRPORT, state.TARGET, 'proxy_sec'])
    assert len(meta) == 2085047
    state.guard()
    receipt = {'status': 'running', 'created_utc': datetime.now(timezone.utc).isoformat(),
               'source_metadata_sha256': source_digest, 'protocol_sha256': sha(protocol_path),
               'builder_sha256': sha(__file__), 'adapter_sha256': sha(state.__file__), 'folds': {}}
    for fold in args.folds:
        idx, split = make_fold(meta.drop(columns=state.TARGET), load_config('configs/folds.yaml')[fold])
        report = {'split': split, 'stages': {}}
        for stage in ('fit', 'tune', 'refit', 'score'):
            before = time.perf_counter()
            features, detail = state.stage_features(meta, idx, stage)
            features.insert(0, state.ID, meta.iloc[idx[stage]][state.ID].to_numpy())
            path = out / f'{fold}_{stage}.parquet'
            features.to_parquet(path, index=False)
            report['stages'][stage] = {'rows': len(features), 'path': str(path.relative_to(root)),
                                       'sha256': sha(path), 'runtime_seconds': time.perf_counter() - before,
                                       'detail': detail}
            print(f'{fold} {stage} {len(features)} rows {time.perf_counter()-before:.3f}s', flush=True)
            del features
        features, detail = state.stage_features(meta, idx, 'score', history_blackout_days=182)
        features.insert(0, state.ID, meta.iloc[idx['score']][state.ID].to_numpy())
        path = out / f'{fold}_score_blackout182.parquet'
        features.to_parquet(path, index=False)
        report['score_blackout182'] = {'rows': len(features), 'path': str(path.relative_to(root)), 'sha256': sha(path), 'detail': detail}
        receipt['folds'][fold] = report
        (out / 'progress.json').write_text(json.dumps(receipt, indent=2, default=str))
    assert sha(source) == source_digest and sha(state.__file__) == protocol['source_sha256']
    import psutil
    receipt.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
                   runtime_seconds=time.perf_counter()-started, rss_bytes_at_finish=psutil.Process().memory_info().rss,
                   statement='Features only, no fitted downstream model or predictive result; source metadata hash verified before/after.')
    (out / 'manifest.json').write_text(json.dumps(receipt, indent=2, default=str))
    print('CACHE_COMPLETE', receipt['runtime_seconds'], receipt['rss_bytes_at_finish'], flush=True)


if __name__ == '__main__':
    main()
