"""Independent replay, immutable-input and external-output checks for the spike."""

import json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from aviation_spike import OUT, ARMS, invariants
from next230_common import WORKSPACE, verify_protocol
from taxiout.artifacts import read_json, sha256, source_hashes, write_json
from taxiout.paths import external_path
from taxiout.schema import ID, TARGET


def main():
    raw_manifest = verify_protocol()
    protocol = read_json(OUT/'protocol.json')
    assert sha256(WORKSPACE/'review_work/aviation_spike.py') == protocol['runner_sha256']
    assert source_hashes() == protocol['source_hashes']
    results = []
    for fold in ['F1','F3']:
        folder = OUT/fold
        evidence = read_json(folder/'evidence.json')
        for name,digest in evidence['files'].items():
            assert sha256(folder/name) == digest
        base = pd.read_parquet(folder/'score_base.parquet')
        reference = pd.read_parquet(folder/'score_reference.parquet')
        changed = reference.route.eq('residual').to_numpy()
        for arm in ARMS:
            features = base
            if arm != 'control':
                extra = pd.read_parquet(folder/f'{arm}.parquet').loc[base.index]
                features = pd.concat([base,extra],axis=1)
            model = CatBoostRegressor().load_model(str(folder/f'{arm}.cbm'))
            gate = np.clip(model.predict(features,thread_count=4),0,1)
            p = reference.prediction_sec.to_numpy().copy()
            pred = reference.residual_expert.to_numpy()+gate*(reference.direct_expert.to_numpy()-reference.residual_expert.to_numpy())
            p[changed] = pred[changed]
            stored = pd.read_parquet(folder/f'{arm}_predictions.parquet')
            assert np.array_equal(stored[ID],reference[ID])
            assert np.array_equal(stored[TARGET],reference[TARGET])
            assert np.array_equal(p,stored.prediction_sec)
            assert np.array_equal(p[~changed],reference.prediction_sec.to_numpy()[~changed])
            rmse = float(np.sqrt(np.mean((p-stored[TARGET])**2)))
            assert abs(rmse-read_json(folder/f'{arm}_result.json')['rmse_sec']) < 1e-9
            if arm=='control':
                assert np.max(np.abs(p-reference.prediction_sec.to_numpy())) < 1e-9
            results.append({'fold':fold,'arm':arm,'replay_exact':True,'rmse_sec':rmse,
                            'model_sha256':sha256(folder/f'{arm}.cbm'),
                            'predictions_sha256':sha256(folder/f'{arm}_predictions.parquet')})
    leaks = []
    for repo in ['knowledgeable-helicopter','knowledgeable-helicopter-screening']:
        for p in (WORKSPACE/repo).rglob('*'):
            if p.is_file() and p.suffix in {'.parquet','.cbm','.pkl','.npy','.npz','.joblib'}:
                leaks.append(str(p))
    assert not leaks,leaks
    for p in [OUT,WORKSPACE/'output',WORKSPACE/'review_work']:
        external_path(p)
    original = read_json(WORKSPACE/'knowledgeable-helicopter/reports/freeze.json')
    changed = [p for p,d in original['source_hashes'].items() if sha256(WORKSPACE/'knowledgeable-helicopter'/p)!=d]
    assert not changed,changed
    report = {'raw_files_rehashed':len(raw_manifest['files']), 'screening_sources_unchanged':True,
              'original_sources_rehashed':len(original['source_hashes']), 'replays':results,
              'synthetic_invariants':invariants(), 'artifacts_in_checkouts':leaks,
              'pdf_sha256':sha256(WORKSPACE/'output/PRC_2026_Aviation_Research_Proposal.pdf'),
              'methodology_pdf_sha256':sha256(OUT/'sources/ATXOT_indicator_documentation_mar23.pdf')}
    write_json(OUT/'final_verification.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
