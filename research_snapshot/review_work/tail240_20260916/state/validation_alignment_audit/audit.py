"""Bounded metadata-only validation history audit; no data/model loading."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'review_work'))
import next230_common as common

sources = {}
def read(relative):
    path = ROOT / relative
    raw = path.read_bytes()
    sources[relative] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)

folds = {f: read(f'private_runs/next_230/models/clock_and_rome_ensemble_{f}_s20260910/manifest.json') for f in ['F1', 'F2', 'F3', 'G1']}
f2, g1 = folds['F2'], folds['G1']
assert f2['score_id_hash'] == g1['score_id_hash']
assert f2['split']['stages']['score'] == g1['split']['stages']['score']
assert all(v == 0 for m in [f2, g1] for v in m['split']['purged_related_departures'].values())
assert f2['config'] == g1['config']

names = {
    'F1': ['residual_long_proxy_specialist-F1-20260916T024604307171-6c34f3be', 'residual_long_proxy_specialist-F1-20260916T024702111872-945ff22a'],
    'F3': ['residual_long_proxy_specialist-F3-20260916T024701784772-6c34f3be', 'residual_long_proxy_specialist-F3-20260916T024948148854-945ff22a'],
}
curves = {}
for fold, runs in names.items():
    old, new = [read(f'private_runs/screening_230/models/{name}/manifest.json') for name in runs]
    assert old['status'] == new['status'] == 'complete'
    assert old['split'] == new['split']
    differences = {k: [old['config'].get(k), new['config'].get(k)] for k in old['config'] if old['config'].get(k) != new['config'].get(k)}
    assert differences == {'train_sample': [250000, 500000]}
    curves[fold] = dict(runs=runs, config_differences=differences, split=old['split'],
        baseline_rmse=old['metrics']['overall']['rmse_sec'], candidate_rmse=new['metrics']['overall']['rmse_sec'],
        gain_sec=old['metrics']['overall']['rmse_sec']-new['metrics']['overall']['rmse_sec'],
        baseline_training=old['training'], candidate_training=new['training'])

final = read('private_runs/submission_v2/protocol.json')
ready = read('private_runs/submission_v2/submission_ready.json')
official = read('private_runs/submission_v2/organizer_v2_result.json')
assert official['used_pairs'] == ready['rows'] == 344841
report = dict(scope='Metadata-only, no new splits/models, no raw data read',
    folds={f: dict(split=m['split'], overall_rmse=m['metrics']['overall']['rmse_sec']) for f,m in folds.items()},
    october_paired=dict(score_ids_hash=f2['score_id_hash'], n=f2['split']['stages']['score']['n'],
        f2_rmse=f2['metrics']['overall']['rmse_sec'], g1_rmse=g1['metrics']['overall']['rmse_sec'],
        difference_sec=g1['metrics']['overall']['rmse_sec']-f2['metrics']['overall']['rmse_sec'],
        config_equal=True, all_purges_zero=True, same_tune=False),
    historical_two_size_comparison=curves,
    final_training=dict(rows=final['training_rows'], policy=final['final_training'],
        ratio_to_f1_refit=final['training_rows']/folds['F1']['split']['stages']['refit']['n'],
        added_rows=final['training_rows']-folds['F1']['split']['stages']['refit']['n']),
    v2_recipe_pair=dict(local=ready['local_validation_rmse_sec'], official=official['score'],
        official_minus_local=official['score']-ready['local_validation_rmse_sec']), sources=sources)
folder=common.external_path(ROOT/'private_runs/tail240_20260916/state/validation_alignment_audit/v1')
folder.mkdir(parents=True, exist_ok=False)
common.write_json(folder/'receipt.json', report)
print(json.dumps({k:v for k,v in report.items() if k not in ['sources','folds','historical_two_size_comparison']}, indent=2))
