"""Apply declared three-expert tune weights; no coefficient fitting here."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import prepare_simplex3 as prepared

ROOT, OUT = prepared.ROOT, prepared.OUT
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'stacking'))
from run_simplex_v2 import day_sensitivity
from taxiout.metrics import evaluate, paired_stability, season_score


def main():
    read_json, write_json, sha256 = prepared.read_json, prepared.write_json, prepared.sha256
    protocol = read_json(OUT/'protocol.json')
    preparation = read_json(OUT/'preparation.json')
    assert preparation['protocol_sha256'] == sha256(OUT/'protocol.json')
    for name, digest in protocol['source_hashes'].items():
        assert sha256(Path(__file__).parent/name) == digest
    scoring = {'created_utc': prepared.utc_now(), 'source_sha256': sha256(__file__),
               'preparation_sha256': sha256(OUT/'preparation.json'), 'protocol_sha256': sha256(OUT/'protocol.json'),
               'variants': ['global3', 'airport3'], 'no_training': True,
               'seed_scope': 'Single expertseed20260916, fixed deterministic convex weights; no multiseed robustness claim',
               'protected_scope': 'Every missing,negative,long proxy remains V2 exactly'}
    if (OUT/'scoring_protocol.json').exists():
        raise ValueError('Preserve existing score attempt')
    write_json(OUT/'scoring_protocol.json', scoring)
    results = {}
    for fold in ['F1', 'F3']:
        directory = OUT/f'{fold}_score'
        directory.mkdir()
        weights = read_json(OUT/f'{fold}_weights.json')
        assert sha256(OUT/f'{fold}_weights.json') == preparation['folds'][fold]['weights_sha256']
        reference, refrecord = prepared.common.reference(fold)
        split = refrecord['split']
        start, end = split['spec']['score']
        meta = pd.read_parquet(ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet',
                               columns=[prepared.ID, 'ADEP_mvt', 'proxy_sec'],
                               filters=[(prepared.MOVEMENT, '>=', pd.Timestamp(start,tz='UTC')),
                                        (prepared.MOVEMENT, '<', pd.Timestamp(end,tz='UTC'))]).set_index(prepared.ID)
        local = meta.loc[reference[prepared.ID]]
        eligible = (np.isfinite(local.proxy_sec)&local.proxy_sec.between(0,7200)).to_numpy()
        experts = []
        for expert in weights['experts']:
            receipt = protocol['sources'][fold][expert]
            folder = Path(receipt['directory'])
            assert sha256(folder/'manifest.json') == receipt['manifest_sha256']
            assert prepared.object_hash(receipt['split']) == prepared.object_hash(split)
            assert sha256(folder/'candidate.parquet') == receipt['outputs']['candidate.parquet']
            frame = pd.read_parquet(folder/'candidate.parquet')
            np.testing.assert_array_equal(frame[prepared.ID], reference[prepared.ID])
            np.testing.assert_array_equal(frame[prepared.TARGET], reference[prepared.TARGET])
            experts.append(frame.prediction_sec.to_numpy())
        matrix = np.column_stack(experts)
        controls = {'reference': reference, 'leaf63_full': pd.read_parquet(Path(protocol['sources'][fold]['lgb63']['directory'])/'candidate.parquet')}
        v3 = ROOT/'private_runs/breakthrough_20260916/models/deeper_audit/simplex_v3'/fold
        v3record = read_json(v3/'manifest.json')
        for label, filename in [('v3_global', 'global_simplex.parquet'), ('v3_airport', 'airport_shrunk_simplex.parquet')]:
            assert sha256(v3/filename) == v3record['outputs'][filename]
            controls[label] = pd.read_parquet(v3/filename)
        metrics, variants = {}, {}
        for variant in ['global3', 'airport3']:
            coefficient = (np.tile(weights['global'], (eligible.sum(),1)) if variant=='global3' else
                           np.array([weights['airports'].get(a,weights['global']) for a in local.ADEP_mvt.astype(str).to_numpy()[eligible]]))
            assert np.all(coefficient>=0) and np.allclose(coefficient.sum(1),1.)
            values = reference.prediction_sec.to_numpy().copy()
            values[eligible] = np.sum(matrix[eligible]*coefficient,axis=1)
            np.testing.assert_array_equal(values[~eligible],reference.prediction_sec.to_numpy()[~eligible])
            assert np.all(values[eligible]>=matrix[eligible].min(1)-1e-8) and np.all(values[eligible]<=matrix[eligible].max(1)+1e-8)
            frame = reference.drop(columns=[prepared.TARGET,'error_sec','squared_error','label_bin','month','day'],errors='ignore').copy()
            frame['prediction_sec'] = values
            metric, errors = evaluate(frame, reference[[prepared.ID,prepared.TARGET]])
            errors.to_parquet(directory/f'{variant}.parquet',index=False)
            replay = pd.read_parquet(directory/f'{variant}.parquet')
            np.testing.assert_array_equal(replay.prediction_sec, values)
            comparisons = {name:{'stability':paired_stability(control,errors,repetitions=1000),
                                'day_removals':day_sensitivity(control,errors)} for name,control in controls.items()}
            protected_sse = float(errors.squared_error.to_numpy()[~eligible].sum())
            variants[variant] = {'metrics':metric,'comparisons':comparisons,'protected_n':int((~eligible).sum()),
                                 'support_floor_rmse':float(np.sqrt(protected_sse/len(errors))),
                                 'protected_sse_share':protected_sse/float(errors.squared_error.sum()),'replay_delta':0.}
            metrics[variant] = metric['overall']
        record = {'status':'complete','fold':fold,'variants':variants,'weights_sha256':sha256(OUT/f'{fold}_weights.json'),
                  'protocol_sha256':sha256(OUT/'scoring_protocol.json'),
                  'outputs':{p.name:sha256(p) for p in directory.iterdir() if p.is_file()}}
        write_json(directory/'manifest.json',record)
        results[fold] = record
        print('SCORED',fold,{v:m['rmse_sec'] for v,m in metrics.items()},flush=True)
    seasonal = {v:season_score(results['F1']['variants'][v]['metrics']['overall'],results['F3']['variants'][v]['metrics']['overall']) for v in ['global3','airport3']}
    write_json(OUT/'score_summary.json',{'created_utc':prepared.utc_now(),'seasonal_rmse':seasonal,'seed_scope':scoring['seed_scope'],
                                       'source_sha256':sha256(__file__),'folds':results})
    print('SEASONAL',seasonal,flush=True)


if __name__ == '__main__':
    main()
