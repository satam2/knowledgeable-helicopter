"""Both frozen four-expert ordinary routes, fixed nonordinary and missing routes."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'review_work/campaign_20260916'))
sys.path.insert(0, str(ROOT/'review_work/breakthrough_20260916/models/verification_id_template'))
import common
from audit import comparison, metric
from taxiout.metrics import evaluate, season_score
ID, TARGET, TIME = common.ID, common.TARGET, common.MOVEMENT
BASE = ROOT/'private_runs/breakthrough_20260916'
OUT = BASE/'missing/route_composition_v3'
STACK = BASE/'models/context_gate/simplex4_context_v1'
VARIANTS = ['global4', 'airport4']
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def registry(fold):
    return {**{v:(STACK/f'{fold}_score', v+'.parquet') for v in VARIANTS},
        'nonordinary':(BASE/'deeper_lgb/combined'/f'lightgbm_leaf63_aobt_allfinite_{fold}_s20260916', 'candidate.parquet'),
        'missing':(BASE/'models/missing_forest'/f'extratrees_missing_template_idcontext_{fold}_s20260916', 'blend25.parquet'),
        'previous_composition':(BASE/'missing/route_composition_v2'/fold, 'candidate.parquet'),
        'previous_ordinary':(BASE/'models/deeper_audit/simplex_v3'/fold, 'airport_shrunk_simplex.parquet')}


def declaration():
    verification = common.read_json(STACK/'independent_verification.json')
    assert verification['status'] == 'passed'
    assert verification['score_summary_sha256'] == common.sha256(STACK/'score_summary.json')
    sources = {}
    for fold in ['F1', 'F3']:
        sources[fold] = {}
        for name, (directory, filename) in registry(fold).items():
            record = common.read_json(directory/'manifest.json')
            assert record['status'] == 'complete'
            sources[fold][name] = dict(directory=str(directory), filename=filename,
                manifest_sha256=common.sha256(directory/'manifest.json'), prediction_sha256=record['outputs'][filename])
    payload = dict(source_sha256=common.sha256(__file__), sources=sources, variants=VARIANTS,
        folds=['F1','F3'], seed=20260916,
        stacker_receipts={p.name:common.sha256(p) for p in [STACK/'protocol.json', STACK/'preparation.json',
            STACK/'scoring_protocol.json', STACK/'independent_verification.json', STACK/'F1_weights.json', STACK/'F3_weights.json']},
        routing='Observed proxy finite0..7200 inclusive: corresponding frozen global4 or airport4. Finite negative/>7200: original225 leaf63 candidate. Missing: ET fixed25percent V2 blend.',
        selection='Both variants fixed before composition score assembly; report both. No score-fitted weights or variant selection.',
        floor='None; no clipping, support projection, label changes, exclusions, new training or weight fit.',
        diagnostics='Full original cohorts; day removals vs V2/old ordinary/old composition/own ordinary; top2 missing gains vs V2; top10 overall gains vs V2 and old composition; weighted error budget.',
        caveat='Adaptive exposed development; no independent holdout or official score. Missing-route gains are tail dependent.',
        declaration_timing='Written before this process opens any row prediction file.')
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/'protocol.json'
    if path.exists():
        assert common.read_json(path)['declaration'] == payload
    else:
        common.write_json(path, dict(created_utc=common.utc_now(), declaration=payload))
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--declare-only', action='store_true')
    args = parser.parse_args()
    protocol = declaration()
    if args.declare_only:
        print('DECLARED_BOTH_VARIANTS_WITHOUT_ROW_READS', flush=True)
        return
    assert not (OUT/'summary.json').exists()
    meta_path = ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert common.sha256(meta_path) == common.read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][meta_path.name]
    meta = pd.read_parquet(meta_path, columns=[ID,TARGET,TIME,'FLIGHT_ID_mvt','proxy_sec'])
    results = {v:{} for v in VARIANTS}
    for fold in ['F1','F3']:
        reference, refrec = common.reference(fold)
        idx, split, _ = common.fold_data(meta, fold, full=True)
        assert common.object_hash(split) == common.object_hash(refrec['split'])
        np.testing.assert_array_equal(reference[ID], meta.iloc[idx['score']][ID])
        np.testing.assert_array_equal(reference[TARGET], meta.iloc[idx['score']][TARGET])
        proxy = meta.iloc[idx['score']].proxy_sec.to_numpy(float)
        np.testing.assert_allclose(proxy, reference.proxy_sec, rtol=0, atol=0, equal_nan=True)
        finite = np.isfinite(proxy)
        masks = {'ordinary':finite&(proxy>=0)&(proxy<=7200), 'nonordinary':finite&((proxy<0)|(proxy>7200)), 'missing':~finite}
        assert np.all(sum(m.astype(int) for m in masks.values()) == 1)
        components = {}
        for name, receipt in protocol['sources'][fold].items():
            source = Path(receipt['directory'])
            assert common.sha256(source/'manifest.json') == receipt['manifest_sha256']
            assert common.sha256(source/receipt['filename']) == receipt['prediction_sha256']
            frame = pd.read_parquet(source/receipt['filename'])
            np.testing.assert_array_equal(frame[ID], reference[ID])
            np.testing.assert_array_equal(frame[TARGET], reference[TARGET])
            components[name] = frame.prediction_sec.to_numpy(float)
        y, base, days = reference[TARGET].to_numpy(float), reference.prediction_sec.to_numpy(float), reference.day.to_numpy()
        for variant in VARIANTS:
            directory = OUT/variant/fold
            directory.mkdir(parents=True, exist_ok=False)
            ordinary = components[variant]
            np.testing.assert_array_equal(ordinary[~masks['ordinary']], base[~masks['ordinary']])
            pred = ordinary.copy()
            pred[masks['nonordinary']] = components['nonordinary'][masks['nonordinary']]
            pred[masks['missing']] = components['missing'][masks['missing']]
            np.testing.assert_array_equal(pred[~masks['ordinary']], components['previous_composition'][~masks['ordinary']])
            assert np.isfinite(pred).all()
            route = np.select([masks['missing'],masks['nonordinary']], ['missing','nonordinary'], default='ordinary')
            frame = reference.drop(columns=[TARGET,'error_sec','squared_error','label_bin','month','day'], errors='ignore').copy()
            frame['prediction_sec'] = pred
            full, scored = evaluate(frame, reference[[ID,TARGET]])
            scored['composition_route'] = route
            scored.to_parquet(directory/'candidate.parquet', index=False)
            controls = {'V2':base, 'previous_ordinary':components['previous_ordinary'],
                'previous_composition':components['previous_composition'], 'own_ordinary':ordinary, 'composed':pred}
            comparisons = {name:comparison(y,value,pred,days) for name,value in controls.items() if name!='composed'}
            missing_gain = np.where(masks['missing'],(base-y)**2-(pred-y)**2,-np.inf)
            largest = np.argsort(missing_gain)[-2:]
            keep = ~np.isin(np.arange(len(y)), largest)
            removed = {name:metric(y,value,keep) for name,value in controls.items()}
            sensitivities = {}
            for name in ['V2','previous_composition']:
                gain = (controls[name]-y)**2-(pred-y)**2
                top = np.argsort(gain)[-10:]
                remaining = ~np.isin(np.arange(len(y)),top)
                sensitivities[name] = dict(ids=reference.iloc[top][ID].tolist(),
                    control=metric(y,controls[name],remaining), composed=metric(y,pred,remaining))
                sensitivities[name]['gain_seconds'] = sensitivities[name]['control']['rmse_sec']-sensitivities[name]['composed']['rmse_sec']
            top_rows = scored.iloc[largest][[ID,TARGET,'prediction_sec','day','composition_route']].copy()
            top_rows['V2'] = base[largest]
            top_rows['gain_sse'] = missing_gain[largest]
            top_rows.to_parquet(directory/'missing_top2_gain.parquet',index=False)
            rec = dict(status='complete',fold=fold,variant=variant,source_sha256=common.sha256(__file__),
                protocol_sha256=common.sha256(OUT/'protocol.json'),source_receipts=protocol['sources'][fold],split=split,
                route_counts={k:int(v.sum()) for k,v in masks.items()},reports={'candidate':{'metrics':full}},
                metrics={name:metric(y,value) for name,value in controls.items()}, comparisons=comparisons,
                remove_missing_top2=removed,remove_top10_gain=sensitivities,
                missing_top2=dict(ids=reference.iloc[largest][ID].tolist(),share=float(missing_gain[largest].sum()/missing_gain[masks['missing']].sum())),
                no_floor=True,no_new_weight_fit=True,outputs={p.name:common.sha256(p) for p in directory.glob('*.parquet')})
            common.write_json(directory/'manifest.json',rec)
            results[variant][fold] = rec
            print('COMPOSED',variant,fold,rec['metrics']['composed']['rmse_sec'],flush=True)
    summary = {}
    for variant, folds in results.items():
        summary[variant] = dict(seasonal_rmse={name:season_score(folds['F1']['metrics'][name],folds['F3']['metrics'][name]) for name in folds['F1']['metrics']},
            remove_missing_top2_seasonal={name:season_score(folds['F1']['remove_missing_top2'][name],folds['F3']['remove_missing_top2'][name]) for name in folds['F1']['metrics']},
            every_day_removal_improves_vs_previous=all(r['comparisons']['previous_composition']['all_day_removals_improve'] for r in folds.values()))
    common.write_json(OUT/'summary.json',dict(status='complete',protocol_sha256=common.sha256(OUT/'protocol.json'),variants=summary,
        main_is_unfloored=True,no_variant_selected=True))
    print('SEASONAL',summary,flush=True)


if __name__ == '__main__':
    main()
