"""Read-only fit/tune scaled-model magnitude, support and observable-neighbor audit."""
import lightgbm
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'review_work/tail240_20260916/models'))
import normalized_missing_tune as base
common, ID, TARGET, TIME = base.common, base.ID, base.TARGET, base.TIME
OUT = common.external_path(ROOT/'private_runs/tail240_20260916/state/scaled_forensics/v1')
GROUPS = ['airport', 'airport_flight', 'airport_stand', 'source_record_regime']


def ess(weights):
    return float(weights.sum()**2/np.sum(weights**2)) if len(weights) else 0.


def numeric(series):
    return pd.to_numeric(series).replace(-999999, np.nan).to_numpy(float)


def distance(query, candidates):
    """Fixed label-free clock/support distance; no learned metric or threshold search."""
    score = np.zeros(len(candidates))
    for field, scale in [('schedule_proxy_sec',3600.), ('idctx_peer_age_w8_sec',3600.),
                         ('schedule_calendar_day_delta',1.)]:
        left = float(query[field])
        right = numeric(candidates[field])
        if left == -999999 or not np.isfinite(left):
            score += np.isfinite(right)
        else:
            delta = np.abs(right-left)/scale
            score += np.where(np.isfinite(delta), np.log1p(delta), 1.)
    for field in ['flight', 'stand', 'runway', 'destination']:
        score += (candidates[field].astype(str).to_numpy() != str(query[field])).astype(float)
    return score


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    protocol = dict(source_sha256=common.sha256(__file__), scope='Originalpurgedmissing fit/tune F1/F3 only; no fitting or score/ranking labels.',
        source_model='Frozen normalized_missing_tune_v1 observed_schedule_scale; rawscale target identity preserved.',
        error_ranks=[1,5,10,20,50], support_keys=GROUPS,
        counterpart='Sameairport and observedsource_record_regime; exclude queryID and samenonnullFLIGHT_ID; fixed label-free distance ofscheduleproxy/IDpeerage/calendarday pluscategorymismatches. Report closest3 and best-predicted amongclosest20tune only aftermatching.',
        limits='Toperrors andgoodprediction comparison are target-defined diagnostics, not inferenceroutes. Raw private rows staylocal. No causality or irreducibility claim fromsimilarity.')
    common.write_json(OUT/'protocol.json', protocol)
    x, meta = base.shared.load_missing()
    indexed = meta.set_index(ID)
    all_reports = {}
    for fold in ['F1','F3']:
        idx, split, _ = common.fold_data(meta, fold, full=True)
        missing = ~np.isfinite(meta.proxy_sec.to_numpy())
        pos = {stage:idx[stage][missing[idx[stage]]] for stage in ['fit','tune']}
        frames = {stage:x.loc[meta.iloc[rows][ID]].copy() for stage, rows in pos.items()}
        labels = {stage:meta.iloc[rows][TARGET].to_numpy(float) for stage, rows in pos.items()}
        model_dir = base.OUT/fold/'observed_schedule_scale'
        manifest = common.read_json(model_dir/'manifest.json')
        for name, digest in manifest['outputs'].items():
            assert common.sha256(model_dir/name) == digest
        saved = joblib.load(model_dir/'model.joblib')
        tune = pd.read_parquet(model_dir/'tune.parquet')
        assert np.array_equal(tune[ID], frames['tune'].index)
        assert np.array_equal(tune.raw_target_sec, labels['tune'])
        replay = base.reconstruct(saved['model'].predict(saved['encoder'].transform(frames['tune']), num_iteration=manifest['steps']), base.scale_of(frames['tune']))
        assert np.array_equal(replay, tune.prediction_sec)
        controls = {}
        for name, path in [('unscaled',base.OUT/fold/'unscaled/tune.parquet'),
                           ('context451',ROOT/f'private_runs/tail240_20260916/state/normalized_context/v1/{fold}/context451/tune.parquet'),
                           ('historical_template',ROOT/f'private_runs/tail240_20260916/state/models_v2/control/{fold}/tune_predictions.parquet')]:
            frame = pd.read_parquet(path)
            assert np.array_equal(frame[ID], tune[ID])
            controls[name] = frame.prediction_sec.to_numpy(float)
        weights = base.scale_of(frames['fit'])**2/manifest['normalizer']
        y = labels['tune']
        pred = tune.prediction_sec.to_numpy()
        squared = (y-pred)**2
        rank = np.argsort(-squared, kind='stable')
        query = frames['tune'].copy()
        query['raw_Y'] = y
        query['prediction'] = pred
        query['error'] = pred-y
        query['squared_error'] = squared
        query['scale'] = tune.scale.to_numpy()
        query['error_z'] = (pred-y)/query.scale
        query['raw_Z'] = (y-900.)/query.scale
        query['movement_time'] = indexed.loc[query.index,TIME]
        for name, values in controls.items():
            query[name+'_prediction'] = values
        for key in GROUPS:
            table = pd.DataFrame({key:frames['fit'][key].astype(str).to_numpy(), 'weight':weights, 'weight2':weights**2})
            grouped = table.groupby(key).agg(n=('weight','size'), sum_w=('weight','sum'), sum_w2=('weight2','sum'))
            query[key+'_fit_n'] = query[key].astype(str).map(grouped.n).fillna(0)
            query[key+'_fit_ess'] = query[key].astype(str).map(grouped.sum_w**2/grouped.sum_w2).fillna(0)
        for name, mapping in saved['encoder'].categories.items():
            query['unseen_'+name] = ~frames['tune'][name].astype('string').isin(mapping)
        comparisons = []
        for query_position in rank[:20]:
            q = frames['tune'].iloc[query_position]
            qid = frames['tune'].index[query_position]
            qfid = indexed.loc[qid,'FLIGHT_ID_mvt']
            for stage in ['fit','tune']:
                candidates = frames[stage]
                mask = candidates.airport.eq(q.airport) & candidates.source_record_regime.eq(q.source_record_regime) & (candidates.index != qid)
                if pd.notna(qfid):
                    mask &= indexed.loc[candidates.index,'FLIGHT_ID_mvt'].ne(qfid).to_numpy()
                selected = candidates.loc[mask]
                if not len(selected):
                    continue
                distances = distance(q, selected)
                order = np.argsort(distances, kind='stable')
                closest = selected.index[order[:20]]
                choices = [('nearest'+str(i+1), closest[i]) for i in range(min(3,len(closest)))]
                if stage == 'tune':
                    best = query.loc[closest,'squared_error'].idxmin()
                    choices.append(('best_error_among_nearest20',best))
                for role, nid in choices:
                    n = selected.loc[nid]
                    comparisons.append(dict(fold=fold, query_id=qid, neighbor_id=nid, stage=stage, role=role,
                        distance=float(distances[selected.index.get_loc(nid)]),
                        query_airport=q.airport, query_flight=q.flight, neighbor_flight=n.flight,
                        same_flight=q.flight == n.flight, same_stand=q.stand == n.stand,
                        query_Y=float(y[query_position]), neighbor_Y=float(indexed.loc[nid,TARGET]),
                        query_schedule=float(q.schedule_proxy_sec), neighbor_schedule=float(n.schedule_proxy_sec),
                        query_scale=float(query.loc[qid,'scale']), neighbor_scale=float(base.scale_of(selected.loc[[nid]])[0]),
                        query_prediction=float(pred[query_position]),
                        neighbor_prediction=float(query.loc[nid,'prediction']) if stage=='tune' else None,
                        query_error=float(pred[query_position]-y[query_position]),
                        neighbor_error=float(query.loc[nid,'error']) if stage=='tune' else None,
                        query_time=indexed.loc[qid,TIME], neighbor_time=indexed.loc[nid,TIME]))
        counterpart = pd.DataFrame(comparisons)
        counterpart.to_parquet(OUT/(fold+'_counterparts.parquet'),index=False)
        query.iloc[rank].reset_index().to_parquet(OUT/(fold+'_error_rows.parquet'),index=False)
        selected = query.iloc[rank[:20]]
        report = dict(original_total_tune_rows=len(idx['tune']), missing_fit_rows=len(frames['fit']), missing_tune_rows=len(query),
            normalized_rmse=float(np.sqrt(squared.mean())), normalized_missing_sse=float(squared.sum()),
            missing_error_only_full_tune_rmse=float(np.sqrt(squared.sum()/len(idx['tune']))),
            fit_weight_ess=ess(weights), fit_scale_quantiles=np.quantile(base.scale_of(frames['fit']),[0,.5,.9,.99,1]).tolist(),
            tune_scale_quantiles=np.quantile(query.scale,[0,.5,.9,.99,1]).tolist(),
            controls={name:dict(rmse=float(np.sqrt(np.mean((y-values)**2))), top20_same_rows_sse=float(np.sum((y[rank[:20]]-values[rank[:20]])**2))) for name,values in controls.items()},
            error_budget={str(k):dict(sse_share=float(squared[rank[:k]].sum()/squared.sum()),
                remaining_rmse_oracle=float(np.sqrt((squared.sum()-squared[rank[:k]].sum())/len(y)))) for k in [1,5,10,20,50]},
            top20=dict(sse_share=float(selected.squared_error.sum()/squared.sum()),
                scale_min=float(selected.scale.min()), scale_max=float(selected.scale.max()),
                median_abs_error_z=float(selected.error_z.abs().median()),
                unseen_flight=int(selected.unseen_flight.sum()),
                fit_zero_support={key:int(selected[key+'_fit_n'].eq(0).sum()) for key in GROUPS},
                median_fit_ess={key:float(selected[key+'_fit_ess'].median()) for key in GROUPS},
                airport_counts=selected.airport.astype(str).value_counts().to_dict(),
                same_airport_regime_good_counterparts=int(counterpart.loc[counterpart.role.eq('best_error_among_nearest20'),'neighbor_error'].abs().lt(600).sum())),
            producer_manifest_sha256=common.sha256(model_dir/'manifest.json'), split=split,
            statement='Oracle remainingRMSE is errorbudget only, not achievableprediction; allactualrows retained. Fit/tune labels used only after observable neighbor selection.')
        common.write_json(OUT/(fold+'.json'), report)
        all_reports[fold] = report
        print('AUDIT',fold,report,flush=True)
    common.write_json(OUT/'summary.json',dict(status='complete',protocol_sha256=common.sha256(OUT/'protocol.json'),folds=all_reports,
        outputs={path.name:common.sha256(path) for path in OUT.iterdir() if path.is_file() and path.name not in ['summary.json']}))


if __name__ == '__main__':
    with threadpool_limits(2):
        main()
