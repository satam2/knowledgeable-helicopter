"""Verify mix reweighting from independently bound local replay rows."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import lightgbm
from replay_weight_policy import ROOT, OUT, ID, TARGET, TIME, read, sha, SEASONS, guard
import json
import math
import numpy as np
import pandas as pd
import pyarrow as pa

pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-8)


def routes(frame):
    values = frame.proxy_sec.to_numpy(dtype=float)
    result = np.full(len(frame), 'finite_nonordinary', dtype=object)
    result[np.isfinite(values) & (values >= 0) & (values <= 7200)] = 'ordinary'
    result[~np.isfinite(values)] = 'missing'
    return result


def main():
    guard()
    folder = ROOT/'private_runs/tail240_20260916/score_gap/drift/v2'
    report = read(folder/'receipt.json')
    wrapper = read(folder/'wrapper_receipt.json')
    assert report['status'] == 'complete'
    source = ROOT/'review_work/tail240_20260916/score_gap/drift'
    assert wrapper['source_sha256'] == sha(source/'audit_v2.py')
    assert wrapper['imported_source_sha256'] == report['source_sha256'] == sha(source/'audit_v1.py')
    assert wrapper['receipt_sha256'] == sha(folder/'receipt.json')
    assert report['airport_route_cells_sha256'] == sha(folder/'airport_route_cells.json')
    cells = read(folder/'airport_route_cells.json')
    frame_json = pd.DataFrame(cells)
    frame_csv = pd.read_csv(folder/'airport_route_cells.csv')
    pd.testing.assert_frame_equal(frame_json, frame_csv[frame_json.columns], check_exact=False, rtol=1e-12, atol=1e-8)
    verified = read(OUT/'receipt.json')
    assert verified['status'] == 'passed'
    rankpath = ROOT/'private_runs/submission_v2/ranking_meta.parquet'
    rankbinding = read(rankpath.parent/'ranking_inputs.json')
    assert sha(rankpath) == rankbinding['files'][rankpath.name]
    rank = pd.read_parquet(rankpath, columns=[ID, TIME, 'ADEP_mvt', 'proxy_sec'])
    assert rank[ID].is_unique and len(rank) == 344841
    rank['route'] = routes(rank)
    rank['airport'] = rank.ADEP_mvt.astype(str)
    rank['month'] = rank[TIME].dt.strftime('%Y-%m')
    assert rank.month.value_counts().to_dict() == {'2026-07':192122, '2026-01':152719}
    summary = {}
    unsupported, local_only, small = [], [], []
    for fold, local_month, rank_month in [('F1','2025-07','2026-07'), ('F3','2025-11','2026-01')]:
        replay = OUT/f'{fold}_replay.parquet'
        assert sha(replay) == verified['outputs'][replay.name]
        local = pd.read_parquet(replay)
        assert local[ID].is_unique
        assert set(local[TIME].dt.strftime('%Y-%m')) == {local_month}
        np.testing.assert_array_equal(local.route.to_numpy(), routes(local))
        local['airport'] = local.ADEP_mvt.astype(str)
        local['squared_error'] = np.square(local.local_fold_weights_raw-local[TARGET])
        local_groups = local.groupby(['airport','route'], observed=True).agg(n=(ID,'size'), sse=('squared_error','sum'))
        rank_month_rows = rank.loc[rank.month.eq(rank_month)]
        rank_counts = rank_month_rows.groupby(['airport','route'], observed=True).size()
        produced = {(c['airport'],c['route']):c for c in cells if c['fold']==fold}
        keys = set(local_groups.index) | set(rank_counts.index)
        assert len(produced) == len([c for c in cells if c['fold']==fold])
        assert set(produced) == keys
        old, new = 0., 0.
        for key in sorted(keys):
            nr = int(rank_counts.get(key,0))
            nl = int(local_groups.loc[key,'n']) if key in local_groups.index else 0
            identity = dict(fold=fold,airport=key[0],route=key[1],local_rows=nl,ranking_rows=nr)
            if nr and not nl:
                unsupported.append(identity)
                continue
            if nl and not nr:
                local_only.append(identity)
            if nr and nl < 30:
                small.append(identity)
            mse = float(local_groups.loc[key,'sse']/nl) if nl else 0.
            ls, rs = nl/len(local), nr/len(rank_month_rows)
            delta = (rs-ls)*mse
            expected = dict(**identity,local_month=local_month,ranking_month=rank_month,
                local_share=ls,ranking_share=rs,local_conditional_mse=mse,
                mix_delta_mse=delta,seasonal_mix_delta_mse=SEASONS[fold]*delta)
            assert set(expected) == set(produced[key])
            for name,value in expected.items():
                if isinstance(value,float): close(value,produced[key][name])
                else: assert value == produced[key][name], (key,name)
            old += ls*mse
            new += rs*mse
        assert not unsupported, unsupported
        close(old,local.squared_error.mean())
        computed = dict(local_mse=old,local_rmse=math.sqrt(old),reweighted_mse=new,reweighted_rmse=math.sqrt(new))
        for key,value in computed.items(): close(value,report['composition_diagnostic'][fold][key])
        summary[fold] = dict(**computed,local_rows=len(local),ranking_rows=len(rank_month_rows),cells=len(keys))
        guard()
    old = math.sqrt(sum(SEASONS[f]*summary[f]['local_mse'] for f in SEASONS))
    new = math.sqrt(sum(SEASONS[f]*summary[f]['reweighted_mse'] for f in SEASONS))
    seasonal = dict(local_rmse=old,reweighted_rmse=new,mix_only_rmse_delta=new-old)
    for key,value in seasonal.items():close(value,report['composition_diagnostic']['seasonal'][key])
    result = dict(status='passed',verifier_sha256=sha(__file__),
        producer_receipt_sha256=sha(folder/'receipt.json'),producer_wrapper_sha256=sha(folder/'wrapper_receipt.json'),
        independent_replay_receipt_sha256=sha(OUT/'receipt.json'),ranking_metadata_sha256=sha(rankpath),
        airport_route_json_sha256=sha(folder/'airport_route_cells.json'),airport_route_csv_sha256=sha(folder/'airport_route_cells.csv'),
        all_cell_counts_shares_conditional_mse_and_deltas_verified=True,json_csv_consistent=True,
        folds=summary,seasonal=seasonal,unsupported_ranking_cells=unsupported,local_only_cells=local_only,
        supported_cells_with_fewer_than_30_local_rows=small,
        interpretation='Descriptive mix sensitivity assumes constant local conditional MSE within airport/route/month. It is neither a ranking forecast nor causal attribution of the official gap. November is only a proxy for January; local folds were used in development. A positive supported-cell count does not imply precise conditional error estimates.',
        peak_bytes=guard(),cpu_threads=1,gpu_used=False,fit_used=False,ranking_labels_read=False)
    dest = OUT/'mix_reweight_review.json'
    assert not dest.exists()
    dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
