"""Check fit/tune/refit cohorts and calibration rows against original purged folds."""

import numpy as np
import pandas as pd

from next230_common import OUT, load_data, load_reference, same_split
from taxiout.artifacts import object_hash, read_json, sha256, write_json
from taxiout.models.residual import proxy_status
from taxiout.schema import ID, TARGET


def verify():
    x, meta, labels = load_data()
    reference = {f: load_reference(f, meta) for f in ['F1', 'F2', 'F3', 'G1']}
    receipts = []
    for file in sorted((OUT/'models').glob('*/manifest.json')):
        r=read_json(file)
        if r['status'] != 'complete':
            raise ValueError('Split audit requires completed model containers')
        _,_,idx,split=reference[r['fold']]
        if not same_split(r['split'],split):
            raise ValueError('Stored split differs from reconstructed original fold')
        config=r['config']
        if 'fit_id_hash' in r:
            status=proxy_status(meta.proxy_sec,config)
            mask=((status=='missing') & np.isfinite(meta.schedule_sec)) if r['candidate'].startswith('rome_') else status=='present'
            for stage in ['fit','tune','refit']:
                positions=idx[stage][mask[idx[stage]]]
                if r[f'{stage}_id_hash'] != object_hash(x.index[positions].tolist()):
                    raise ValueError('Stored component cohort mismatch')
            if np.intersect1d(idx['fit'],idx['tune']).size or np.intersect1d(idx['refit'],idx['score']).size:
                raise ValueError('Temporal training leakage')
        elif 'refit_id_hash' in r and 'tree_selection_source' in r:
            from pathlib import Path
            status=proxy_status(meta.proxy_sec,config)
            pos=idx['refit'][status[idx['refit']]=='present']
            if r['refit_id_hash']!=object_hash(x.index[pos].tolist()):
                raise ValueError('Seed repeat refit cohort mismatch')
            source=Path(r['tree_selection_source'])/'manifest.json'
            original=read_json(source)
            if sha256(source)!=r['tree_selection_source_manifest_sha256'] or r['trees']!=original['trees']:
                raise ValueError('Repeated seed changed selected tree count')
        if 'calibration_id_hash' in r:
            pos=idx['tune']
            mask=(proxy_status(meta.proxy_sec,config)=='missing') & np.isfinite(meta.schedule_sec) & meta.ADEP_mvt.eq('LIRF')
            pos=pos[mask.iloc[pos].to_numpy()]
            if r['calibration_id_hash'] != object_hash(x.index[pos].tolist()):
                raise ValueError('Calibration uses non-tuning rows')
        if r.get('learned_gate'):
            from pathlib import Path
            from next230_clock_gate import gate_targets
            direct=pd.read_parquet(Path(r['direct_expert_path'])/'tune_direct.parquet')
            residual=pd.read_parquet(Path(r['residual_expert_path'])/'tune_predictions.parquet')
            if not np.array_equal(direct[ID],residual[ID]):
                raise ValueError('Learned gate expert IDs mismatch')
            pos=pd.Index(meta[ID]).get_indexer(direct[ID])
            if not np.isin(pos,idx['tune']).all():
                raise ValueError('Learned gate rows outside tuning month')
            eligible,_,_,_=gate_targets(direct[TARGET],meta.iloc[pos].proxy_sec.to_numpy()+residual.prediction.to_numpy(),direct.direct)
            if r['gate_training_id_hash']!=object_hash(direct.loc[eligible,ID].tolist()):
                raise ValueError('Learned gate fit row hash mismatch')
        if r['seed'] != 20260910 and r['candidate']=='rome_schedule_mixture':
            seed_protocol=read_json(OUT/f'mixture_seed_protocol_{r["seed"]}.json')
            if seed_protocol['config'] != config:
                raise ValueError('Seed wrapper config mismatch')
        receipts.append({'run':file.parent.name,'manifest_sha256':sha256(file),'split_hash':split['split_hash']})
    for file in sorted((OUT/'clock_cpu_experts').glob('*/expert.json')):
        r=read_json(file)
        _,_,idx,split=reference[file.parent.name]
        status=proxy_status(meta.proxy_sec,r['config'])
        if not same_split(r['split'],split):
            raise ValueError('Clock expert split mismatch')
        for stage in ['fit','tune','refit']:
            pos=idx[stage][status[idx[stage]]=='present']
            if r[f'{stage}_id_hash']!=object_hash(x.index[pos].tolist()):
                raise ValueError('Clock expert training cohort mismatch')
        frame=pd.read_parquet(file.parent/'tune_direct.parquet')
        pos=idx['tune'][status[idx['tune']]=='present']
        if not np.array_equal(frame[ID],x.index[pos]) or not np.array_equal(frame[TARGET],labels.iloc[pos][TARGET]):
            raise ValueError('Clock calibration labels or rows changed')
    write_json(OUT/'split_verification.json',{'status':'verified','count':len(receipts),
        'runs':receipts,'clock_experts':len(list((OUT/'clock_cpu_experts').glob('*/expert.json'))),
        'verifier_sha256':sha256(__file__)})
    print(f'VERIFIED {len(receipts)} model split receipts and all clock calibration cohorts',flush=True)


if __name__=='__main__':
    verify()
