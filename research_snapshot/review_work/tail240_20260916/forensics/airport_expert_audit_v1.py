"""Read-only model metadata inventory and original finite-cohort airport counts."""
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'review_work/campaign_20260916'))
import common
OUT=ROOT/'private_runs/tail240_20260916/forensics/airport_expert_audit_v1'


def main():
    assert not OUT.exists()
    inventory=[]
    checked=0
    for base in ['breakthrough_20260916','tail240_20260916']:
        for path in (ROOT/'private_runs'/base).rglob('manifest.json'):
            try:
                m=json.loads(path.read_text(encoding='utf-8'))
            except (ValueError,OSError):
                continue
            checked+=1
            columns=next((m[k] for k in ['feature_columns','features_used','columns'] if isinstance(m.get(k),list)),[])
            if len(columns)>=387 or m.get('per_airport') is True:
                inventory.append(dict(path=str(path.relative_to(ROOT)),sha256=common.sha256(path),status=m.get('status'),
                    feature_count=len(columns),per_airport=m.get('per_airport'),family=m.get('family'),
                    model_paths=[k for k in m.get('outputs',{}) if str(k).endswith(('.cbm','.joblib','.txt','.json')) and 'model' in str(k)]))
    path=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert common.sha256(path)==common.read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][path.name]
    meta=pd.read_parquet(path,columns=[common.ID,'FLIGHT_ID_mvt',common.MOVEMENT,'ADEP_mvt','proxy_sec'])
    meta[common.TARGET]=np.nan
    counts={}
    for fold in ['F1','F3']:
        idx,split,_=common.fold_data(meta,fold,full=True)
        groups={}
        for stage in ['fit','tune']:
            x=meta.iloc[idx[stage]]
            x=x.loc[np.isfinite(x.proxy_sec)]
            groups[stage]={str(a):dict(allfinite=len(g),ordinary=int(g.proxy_sec.between(0,7200).sum())) for a,g in x.groupby('ADEP_mvt')}
        counts[fold]=groups
    OUT.mkdir()
    common.write_json(OUT/'receipt.json',dict(status='complete',source_sha256=common.sha256(__file__),
        metadata_sha256=common.sha256(path),manifest_files_read=checked,relevant_inventory=inventory,counts=counts,
        label_columns_read=False,scope='Source review plus named manifests under breakthrough/tail240; absence of per_airport key alone is not proof against other implementations.'))
    print('MANIFESTS',checked,'RELEVANT',len(inventory),'COUNTS',counts,flush=True)


if __name__=='__main__':main()
