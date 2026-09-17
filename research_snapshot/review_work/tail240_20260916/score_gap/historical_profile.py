"""Descriptive 2025 label regimes, never a final-model holdout evaluation."""
import lightgbm
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'review_work/campaign_20260916'))
import common
OUT=common.external_path(ROOT/'private_runs/tail240_20260916/score_gap/historical_profile')


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    source=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert common.sha256(source)==common.read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][source.name]
    data=pd.read_parquet(source,columns=[common.MOVEMENT,'ADEP_mvt','proxy_sec',common.TARGET])
    assert len(data)==2085047 and data[common.MOVEMENT].dt.year.eq(2025).all()
    data['month']=data[common.MOVEMENT].dt.month
    missing=~np.isfinite(data.proxy_sec)
    records=[]
    for month,part in data.loc[missing&data.ADEP_mvt.eq('LIRF')].groupby('month'):
        values=part[common.TARGET].to_numpy(float)
        records.append(dict(month=int(month),rows=len(values),mean=float(values.mean()),median=float(np.median(values)),
                            std_sample=float(values.std(ddof=1)),maximum=float(values.max())))
    finite=[]
    for month,part in data.loc[~missing].groupby('month'):
        gap=(part[common.TARGET]-part.proxy_sec).to_numpy(float)
        finite.append(dict(month=int(month),rows=len(gap),source_disagreement_rms=float(np.sqrt(np.mean(gap**2))),
                           abs_source_disagreement_gt1800_share=float(np.mean(np.abs(gap)>1800))))
    result=dict(status='complete',source_sha256=common.sha256(__file__),metadata_sha256=common.sha256(source),
                rome_missing=records,finite_source_disagreement=finite,
                interpretation='Historical label/source-regime instability only. Every row is2025; no ranking labels. These are not model predictions, out-of-sample errors, or estimates of2026 outcomes.')
    common.write_json(OUT/'receipt.json',result)
    print('Historical profile complete',len(records),'months')


if __name__=='__main__':main()
