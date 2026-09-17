"""Fresh-process native LightGBM diagnostic with selectable original import order."""
import sys
if '--original-order' in sys.argv:
    import numpy
    import pandas
    import xgboost
    import lightgbm
else:
    import lightgbm
    import numpy
    import pandas
    import xgboost
import argparse
import traceback
import time
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'review_work/campaign_20260916'))
sys.path.insert(0,str(ROOT/'review_work/breakthrough_20260916/deeper_lgb'))
import adapter
import common
from taxiout.artifacts import read_json,write_json,sha256
from taxiout.schema import TARGET
OUT=ROOT/'private_runs/breakthrough_20260916/models/sequence_result_audit/native_canary'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--original-order',action='store_true')
    parser.add_argument('--repeat',type=int,default=1)
    args=parser.parse_args()
    name=('original' if args.original_order else 'lightgbm_first')+f'_{args.repeat}'
    path=OUT/f'{name}.json'
    if path.exists():
        raise ValueError('Preserve prior probe')
    fixture=read_json(OUT/'fixture.json')
    assert sha256(OUT/'features.parquet')==fixture['feature_sha256']
    x=pd.read_parquet(OUT/'features.parquet')
    meta=pd.read_parquet(OUT/'labels.parquet')
    y=(meta[TARGET]-meta.proxy_sec).to_numpy(float)
    record={'status':'running','name':name,'fixture_sha256':sha256(OUT/'fixture.json'),'source_sha256':sha256(__file__),
            'adapter_sha256':sha256(adapter.__file__),'rows':len(x),'features':len(x.columns),'threads':2}
    write_json(path,record)
    began=time.monotonic()
    try:
        model,evidence=adapter.fit(x,y,steps=10,threads=2,seed=20260916)
        values=adapter.predict(model,x.iloc[:128])
        assert np.isfinite(values).all()
        record.update(status='passed',fit=evidence,predictions=values.tolist())
    except Exception as error:
        record.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        record.update(runtime_sec=time.monotonic()-began,peak_rss_bytes=getattr(psutil.Process().memory_info(),'peak_wset',psutil.Process().memory_info().rss))
        write_json(path,record)
    print('NATIVE_PROBE',name,record['status'],record['runtime_sec'],flush=True)


if __name__=='__main__':
    with threadpool_limits(2):
        main()
