"""Independent source-to-frame reconstruction for final-model verification."""
from preflight import ROOT, ID, read, sha
import gc
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil

BASE_CATS={'ADEP_mvt','RUNWAY_mvt','STAND_mvt','ADES_mvt','AIRCRAFT_TYPE_mvt','AIRCRAFT_OPERATOR_flt',
           'WK_TBL_CAT_flt','MARKET_SEGMENT_flt','FLIGHT_TYPE_flt','airport_stand','airport_runway'}


def guard():
    info=psutil.Process().memory_info()
    assert info.rss < 3*1024**3
    assert psutil.virtual_memory().available >= 8*1024**3
    return int(info.peak_wset)


def reconstruct(columns,fit_ids,rank_ids):
    sys.path.insert(0,str(ROOT/'review_work/tail240_20260916/state/final_ple387'))
    import sources
    tuples=sources.feature_sources(columns)
    fit_ids,rank_ids=pd.Index(fit_ids),pd.Index(rank_ids)
    assert fit_ids.is_unique and rank_ids.is_unique and not fit_ids.isin(rank_ids).any()
    categories=set()
    for path,digest,fill,names in tuples:
        assert sha(path)==digest
        schema=pq.read_schema(path)
        for name in names:
            dtype=schema.field(name).type
            if pa.types.is_dictionary(dtype):dtype=dtype.value_type
            if not (pa.types.is_integer(dtype) or pa.types.is_floating(dtype) or pa.types.is_boolean(dtype)):
                categories.add(name)
    counts={name:np.zeros(len(rank_ids),np.uint8) for name in columns}
    numeric={name:np.full(len(rank_ids),np.nan,dtype='float32') for name in columns if name not in categories}
    strings={name:np.full(len(rank_ids),None,dtype=object) for name in categories}
    vocab={name:set() for name in categories}
    receipts=[]
    for path,digest,fill,names in tuples:
        source_rows=0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8192,columns=[ID,*names],use_threads=False):
            frame=batch.to_pandas()
            rp=rank_ids.get_indexer(frame[ID]);fp=fit_ids.get_indexer(frame[ID])
            fitmask=fp>=0;rankmask=rp>=0;position=rp[rankmask]
            source_rows+=int(rankmask.sum())
            assert len(np.unique(position))==len(position)
            for name in names:
                if name in categories:
                    vocab[name].update(frame.loc[fitmask,name].dropna().astype(str))
                    if name not in BASE_CATS and frame.loc[fitmask,name].isna().any():
                        vocab[name].add('MISSING')
                assert not counts[name][position].any(),(path,name)
                counts[name][position]=1
                if name in categories:
                    strings[name][position]=frame.loc[rankmask,name].astype('string').to_numpy(na_value=None)
                else:
                    value=pd.to_numeric(frame.loc[rankmask,name]).to_numpy(dtype='float32',na_value=np.nan)
                    value[~np.isfinite(value)]=np.nan
                    if fill:value[np.isnan(value)]=-999999
                    numeric[name][position]=value
        receipts.append(dict(path=str(path),sha256=digest,fill=fill,columns=names,ranking_rows=source_rows))
        guard()
    assert all(value.all() for value in counts.values())
    del counts
    values={}
    for name in columns:
        if name in categories:
            value=pd.Series(strings.pop(name),dtype='string')
            value=value.where(value.isna()|value.isin(vocab[name]),'__UNSEEN_CONTEXT_VALUE__')
            if name not in BASE_CATS:value=value.fillna('MISSING')
            values[name]=pd.Categorical(value)
        else:values[name]=numeric.pop(name)
    frame=pd.DataFrame(values,index=rank_ids,copy=False)
    gc.collect();guard()
    return frame,{name:sorted(known) for name,known in vocab.items()},receipts
