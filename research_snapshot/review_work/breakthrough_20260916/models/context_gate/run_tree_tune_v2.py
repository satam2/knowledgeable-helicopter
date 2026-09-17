"""Memory-only loader repair; frozen treegate/model/chronological design."""
import lightgbm
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import run_tree_tune as frozen

original_reader=pd.read_parquet
CONVENTIONS=frozen.ROOT/'private_runs/breakthrough_20260916/missing/source_conventions/features.parquet'


def bounded_reader(path,*args,**kwargs):
    if Path(path).resolve()!=CONVENTIONS.resolve():
        return original_reader(path,*args,**kwargs)
    filters=kwargs['filters']
    assert len(filters)==1 and filters[0][:2]==(frozen.ID,'in')
    expected=pd.Index(filters[0][2])
    source=pq.ParquetFile(path)
    parts=[]
    for i in range(source.num_row_groups):
        ids=source.read_row_group(i,columns=[frozen.ID],use_threads=False).column(0)
        choose=pc.is_in(ids,value_set=pa.array(expected.to_numpy(),type=ids.type))
        if pc.any(choose).as_py():
            part=source.read_row_group(i,columns=kwargs['columns'],use_threads=False).filter(choose)
            parts.append(part.to_pandas())
            del part
    result=pd.concat(parts,ignore_index=True)
    assert len(result)==len(expected) and result[frozen.ID].is_unique
    return result


if __name__=='__main__':
    previous=frozen.OUT
    frozen.OUT=previous.with_name('tree_early60_late40_v2')
    receipt={'source_sha256':frozen.sha256(__file__),'frozen_runner_sha256':frozen.sha256(frozen.__file__),
             'original_attempt_protocol_sha256':frozen.sha256(previous/'protocol.json'),
             'repair':'Read IDcolumn perrowgroup, decodeonlymatchinggroups; no model/objective/splitparameter changes',
             'prior_attempt':'Stoppedbeforefit on1GiBbudgetcheck;no modelorlate metrics produced'}
    frozen.write_json(frozen.OUT.with_name(frozen.OUT.name+'_loader_receipt.json'),receipt)
    pd.read_parquet=bounded_reader
    frozen.main()
