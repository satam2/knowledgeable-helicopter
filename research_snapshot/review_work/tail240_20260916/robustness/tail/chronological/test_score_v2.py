"""Exercise the actual scorer with index gaps while preserving strict ID/value checks."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import score_v2 as scorer


@pytest.mark.parametrize('mutation',[None,'id','value'])
def test_gapped_source_indices_and_real_mismatches(tmp_path,monkeypatch,mutation):
    run=scorer.run
    frame=pd.DataFrame({run.ID:np.arange(100.,112.),run.TIME:pd.date_range('2025-07-01',periods=12,tz='UTC'),
        'FLIGHT_ID_mvt':np.arange(200.,212.),'ADEP_mvt':['A']*12,'proxy_sec':np.nan})
    expected=frame.copy()
    expected.index=np.arange(0,24,2)
    if mutation=='id':
        expected.loc[0,run.ID]=999.
    elif mutation=='value':
        expected.loc[0,'ADEP_mvt']='B'
    arms=['et_leaf1','et_leaf20','normalized_diagnostic']
    release=dict(status='released',protocol_sha256='digest',independent_receipts={
        'F1/'+arm:dict(path=str(tmp_path/(arm+'_review.json')),sha256='digest') for arm in arms})
    marker=dict(status='complete_predictions_frozen',protocol_sha256='digest',
        predictions={'evaluation_2025-07':{'sha256':'digest'}})
    def read(path):
        if Path(path).name=='independent_evaluation_release.json':
            return release
        if Path(path).name.endswith('_review.json'):
            return {'expert_manifest_sha256':'digest'}
        if Path(path).name=='manifest.json':
            return marker
        raise AssertionError(path)
    writes={}
    label_reads=[]
    monkeypatch.setattr(run,'OUT',tmp_path)
    monkeypatch.setattr(run,'PANELS',{'F1':['2025-07']})
    monkeypatch.setattr(run,'ARMS',{a:None for a in arms})
    monkeypatch.setattr(run,'declare',lambda:None)
    monkeypatch.setattr(run,'read_stage',lambda fold,stage:expected)
    monkeypatch.setattr(run.common,'read_json',read)
    monkeypatch.setattr(run.common,'sha256',lambda path:'digest')
    monkeypatch.setattr(run.common,'write_json',lambda path,obj:writes.__setitem__(Path(path).name,obj))
    monkeypatch.setattr(scorer.pd,'read_parquet',lambda path:frame.assign(prediction_sec=np.arange(12.)))
    def labels(value):
        label_reads.append(value.copy())
        return np.arange(12.)
    monkeypatch.setattr(run.maintrain,'read_labels',labels)
    if mutation:
        with pytest.raises(AssertionError):
            scorer.main()
        assert not label_reads
    else:
        scorer.main()
        assert len(label_reads)==1
        assert writes['summary.json']['status']=='complete'
        assert writes['summary.json']['folds']['F1']['2025-07']['rows']==12
