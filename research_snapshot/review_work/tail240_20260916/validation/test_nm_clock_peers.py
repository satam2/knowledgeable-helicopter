"""Rowwise independent peer-window oracle for the frozen cache constructor."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'review_work/tail240_20260916/forensics/nm_clock_peers'))
import build_v1 as b


class OracleTest(unittest.TestCase):
    def test_all_statistics_against_raw_rowwise_membership(self):
        rng=np.random.default_rng(123)
        n=161
        raw=pd.DataFrame({b.ID:np.arange(n,dtype=float),b.FID:rng.integers(0,40,n).astype(float),b.PHASE:'DEP','ADEP_mvt':rng.choice(['A','B',''],n),'STAND_mvt':rng.choice(['S1','S2',''],n)})
        anchors=pd.Timestamp('2025-01-31T23:30:00Z')+pd.to_timedelta(rng.integers(-4000,4000,n),unit='s')
        raw[b.TIME]=anchors+pd.to_timedelta(rng.integers(-100,10000,n),unit='s')
        for clock in b.CLOCKS:raw[clock]=anchors-pd.to_timedelta(rng.integers(-20,20,n),unit='s')
        raw.loc[raw.index%5==0,b.FID]=np.nan
        raw.loc[raw.index%7==0,b.CLOCKS[0]]=pd.NaT
        raw.loc[raw.index%11==0,b.CLOCKS[1]]=pd.NaT
        # Duplicate events with a different stand; smallest movement ID wins globally.
        dup=raw.iloc[3].copy();dup[b.ID]=n;dup.STAND_mvt='DIFFERENT'
        raw=pd.concat([raw,pd.DataFrame([dup])],ignore_index=True)
        got=b.build_frame(b.prepare(raw)).set_index(b.ID)
        events=raw.sort_values(b.ID).copy()
        flight=events[b.FID].astype('string').where(events[b.FID].notna(),'missing:'+events[b.ID].astype(str))
        events['_key']=flight
        events=events.drop_duplicates(['ADEP_mvt','_key',b.TIME])
        for _,query in raw.iterrows():
            for scope in ['airport','stand']:
                for width in [15,60]:
                    p=events.loc[events[b.CLOCKS[0]].notna()&events.ADEP_mvt.eq(query.ADEP_mvt)&events[b.TIME].dt.strftime('%Y-%m').eq(query[b.TIME].strftime('%Y-%m'))]
                    if scope=='stand':p=p.loc[p.STAND_mvt.eq(query.STAND_mvt)]
                    if pd.isna(query[b.CLOCKS[0]]) or query.ADEP_mvt=='' or (scope=='stand' and query.STAND_mvt==''):
                        p=p.iloc[:0]
                    else:
                        p=p.loc[(p[b.CLOCKS[0]]-query[b.CLOCKS[0]]).abs().le(pd.Timedelta(minutes=width))]
                    p=p.loc[p[b.ID].ne(query[b.ID])]
                    if pd.notna(query[b.FID]):p=p.loc[p[b.FID].ne(query[b.FID])]
                    offsets=np.column_stack([(p[b.TIME]-p[c]).dt.total_seconds().to_numpy() for c in b.CLOCKS])
                    means=[float(a[np.isfinite(a)].mean()) if np.isfinite(a).any() else np.nan for a in offsets.T]
                    expected=[len(p),*means,float(np.std(offsets[:,0])) if len(p) else np.nan,float(p[b.CLOCKS[0]].eq(query[b.CLOCKS[0]]).mean()) if len(p) else np.nan,*np.isfinite(offsets[:,1:]).sum(axis=0)]
                    cols=[f'nmpeer_{scope}_{width}m_{name}' for name in b.STATS]
                    np.testing.assert_allclose(got.loc[query[b.ID],cols].to_numpy(float),expected,rtol=2e-6,atol=.002,equal_nan=True)


if __name__=='__main__':unittest.main()
