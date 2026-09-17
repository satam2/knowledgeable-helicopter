"""Prepare bounded128-query fixtures in a process separate from XGB loading."""
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('combined_family_frozen',Path(__file__).with_name('audit.py'))
frozen=importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)
import argparse
import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fold',choices=['F1','F3'],required=True)
    args=parser.parse_args()
    folder=frozen.BASE/'combined_xgb'/f'xgb_aobt_allfinite_{args.fold}_s20260916'
    rec=frozen.checked(folder)
    output=frozen.OUT/'xgb'/(args.fold+'_fixture')
    assert not output.exists()
    reference,_=frozen.common.reference(args.fold)
    frame=pd.read_parquet(folder/'candidate.parquet')
    np.testing.assert_array_equal(frame[frozen.ID],reference[frozen.ID])
    np.testing.assert_array_equal(frame[frozen.TARGET],reference[frozen.TARGET])
    positions=np.flatnonzero(np.isfinite(reference.proxy_sec.to_numpy(float)))
    positions=positions[np.linspace(0,len(positions)-1,128,dtype=int)]
    query=reference.iloc[positions]
    features,sources=frozen.query_features(pd.Index(query[frozen.ID]),query[frozen.TIME],rec['feature_columns'])
    output.mkdir()
    features.to_parquet(output/'features.parquet')
    frame.iloc[positions][[frozen.ID,frozen.TARGET,'proxy_sec','prediction_sec']].to_parquet(output/'expected.parquet',index=False)
    frozen.common.write_json(output/'manifest.json',dict(status='complete',source_sha256=frozen.common.sha256(__file__),
        helper_sha256=frozen.common.sha256(Path(__file__).with_name('audit.py')),model_folder=str(folder),
        model_manifest_sha256=frozen.common.sha256(folder/'manifest.json'),feature_sources=sources,
        query_ID_hash=frozen.common.object_hash(query[frozen.ID].tolist()),
        outputs={p.name:frozen.common.sha256(p) for p in output.glob('*.parquet')},peak_rss_bytes=frozen.guard()))
    print('FIXTURE',args.fold,frozen.guard(),flush=True)


if __name__=='__main__':
    main()
