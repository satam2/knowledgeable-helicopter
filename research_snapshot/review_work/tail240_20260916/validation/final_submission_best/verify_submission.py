"""Final independent composition and serialization gate; never uploads."""
from preflight import ROOT, OUT, RAW, V2, ID, TARGET, read, sha
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RELEASE=ROOT/'private_runs/tail240_20260916/final_submission_v3_v2'


def main():
    target=RELEASE/'independent_verification.json'
    assert not target.exists(),'Preserve completed verification'
    ready,protocol=read(RELEASE/'submission_ready.json'),read(RELEASE/'protocol.json')
    assert ready['protocol_sha256']==sha(RELEASE/'protocol.json')
    assert protocol['source_sha256']==sha(ROOT/'review_work/tail240_20260916/final_submission/compose.py')
    assert protocol['preserved_v1_source_sha256']==sha(ROOT/'review_work/tail240_20260916/final_submission/compose_v1_frozen.py')
    inherited=read(OUT/'preflight_receipt.json');assert inherited['status']=='passed'
    assert sha(RAW/'submitting.parquet')==inherited['template']['sha256']
    meta=pd.read_parquet(V2/'ranking_meta.parquet')
    assert sha(V2/'ranking_meta.parquet')==read(V2/'ranking_inputs.json')['files']['ranking_meta.parquet']
    ids=meta[ID];proxy=meta.proxy_sec.to_numpy(float)
    finite=np.isfinite(proxy);ordinary=finite&(proxy>=0)&(proxy<=7200);unusual=finite&~ordinary
    assert len(ids)==344841 and int((~finite).sum())==5290 and ids.is_unique
    weightroot=ROOT/'private_runs/breakthrough_20260916/models/context_gate/final_simplex9_v1'
    folds=[]
    for fold in ('F1','F3'):
        path=weightroot/f'{fold}_weights.json';assert sha(path)==protocol['weight_sources'][fold]
        folds.append(read(path))
    assert folds[0]['experts']==folds[1]['experts']==protocol['experts']
    weights=(192122*np.array(folds[0]['global'])+152719*np.array(folds[1]['global']))/344841
    weights[weights<1e-12]=0;weights/=weights.sum()
    np.testing.assert_array_equal(weights,protocol['weights'])
    names=dict(tabm_combined='tabm225',tabm_ple8='ple387',lgb63_sequence8='lgb449',lgb63_union='lgb387',catboost_combined='cat225')
    expected=np.empty(len(ids));mixture=np.zeros(int(finite.sum()));bound={}
    for expert,short in names.items():
        folder=Path(protocol['component_directories']['ple']) if short=='ple387' else Path(protocol['component_directories']['ordinary'])/short
        values,binding=read_component(folder,'ranking_predictions.parquet',ids[finite],OUT/f'{short}_native_receipt.json')
        assert ready['components'][expert]['manifest_sha256']==binding['manifest_sha256']
        assert ready['components'][expert]['predictions_sha256']==binding['prediction_sha256']
        coefficient=weights[protocol['experts'].index(expert)]
        assert coefficient==protocol['active_weights'][expert] and coefficient>0
        mixture+=coefficient*values
        bound[short]=binding
    expected[ordinary]=mixture[ordinary[finite]]
    folder=Path(protocol['component_directories']['ordinary'])/'lgb225'
    values,bound['lgb225']=read_component(folder,'ranking_predictions.parquet',ids[finite],OUT/'lgb225_native_receipt.json')
    expected[unusual]=values[unusual[finite]]
    folder=Path(protocol['component_directories']['missing'])
    values,bound['missing']=read_component(folder,'missing_ranking.parquet',ids[~finite],OUT/'missing_native_receipt.json')
    expected[~finite]=values
    assert np.isfinite(expected).all()
    predictions=pd.read_parquet(RELEASE/'ranking_predictions.parquet')
    assert sha(RELEASE/'ranking_predictions.parquet')==ready['prediction_sha256']
    np.testing.assert_array_equal(predictions[ID],ids)
    np.testing.assert_array_equal(predictions.prediction_sec,expected)
    route=np.empty(len(ids),object);route[ordinary]='ordinary_ensemble';route[unusual]='finite_nonordinary_lgb225';route[~finite]='missing_blend'
    np.testing.assert_array_equal(predictions.route,route)
    counts=dict(ordinary=int(ordinary.sum()),finite_nonordinary=int(unusual.sum()),missing=int((~finite).sum()))
    assert counts==ready['route_counts']
    template=pq.read_table(RAW/'submitting.parquet')
    path=RELEASE/protocol['submission']['key'];saved=pq.read_table(path)
    assert saved.schema.remove_metadata()==template.schema.remove_metadata()
    assert saved.schema.names==[ID,TARGET] and saved.schema.field(ID).type==pa.float64() and saved.schema.field(TARGET).type==pa.int32()
    final,ref=saved.to_pandas(),template.to_pandas()
    assert len(final)==344841 and final[ID].is_unique and not final[ID].isna().any()
    np.testing.assert_array_equal(final[ID],ref[ID])
    assert set(ref[ID])==set(ids)
    ordered=pd.Series(expected,index=ids).loc[ref[ID]].to_numpy()
    rounded=np.rint(ordered);limits=np.iinfo(np.int32)
    assert np.isfinite(rounded).all() and ((rounded>=limits.min)&(rounded<=limits.max)).all()
    np.testing.assert_array_equal(final[TARGET],rounded.astype('int32'))
    digest=sha(path);assert digest==ready['submission']['sha256']
    # Recheck raw/template and model bindings after arithmetic to catch concurrent mutation.
    for key,record in bound.items():
        folder=Path(record['folder']);assert sha(folder/'manifest.json')==record['manifest_sha256']
        assert sha(folder/record['prediction_file'])==record['prediction_sha256']
    result=dict(status='passed',submission_sha256=digest,submission_file=str(path),rows=len(final),
        schema=str(saved.schema.remove_metadata()),template_sha256=sha(RAW/'submitting.parquet'),
        exact_template_order=True,exact_native_component_composition=True,nearest_even_int32_serialization_exact=True,
        route_counts=counts,protocol_sha256=sha(RELEASE/'protocol.json'),ready_sha256=sha(RELEASE/'submission_ready.json'),
        verifier_sha256=sha(__file__),components=bound,uploaded=False,official_score=None,
        limitations='Development268.662991 is not official performance. Current identity/quota/unused key and remote hash/organizer acceptance are uploader responsibilities.')
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='components'},indent=2),flush=True)


def read_component(folder,filename,ids,receipt_path):
    receipt=read(receipt_path);assert receipt['status']=='passed'
    marker=read(folder/'manifest.json');assert marker['status']=='complete'
    assert sha(folder/'manifest.json')==receipt['manifest_sha256']
    digest=sha(folder/filename)
    assert marker['outputs'][filename]==digest==receipt['prediction_sha256']
    frame=pd.read_parquet(folder/filename)
    assert len(frame)==len(ids) and frame[ID].is_unique and set(frame[ID])==set(ids)
    values=frame.set_index(ID).loc[ids,'prediction_sec'].to_numpy(float)
    assert np.isfinite(values).all()
    return values,dict(folder=str(folder),prediction_file=filename,manifest_sha256=sha(folder/'manifest.json'),prediction_sha256=digest,
                       independent_receipt=str(receipt_path),independent_receipt_sha256=sha(receipt_path))


if __name__=='__main__':main()
