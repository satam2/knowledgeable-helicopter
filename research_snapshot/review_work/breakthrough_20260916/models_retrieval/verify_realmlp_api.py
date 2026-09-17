"""Authorized CPU-only synthetic validation of installed RealMLP fit/refit API."""
import torch
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

import realmlp_adapter as adapter
import run_pilot as core
from pytabkit.models.alg_interfaces.nn_interfaces import NNAlgInterface
from pytabkit.models.data.splits import RandomSplitter
from pytabkit.models.training.lightning_callbacks import StopAtEpochsCallback


def main():
    out=core.OUT/"canaries/realmlp_cpu_s20260917/api_refit_schedule_verification.json"
    if out.exists():raise ValueError("Preserve verification")
    rng=np.random.default_rng(20260917)
    x=pd.DataFrame({"value":rng.normal(size=320),"airport":pd.Categorical(np.where(np.arange(320)%2,"A","B"))})
    y=100+3*x.value.to_numpy()+rng.normal(size=320)
    audits=[]
    original=NNAlgInterface.fit
    stop_callback=StopAtEpochsCallback.on_train_epoch_end
    refit_epochs=[]
    def audit_epoch(self,trainer,pl_module):
        refit_epochs.append(trainer.current_epoch+1)
        return stop_callback(self,trainer,pl_module)
    def audit_fit(self,*args,**kwargs):
        splits=kwargs.get("idxs_list",args[1] if len(args)>1 else None)
        for split in splits:
            audits.append({"train_indices":split.train_idxs.tolist(),"validation_indices":None if split.val_idxs is None else split.val_idxs.tolist()})
        return original(self,*args,**kwargs)
    with patch.object(RandomSplitter,"get_idxs",side_effect=AssertionError("Forbidden random split")),patch.object(NNAlgInterface,"fit",audit_fit),patch.object(StopAtEpochsCallback,"on_train_epoch_end",audit_epoch):
        fitted,evidence=adapter.fit(x.iloc[:256],y[:256],(x.iloc[256:288],y[256:288]),device="cpu",threads=2,seed=20260917,max_epochs=2)
        steps=evidence["steps"]
        model,refit=adapter.fit(x.iloc[:288],y[:288],steps=steps,device="cpu",threads=2,seed=20260917,max_epochs=64)
        prediction=adapter.predict(model,x.iloc[288:])
    assert audits[0]["train_indices"]==[list(range(256))]
    assert audits[0]["validation_indices"]==[list(range(256,288))]
    assert audits[1]["train_indices"]==[list(range(288))]
    assert audits[1]["validation_indices"] is None
    assert len(prediction)==32 and np.isfinite(prediction).all()
    assert refit_epochs==list(range(1,steps+1))
    report={"status":"passed","source_hashes":core.hashes(),"verification_source_sha256":core.sha256(__file__),"device":"cpu","threads":2,"private_data_loaded":False,
        "fit_selected_epoch":steps,"fit_params":evidence["fit_params"],"refit_params":refit["fit_params"],"train_validation_index_audits":audits,
        "random_split_forbidden_and_not_called":True,"prediction_rows":len(prediction),"refit_schedule_epochs":refit["params"]["n_epochs"],"refit_stop_epoch":refit["params"]["stop_epoch"],"actual_refit_epochs":refit_epochs,
        "refit_fit_params_caveat":"Library reports unused best-validation epoch0 when no validation exists; actual callback epochs and explicit stop_epoch are authoritative for refit"}
    core.write_json(out,core.numeric_json(report))
    print("ACTUAL_API_PASSED",steps,evidence["fit_params"],refit["fit_params"],flush=True)


if __name__=="__main__":main()
