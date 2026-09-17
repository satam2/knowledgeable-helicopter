"""Full-data GPU histogram XGBoost with native categories and RMSE stopping."""
import time
import xgboost as xgb
import numpy as np
from encoders import FrameEncoder,json_safe


def fit(x,y,tuning=None,*,steps=None,seed=20260916,threads=4):
    y=np.asarray(y,float)
    if not len(y) or len(y)!=len(x) or not np.isfinite(y).all():
        raise ValueError("Training requires aligned nonempty finite targets")
    if steps is not None and (steps<1 or tuning is not None):
        raise ValueError("Refit must use selected steps without tuning")
    started=time.monotonic()
    encoder=FrameEncoder().fit(x)
    train=encoder.transform(x)
    estimator=xgb.XGBRegressor(n_estimators=2500 if steps is None else int(steps),
        learning_rate=.04,max_depth=8,max_bin=256,objective="reg:squarederror",
        eval_metric="rmse",tree_method="hist",device="cuda:0",enable_categorical=True,
        max_cat_to_onehot=4,max_cat_threshold=64,min_child_weight=20,reg_lambda=10.,
        subsample=1.,colsample_bytree=1.,n_jobs=threads,random_state=seed,
        early_stopping_rounds=100 if tuning is not None else None)
    options={}
    if tuning is not None:
        tx,ty=tuning
        options["eval_set"]=[(encoder.transform(tx),np.asarray(ty,float))]
    estimator.fit(train,y,verbose=100,**options)
    selected=int(estimator.best_iteration+1) if tuning is not None else int(steps)
    return {"estimator":estimator,"encoder":encoder,"steps":selected}, {
        "steps":selected,"fit_runtime_sec":time.monotonic()-started,"rows":len(x),"features":len(x.columns),
        "version":xgb.__version__,"device":"cuda:0","params":json_safe(estimator.get_params()),
        "history":estimator.evals_result() if tuning is not None else {},
        "encoding":"Native categorical fit-only vocab missing0 unknown1; numeric nonfinite/-999999 treated missing; all original labels retained."}


def predict(model,x):
    return np.asarray(model["estimator"].predict(model["encoder"].transform(x),iteration_range=(0,model["steps"])),float)
