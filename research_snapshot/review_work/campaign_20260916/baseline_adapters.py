"""Matched CatBoost and a sparse Ridge conditional-mean comparator."""
import numpy as np
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def fit_catboost(x,y,tuning=None,*,steps=None,seed=20260916,threads=4):
    model=CatBoostRegressor(iterations=steps or 600,depth=6,learning_rate=.05,
        l2_leaf_reg=10,loss_function='RMSE',random_seed=seed,thread_count=threads,
        task_type='CPU',allow_writing_files=False,verbose=200)
    kw={'cat_features':list(x.select_dtypes('category').columns)}
    if tuning is not None:
        kw.update(eval_set=tuning,early_stopping_rounds=60,use_best_model=True)
    model.fit(x,y,**kw)
    return model,{'steps':model.tree_count_,'task_type':'CPU'}


def fit_ridge(x,y,tuning=None,*,steps=None,seed=20260916,threads=4):
    cats=list(x.select_dtypes('category').columns)
    nums=[c for c in x if c not in cats]
    preprocess=ColumnTransformer([
        ('cat',OneHotEncoder(handle_unknown='ignore',min_frequency=10),cats),
        ('num',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),nums)
    ],sparse_threshold=1.0)
    model=make_pipeline(preprocess,Ridge(alpha=1000,solver='lsqr',tol=1e-5,max_iter=500))
    model.fit(x,y)
    return model,{'steps':1,'alpha':1000,'solver':'lsqr','max_iter':500,
                  'encoding':'train-only sparse onehot + scaled numeric; fixed regularization'}


def predict(model,x):
    return np.asarray(model.predict(x)).reshape(-1)
