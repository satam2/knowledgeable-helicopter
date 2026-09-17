"""Fit-only missing-value and category handling shared by bounded model pilots."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class FramePreprocessor:
    def fit(self,x):
        self.columns=list(x)
        self.numeric=[c for c in x if pd.api.types.is_numeric_dtype(x[c])]
        self.categorical=[c for c in x if c not in self.numeric]
        a=self.numbers(x)
        self.medians=np.array([np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else 0. for v in a.T],dtype=np.float32)
        self.vocabulary={c:sorted(x[c].astype("string").dropna().unique().tolist()) for c in self.categorical}
        return self

    def numbers(self,x):
        a=x[self.numeric].to_numpy(dtype=np.float32,na_value=np.nan).copy()
        a[(a==-999999)|~np.isfinite(a)]=np.nan
        return a

    def transform(self,x):
        if list(x)!=self.columns:raise ValueError("Changed feature schema/order")
        a=self.numbers(x);missing=~np.isfinite(a)
        data={c:np.where(missing[:,i],self.medians[i],a[:,i]).astype(np.float32) for i,c in enumerate(self.numeric)}
        for i,c in enumerate(self.numeric):data["missing__"+c]=missing[:,i].astype(np.float32)
        for c in self.categorical:
            values=x[c].astype("string")
            known=values.isin(self.vocabulary[c])
            mapped=("value:"+values).where(known,"unknown:").where(values.notna(),"missing:")
            data[c]=pd.Categorical(mapped,categories=["missing:","unknown:",*["value:"+v for v in self.vocabulary[c]]])
        return pd.DataFrame(data,index=x.index).reset_index(drop=True)


class RetrievalPreprocessor:
    def fit(self,x):
        self.frame=FramePreprocessor().fit(x)
        values=self.frame.transform(x)
        self.numerical=[c for c in values if c not in self.frame.categorical]
        self.onehot=None
        if self.frame.categorical:
            self.onehot=OneHotEncoder(handle_unknown="infrequent_if_exist",max_categories=16,sparse_output=False,dtype=np.float32)
            self.onehot.fit(values[self.frame.categorical].astype("string"))
        return self

    def transform(self,x):
        values=self.frame.transform(x)
        numeric=values[self.numerical].to_numpy(np.float32)
        categorical=self.onehot.transform(values[self.frame.categorical].astype("string")) if self.onehot is not None else np.empty((len(x),0),np.float32)
        result=np.concatenate([numeric,categorical],axis=1)
        if not np.isfinite(result).all():raise ValueError("Nonfinite encoded values")
        return np.ascontiguousarray(result,dtype=np.float32)
