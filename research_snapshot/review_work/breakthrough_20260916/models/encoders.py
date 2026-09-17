"""Training-only categorical identities and neural numeric transforms."""
import numpy as np
import pandas as pd


class FrameEncoder:
    def fit(self, x, neural=False):
        self.columns = list(x)
        self.categories = {}
        self.numeric = []
        self.neural = neural
        for column in self.columns:
            if isinstance(x[column].dtype, pd.CategoricalDtype) or not pd.api.types.is_numeric_dtype(x[column]):
                values = x[column].astype("string")
                self.categories[column] = {value: i + 2 for i, value in enumerate(sorted(values.dropna().unique()))}
            else:
                self.numeric.append(column)
        if neural:
            values = self.numbers(x)
            self.medians = np.array([np.median(a[np.isfinite(a)]) if np.isfinite(a).any() else 0. for a in values.T], dtype=np.float32)
            filled = np.where(np.isfinite(values), values, self.medians)
            self.means = filled.mean(axis=0, dtype=np.float64).astype(np.float32)
            scale = filled.std(axis=0, dtype=np.float64)
            self.scales = np.where(scale > 1e-6, scale, 1.).astype(np.float32)
        return self

    def numbers(self, x):
        values = x[self.numeric].to_numpy(dtype=np.float32, na_value=np.nan).copy()
        values[(values == -999999) | ~np.isfinite(values)] = np.nan
        return values

    def transform(self, x):
        if list(x) != self.columns:
            raise ValueError("Input feature schema/order changed")
        codes = {}
        for column, mapping in self.categories.items():
            values = x[column].astype("string")
            code = values.map(mapping).fillna(1).to_numpy(dtype=np.int32)
            code[values.isna().to_numpy()] = 0
            codes[column] = code
        numeric = self.numbers(x)
        if self.neural:
            missing = ~np.isfinite(numeric)
            numeric = (np.where(missing, self.medians, numeric) - self.means) / self.scales
            numeric = np.concatenate([numeric, missing.astype(np.float32)], axis=1)
            categorical = np.column_stack(list(codes.values())).astype(np.int64) if codes else np.empty((len(x),0),dtype=np.int64)
            return np.ascontiguousarray(numeric), np.ascontiguousarray(categorical)
        numeric_columns = {column: numeric[:, index] for index,column in enumerate(self.numeric)}
        for column, code in codes.items():
            numeric_columns[column] = pd.Categorical(code, categories=range(len(self.categories[column])+2))
        return pd.DataFrame(numeric_columns,index=x.index)[self.columns]


def json_safe(value):
    if isinstance(value, dict):
        return {key:json_safe(item) for key,item in value.items()}
    if isinstance(value, (tuple,list)):
        return [json_safe(item) for item in value]
    if isinstance(value,float) and not np.isfinite(value):
        return str(value)
    if isinstance(value,np.generic):
        return json_safe(value.item())
    return value
