# src/preprocess.py
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def zscore_fit(Xtr_df: pd.DataFrame):
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(Xtr_df.T.values)  # (n_samples, n_features)
    feat_order = list(Xtr_df.index)
    return scaler, feat_order

def zscore_apply(scaler: StandardScaler, X_df: pd.DataFrame, feat_order=None, return_dataframe: bool = True):
    Xv = X_df if feat_order is None else X_df.reindex(index=feat_order)
    X_np = scaler.transform(Xv.T.values).T  # (n_features, n_samples)
    if return_dataframe:
        return pd.DataFrame(X_np, index=Xv.index, columns=Xv.columns)
    else:
        return X_np
