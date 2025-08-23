# -*- coding: utf-8 -*-
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def filter_mutation_lowfreq(mut_df: pd.DataFrame, train_cols, min_frac=0.01, min_abs=3):
    """训练折内低频过滤；返回保留行索引（基因）列表"""
    sub = mut_df[train_cols]
    n = sub.shape[1]
    thr = max(min_abs, int(np.ceil(min_frac*n)))
    gene_counts = (sub > 0.5).sum(axis=1).astype(int)   # 二值
    keep = gene_counts[gene_counts >= thr].index.tolist()
    return keep

def filter_cnv_nonzero_frac(cnv_df: pd.DataFrame, train_cols, nz_frac=0.05):
    """训练折内 非0比例过滤；返回保留行索引（基因）列表"""
    sub = cnv_df[train_cols]
    nz = (sub != 0).sum(axis=1) / sub.shape[1]
    keep = nz[nz >= nz_frac].index.tolist()
    return keep

def encode_cnv_numeric(cnv_df: pd.DataFrame, cols, genes_keep):
    """直接使用-2..2数值；返回 (X, gene_list)"""
    X = cnv_df.loc[genes_keep, cols].T.values.astype(np.float32)
    return X, genes_keep

def encode_cnv_onehot(cnv_df: pd.DataFrame, cols, genes_keep):
    """将每个基因的-2..2独热；注意维度会×5，内存更大"""
    sub = cnv_df.loc[genes_keep, cols].T   # (n_samples, n_genes)
    enc = OneHotEncoder(categories=[[-2,-1,0,1,2]]*sub.shape[1], sparse=False, handle_unknown="ignore")
    X = enc.fit_transform(sub.values)
    # 生成展开后的特征名（gene|level）
    gene_levels = []
    for g in genes_keep:
        for lv in [-2,-1,0,1,2]:
            gene_levels.append(f"{g}|{lv}")
    return X.astype(np.float32), gene_levels

def standardize_fit(X: np.ndarray):
    """返回 (Xs, scaler)；X: (n_samples, n_features)"""
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)
    return Xs.astype(np.float32), scaler

def standardize_apply(X: np.ndarray, scaler: StandardScaler):
    return scaler.transform(X).astype(np.float32)

def apply_block_weight(X: np.ndarray, w: float):
    return (X * float(w)).astype(np.float32)

def concat_blocks(*Xs):
    return np.concatenate(Xs, axis=1).astype(np.float32)
