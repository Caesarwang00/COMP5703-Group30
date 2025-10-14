
# -*- coding: utf-8 -*-
"""
特征工程 & 预处理
"""
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif

# === 基础筛选 ===
def filter_mutation_lowfreq(mut_df, train_ids, min_frac=0.01, min_abs=3):
    """筛除低频突变基因（在 train_ids 的样本中计算阳性比例）"""
    X = mut_df[train_ids].values
    pos = X.sum(axis=1)
    thr = max(min_abs, int(np.ceil(min_frac * len(train_ids))))
    keep = np.where(pos >= thr)[0]
    return list(mut_df.index[keep])

def filter_cnv_nonzero_frac(cnv_df, train_ids, nz_frac=0.05):
    """按 train_ids 的非零占比保留 CNV 基因"""
    X = cnv_df[train_ids].values
    frac = (X != 0).mean(axis=1)
    keep = np.where(frac >= nz_frac)[0]
    return list(cnv_df.index[keep])

# === 编码 ===
def encode_cnv_numeric(cnv_df, sample_ids, keep_genes):
    """将 thresholded CNV 直接作为数值（-2..2），返回 (X, genes)"""
    sub = cnv_df.loc[keep_genes, sample_ids]
    return sub.T.values.astype(np.float32), list(sub.index)

def encode_cnv_onehot(cnv_df, sample_ids, keep_genes):
    """将 thresholded CNV one-hot 编码"""
    sub = cnv_df.loc[keep_genes, sample_ids]
    enc = OneHotEncoder(categories='auto', sparse=False, handle_unknown='ignore')
    X = enc.fit_transform(sub.T.values)
    return X.astype(np.float32), enc

# === 标准化 ===
def standardize_fit(X):
    sc = StandardScaler(with_mean=True, with_std=True)
    Xs = sc.fit_transform(X)
    return Xs.astype(np.float32), sc

def standardize_apply(X, scaler):
    return scaler.transform(X).astype(np.float32)

# === 融合 & 加权 ===
def apply_block_weight(X, w=1.0):
    if w is None: w = 1.0
    return (X * float(w)).astype(np.float32)

def concat_blocks(*blocks):
    return np.concatenate(blocks, axis=1).astype(np.float32)

# === 可选：全局 TopK ===
def select_kbest_global(Xtr, ytr, Xva, k):
    sel = SelectKBest(score_func=f_classif, k=k)
    Xtr2 = sel.fit_transform(Xtr, ytr)
    Xva2 = sel.transform(Xva)
    return Xtr2.astype(np.float32), Xva2.astype(np.float32), sel
