# -*- coding: utf-8 -*-
"""
特征工程：筛选、标准化、降维、加权与拼接
"""
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

# ===== 基础筛选 =====
def filter_cnv_nonzero_frac(cnv_df, train_ids, nz_frac=0.05):
    X = cnv_df[train_ids].values
    frac = (X != 0).mean(axis=1)
    keep = np.where(frac >= nz_frac)[0]
    return list(cnv_df.index[keep])

def filter_rna_lowvar(rna_df, train_ids, min_std=0.1):
    X = rna_df[train_ids].values.astype(np.float32)
    std = X.std(axis=1)
    keep = np.where(std >= float(min_std))[0]
    return list(rna_df.index[keep])

def filter_meth_by_var(meth_df, train_ids, topk_var=None, min_std=None):
    X = meth_df[train_ids].values.astype(np.float32)
    if topk_var is not None and int(topk_var) > 0:
        var = X.var(axis=1)
        idx = np.argsort(-var)[:int(topk_var)]
        return list(meth_df.index[idx])
    if min_std is not None:
        std = X.std(axis=1)
        keep = np.where(std >= float(min_std))[0]
        return list(meth_df.index[keep])
    return list(meth_df.index)

# ===== 编码（CNV） =====
def encode_cnv_numeric(cnv_df, sample_ids, keep_genes):
    sub = cnv_df.loc[keep_genes, sample_ids]
    return sub.T.values.astype(np.float32), list(sub.index)

def encode_cnv_onehot(cnv_df, sample_ids, keep_genes):
    sub = cnv_df.loc[keep_genes, sample_ids]
    enc = OneHotEncoder(categories='auto', sparse=False, handle_unknown='ignore')
    X = enc.fit_transform(sub.T.values)
    return X.astype(np.float32), enc

# ===== 标准化 =====
def standardize_fit(X):
    sc = StandardScaler(with_mean=True, with_std=True)
    Xs = sc.fit_transform(X)
    return Xs.astype(np.float32), sc

def standardize_apply(X, scaler):
    return scaler.transform(X).astype(np.float32)

# ===== PCA 降维 =====
def pca_fit(X, n_components=256, random_state=42):
    p = PCA(n_components=n_components, svd_solver="auto", random_state=random_state)
    X2 = p.fit_transform(X)
    return X2.astype(np.float32), p

def pca_apply(X, pca):
    return pca.transform(X).astype(np.float32)

# ===== 加权与拼接 =====
def apply_block_weight(X, w=1.0):
    return (X * float(w)).astype(np.float32)

def concat_blocks(*blocks):
    return np.concatenate(blocks, axis=1).astype(np.float32)

# ===== 融合后TopK（可选） =====
def select_kbest_global(Xtr, ytr, Xva, k):
    sel = SelectKBest(score_func=f_classif, k=k)
    Xtr2 = sel.fit_transform(Xtr, ytr)
    Xva2 = sel.transform(Xva)
    return Xtr2.astype(np.float32), Xva2.astype(np.float32), sel
