# src/features.py
# -*- coding: utf-8 -*-
import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ---------- 简单工具 ----------
def _variance_topk(X: pd.DataFrame, k: int) -> pd.DataFrame:
    if k <= 0 or k >= X.shape[0]:
        return X
    var = X.var(axis=1)
    idx = var.nlargest(k).index
    return X.loc[idx]


def _binary_freq_filter(X: pd.DataFrame, min_frac: float, min_abs: int, topk: int) -> pd.DataFrame:
    """针对突变矩阵（0/1 或稀疏计数）按出现频率筛选"""
    cnt = (X.values != 0).sum(axis=1)
    keep_mask = (cnt >= max(min_abs, 0)) & (cnt >= min_frac * X.shape[1])
    X2 = X.loc[keep_mask]
    if topk > 0 and topk < X2.shape[0]:
        # 按频次取前 topk
        cnt2 = (X2.values != 0).sum(axis=1)
        top_idx = X2.index[np.argsort(-cnt2)[:topk]]
        X2 = X2.loc[top_idx]
    return X2


def _prescreen_by_mod(X: pd.DataFrame, modality: str, cfg: dict) -> pd.DataFrame:
    """按模态做预筛（只用训练集统计来选特征！）"""
    if modality == "cnv":
        min_nz = float(cfg.get("CNV_PRE_MIN_NONZERO_FRAC", 0.05))
        topk = int(cfg.get("CNV_PRE_TOPK_VAR", 5000))
        nz_frac = (X.values != 0).mean(axis=1)
        X = X.loc[nz_frac >= min_nz]
        X = _variance_topk(X, topk)

    elif modality == "rna":
        topk = int(cfg.get("RNA_PRE_TOPK_VAR", 5000))
        X = _variance_topk(X, topk)

    elif modality == "mut":
        min_frac = float(cfg.get("MUT_MIN_FRAC", 0.02))
        min_abs = int(cfg.get("MUT_MIN_ABS", 5))
        topk = int(cfg.get("MUT_PRE_TOPK_FREQ", 1000))
        X = _binary_freq_filter(X, min_frac=min_frac, min_abs=min_abs, topk=topk)

    return X


def _second_stage_var_topk(X: pd.DataFrame, modality: str, cfg: dict) -> pd.DataFrame:
    """二级方差TopK（与原配置名对齐）"""
    key = {
        "cnv": "CNV_TOPK_VAR",
        "rna": "RNA_TOPK_VAR",
        "mut": "MUT_TOPK_VAR",
        "meth": "METH_TOPK_VAR",
    }.get(modality, None)
    if key is None:
        return X
    k = int(cfg.get(key, 0))
    if k <= 0:
        return X
    return _variance_topk(X, k)


def _fit_pca(tr: np.ndarray, dim: int) -> Tuple[PCA, np.ndarray]:
    pca = PCA(n_components=dim, random_state=0)
    Ztr = pca.fit_transform(tr)
    return pca, Ztr


def _latent_dim_for(modality: str, cfg: dict) -> int:
    dmap = cfg.get("AE_LATENT_DIM", {})
    if isinstance(dmap, dict):
        return int(dmap.get(modality, dmap.get("default", 32)))
    return int(dmap)


def _build_matrix_for_ids(X_feat_by_samp: pd.DataFrame,
                          ids_order: List[str],
                          add_missing_indicator: bool,
                          scaler: StandardScaler = None,
                          pca: PCA = None) -> Tuple[np.ndarray, List[str]]:
    """
    X_feat_by_samp: 行=特征, 列=样本；已经完成特征选择。
    返回：矩阵按 ids_order 排列；缺失样本自动用零向量；可末尾拼接缺失指示列。
    """
    present = [sid for sid in ids_order if sid in X_feat_by_samp.columns]
    # 先对 present 样本做变换
    Xp = X_feat_by_samp[present].T.values  # (n_present, d)
    if scaler is not None and Xp.size > 0:
        Xp = scaler.transform(Xp)
    if pca is not None and Xp.size > 0:
        Xp = pca.transform(Xp)

    d_lat = Xp.shape[1] if Xp.size > 0 else (pca.n_components_ if pca is not None else X_feat_by_samp.shape[0])
    out = np.zeros((len(ids_order), d_lat), dtype=np.float32)

    # 写回 present 的行
    pos = {sid: i for i, sid in enumerate(ids_order)}
    for j, sid in enumerate(present):
        out[pos[sid], :] = Xp[j]

    if add_missing_indicator:
        miss = np.zeros((len(ids_order), 1), dtype=np.float32)
        for sid in present:
            miss[pos[sid], 0] = 1.0
        out = np.concatenate([out, miss], axis=1)

    return out, present


def select_scale_reduce(
    df: pd.DataFrame,
    modality: str,
    tr_ids: List[str],
    va_ids: List[str],
    ytr_map: Dict[str, int],   # 兼容参数（未使用）
    yva_map: Dict[str, int],   # 兼容参数（未使用）
    cfg: dict,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    统一的特征选择 + 标准化 + （可选）PCA。
    - df: 行=特征，列=样本（样本名必须能在 tr_ids/va_ids 中找到）
    - 返回的 Ztr/Zva 形状分别为 (len(tr_ids), d) / (len(va_ids), d)，即对缺失样本补零行。
    - tr_present/va_present 返回有该模态数据的样本ID列表（用于单模态头训练筛样本）。
    """
    t0 = time.time()

    # 只用训练集出现的样本做“基于训练”的统计与选择
    tr_present = [sid for sid in tr_ids if sid in df.columns]
    va_present = [sid for sid in va_ids if sid in df.columns]

    Xtr_full = df[tr_present] if len(tr_present) > 0 else df.iloc[:, :0]
    # 预筛（日志）
    X0 = Xtr_full.copy()
    Xtr_full = _prescreen_by_mod(Xtr_full, modality, cfg)
    print(f"[{modality}] 预筛: {X0.shape[0]} → {Xtr_full.shape[0]} 特征，耗时 {time.time()-t0:.1f}s")

    # 二级 TopK
    Xtr_full = _second_stage_var_topk(Xtr_full, modality, cfg)

    # ======= 标准化 + PCA =======
    scaler = StandardScaler(with_mean=True, with_std=True)
    tr_mat_for_fit = Xtr_full.T.values if Xtr_full.shape[1] > 0 else np.zeros((0, Xtr_full.shape[0]))
    if tr_mat_for_fit.shape[0] > 0:
        scaler.fit(tr_mat_for_fit)

    use_pca = bool(cfg.get("USE_PCA", True))
    pca = None
    if use_pca:
        d_lat = _latent_dim_for(modality, cfg)
        tr_scaled = scaler.transform(tr_mat_for_fit) if tr_mat_for_fit.shape[0] > 0 else np.zeros_like(tr_mat_for_fit)
        if tr_scaled.shape[0] > 0 and tr_scaled.shape[1] > 0:
            pca, _ = _fit_pca(tr_scaled, dim=min(d_lat, tr_scaled.shape[1]))

    add_ind = bool(cfg.get("ADD_MISSING_INDICATOR", True))

    # 组装训练/验证矩阵（包含缺失样本行，统一顺序）
    Ztr, tr_present = _build_matrix_for_ids(Xtr_full, tr_ids, add_missing_indicator=add_ind, scaler=scaler, pca=pca)
    # 验证用训练阶段选出的特征集合
    Xva_feat = df.loc[Xtr_full.index]
    Zva, va_present = _build_matrix_for_ids(Xva_feat, va_ids, add_missing_indicator=add_ind, scaler=scaler, pca=pca)

    return Ztr, Zva, tr_present, va_present
