# -*- coding: utf-8 -*-
from typing import List, Union, Dict, Any
import numpy as np
from sklearn.metrics import pairwise_distances


def _row_normalize(A: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return A / (A.sum(axis=1, keepdims=True) + eps)


def _affinity_matrix_local_scaling(
    X: np.ndarray, k: int = 20, metric: str = "euclidean", eps: float = 1e-9
) -> np.ndarray:
    """
    Self-tuning affinity (Zelnik-Manor & Perona, 2004):
      A_ij = exp( -||xi-xj||^2 / (sigma_i * sigma_j + eps) )
    sigma_i: xi 到其第 k 个近邻的距离（局部尺度，排除自身零距离）。
    最后做行归一化。
    """
    X = np.asarray(X, dtype=np.float32)
    D = pairwise_distances(X, metric=metric)  # (n, n)
    n = D.shape[0]
    # 第 k 邻距离作为局部尺度（k 至少为 1，小于样本数）
    k_eff = max(1, min(k, n - 1))
    idx = np.argsort(D, axis=1)                  # 每行升序，[self, 1st, 2nd, ...]
    kth = D[np.arange(n), idx[:, k_eff]]         # 排除自身后第 k 个近邻
    sigma = kth + eps
    denom = (sigma[:, None] * sigma[None, :]) + eps
    A = np.exp(- (D ** 2) / denom)
    np.fill_diagonal(A, 1.0)
    A = _row_normalize(A, eps)
    return A


def _knn_graph(W: np.ndarray, k: int) -> np.ndarray:
    """
    对相似度矩阵做 KNN 稀疏化，并保持对称 + 行归一化。
    输入/输出均为 (n, n) 的行归一化相似度矩阵。
    """
    n = W.shape[0]
    k_eff = max(1, min(k, n - 1))
    W_knn = np.zeros_like(W)
    # 去掉自身后选前 k 个最大相似度
    idx = np.argsort(-W, axis=1)[:, 1:k_eff + 1]
    rows = np.repeat(np.arange(n), k_eff)
    cols = idx.reshape(-1)
    W_knn[rows, cols] = W[rows, cols]
    # 对称化 + 行归一化
    W_knn = np.maximum(W_knn, W_knn.T)
    W_knn = _row_normalize(W_knn)
    return W_knn


def snf(sim_list: List[np.ndarray], K: int = 20, T: int = 20, eps: float = 1e-9) -> np.ndarray:
    """
    Similarity Network fusion (Wang et al., 2014)：
      对每个模态 m：
        Pm^{t+1} = S_m * (平均_{i!=m} P_i^{t}) * S_m^T
      其中 S_m 是模态 m 的 KNN 稀疏相似度，最后取各 Pm 的平均。
    输入：每个模态的“行归一化”相似度矩阵。
    返回：融合后的相似度矩阵（行归一化、对称）。
    """
    assert len(sim_list) > 0, "sim_list 不能为空"
    M = len(sim_list)
    # KNN 稀疏化
    S_list = [_knn_graph(W, K) for W in sim_list]
    # 初始化 P_list
    P_list = [W.copy() for W in S_list]

    for _ in range(int(T)):
        stackP = np.stack(P_list, axis=0)      # (M, n, n)
        mean_all = np.mean(stackP, axis=0)     # (n, n)
        P_new = []
        for m in range(M):
            if M > 1:
                others = (mean_all * M - P_list[m]) / (M - 1)
            else:
                others = P_list[m]
            Pm = S_list[m] @ others @ S_list[m].T
            Pm = _row_normalize(Pm, eps)
            P_new.append(Pm)
        P_list = P_new

    fused = np.mean(np.stack(P_list, axis=0), axis=0)
    fused = (fused + fused.T) / 2.0
    fused = _row_normalize(fused, eps)
    return fused


def build_snf_from_embeddings(
    emb_list: List[np.ndarray],
    k: int = 20,
    t: int = 20,
    metric: str = "euclidean",
) -> np.ndarray:
    """
    先用“局部自适应尺度”的高斯亲和构图，再做 SNF 融合。
    返回 numpy.ndarray，相容后续 torch 转换。
    """
    sims = [_affinity_matrix_local_scaling(Z, k=k, metric=metric) for Z in emb_list]
    return snf(sims, K=k, T=t)


def build_snf_affinity(
    X_list: List[np.ndarray],
    *args,
    **kwargs
) -> np.ndarray:
    """
    供 train.py 调用的对外接口。
    兼容两种调用方式：
      - build_snf_affinity(X_list, **snf_cfg)
      - build_snf_affinity(X_list, snf_cfg_dict)

    支持的参数键（大小写均可）：
      k / K：KNN 邻居数（默认 20）
      t / T：SNF 迭代轮数（默认 20）
      metric：距离度量（默认 'euclidean'）
    """
    # 兼容 train.py 的 try/except 签名
    if len(args) == 1 and isinstance(args[0], dict):
        params: Dict[str, Any] = args[0]
    else:
        params = kwargs

    # 取参数，支持大小写键名
    def _get(*keys, default=None):
        for k in keys:
            if k in params:
                return params[k]
        return default

    k = int(_get("K", "k", default=20))
    t = int(_get("T", "t", default=20))
    metric = _get("metric", default="euclidean")

    return build_snf_from_embeddings(X_list, k=k, t=t, metric=metric)
