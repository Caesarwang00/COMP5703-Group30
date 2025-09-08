# src/dataio.py
# -*- coding: utf-8 -*-
import os
import re
import pandas as pd
from typing import Dict, List, Tuple

# ---------------- 基础工具 ----------------
_SHORT_RE = re.compile(r'^(TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4})')

def to_short(bar: str):
    if not isinstance(bar, str):
        return None
    m = _SHORT_RE.match(str(bar).strip())
    return m.group(1) if m else None

def looks_like_sample(x: str) -> bool:
    try:
        s = str(x)
    except Exception:
        return False
    return s.startswith("TCGA-")

# ---------------- 方向判断与读取 ----------------
def _guess_matrix_orientation(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    统一为: 行=特征(基因/探针), 列=样本(条形码)；
    列名转短码；如有重复样本列，保留首次。
    """
    df0 = df.copy()
    first_col = df0.columns[0]
    first_lower = str(first_col).strip().lower()

    # 若第一列像“特征名”，先设为索引
    if first_lower in {"gene symbol", "gene", "symbol", "id",
                       "probe", "probes", "composite element ref", "sample"}:
        df0 = df0.set_index(first_col)

    # 列更像样本，否则转置
    col_like = sum(looks_like_sample(c) for c in df0.columns)
    idx_like = sum(looks_like_sample(i) for i in df0.index)
    if idx_like > col_like:
        df0 = df0.T

    # 数值化 + 去全NaN列
    df_num = df0.apply(pd.to_numeric, errors="coerce")
    df_num = df_num.loc[:, df_num.notna().sum(axis=0) > 0]

    # 列名转短码
    new_cols = []
    for c in df_num.columns:
        sc = to_short(str(c))
        new_cols.append(sc if sc else str(c))
    df_num.columns = new_cols

    # 重复样本列去重（保留首次）
    dup_mask = pd.Index(df_num.columns).duplicated(keep="first")
    dup_cnt = int(dup_mask.sum())
    if dup_cnt > 0:
        print(f"[dataio] 矩阵 {name} 存在重复样本列 {dup_cnt} 个（按短码去重，保留首次）。")
        df_num = df_num.loc[:, ~dup_mask]

    return df_num

def load_tsv_as_df(path: str, name: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"[dataio] 文件不存在: {name} -> {path}")
    df = pd.read_csv(path, sep="\t", low_memory=False)
    return _guess_matrix_orientation(df, name)

# ---------------- 样本对齐（支持 intersection/union） ----------------
def align_samples(
    mats: Dict[str, pd.DataFrame],
    modalities: List[str],
    label_index,
    mode: str = "intersection",
    **kwargs,
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    对齐样本：
      - intersection: 各模态列名交集 ∩ label_index
      - union       : 各模态列名并集 ∩ label_index（允许缺模态）
    返回:
      mats2: 仅保留每个模态自己拥有且在 sample_list 内的列
      sample_list: 统一的样本顺序（用于K折划分）
    """
    sets = []
    for m in modalities:
        if mats.get(m, None) is None:
            raise RuntimeError(f"[dataio] 模态 {m} 加载失败")
        sets.append(set(mats[m].columns))

    mode = str(mode).lower().strip()
    if mode not in {"intersection", "union"}:
        print(f"[dataio][WARN] 未知对齐模式 '{mode}'，已回退为 'intersection'")
        mode = "intersection"

    all_ids = (set().union(*sets) if sets else set()) if mode == "union" else (set.intersection(*sets) if sets else set())
    # 只保留有标签的样本，维持 label_index 原顺序，便于与标签一起索引
    sample_list = [s for s in label_index if s in all_ids]

    mats2 = {}
    for m in modalities:
        keep_cols = [s for s in sample_list if s in mats[m].columns]
        mats2[m] = mats[m].loc[:, keep_cols]

    return mats2, sample_list
