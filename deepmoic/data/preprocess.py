# -*- coding: utf-8 -*-
"""
预处理模块（完整版）：
- 读取多模态矩阵（行=特征，列=样本；第一列为特征名），统一样本ID为 TCGA 短条码并聚合同短码重复列；
- 从 TCGA clinicalMatrix 中抽取标签（支持 ROW="auto"，并自动判定布局：行=变量/列=样本 或行=样本/列=变量）；
- 安全 z-score（零方差列不会再引起 NaN/警告）；
- 可选：按方差筛选每个模态的 Top-K 特征（不改样本数）；
- 返回：各模态 X_list、整数编码 y、样本顺序 sample_ids、LabelEncoder。
"""
from typing import Dict, List, Tuple, Optional
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ..utils.io import read_table_auto, read_labels
from ..utils.tcga import to_short_sample_id, is_primary_tumor

# ---------- 工具 ----------

_TCGA_PATTERN = re.compile(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-\d{2}[A-Z]?$", re.I)

def _looks_like_tcga(s: str) -> bool:
    if not isinstance(s, str):
        s = str(s)
    s = s.strip().replace(".", "-")
    return bool(_TCGA_PATTERN.match(s))

def _normalize_sample_ids(ids: List[str]) -> List[str]:
    return [to_short_sample_id(x) for x in ids]

def _aggregate_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    if len(set(df.columns)) < len(df.columns):
        df = df.groupby(axis=1, level=0).mean()
    return df

def _safe_zscore_samples_by_feature(X: np.ndarray) -> np.ndarray:
    """
    自定义 z-score：对每个特征（列）做 (x-mean)/std；
    std<=1e-8 或非法值时用 1 替代，标准化结果该列为 0，不产生 NaN。
    """
    X = np.asarray(X, dtype=np.float32)
    X[~np.isfinite(X)] = 0.0
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    safe_std = np.where((std > 1e-8) & np.isfinite(std), std, 1.0)
    Xn = (X - mean) / safe_std
    Xn = np.nan_to_num(Xn, copy=False)
    return Xn.astype(np.float32)

def _select_top_var(df: pd.DataFrame, k: Optional[int]) -> pd.DataFrame:
    if k is None or k <= 0 or k >= df.shape[0]:
        return df
    var = df.var(axis=1, skipna=True)
    top_idx = var.sort_values(ascending=False).head(int(k)).index
    return df.loc[top_idx]

# ---------- 模态读取 ----------

def _load_modality(path: str, topk: Optional[int] = None) -> pd.DataFrame:
    df = read_table_auto(path)
    if df.shape[1] < 2:
        raise ValueError(f"Modality file has <2 columns: {path}")
    df = df.set_index(df.columns[0])
    # 列名统一为短条码并聚合
    new_cols = _normalize_sample_ids(df.columns.astype(str).tolist())
    df.columns = new_cols
    df = _aggregate_duplicate_columns(df)
    # 可选：按方差筛选 Top-K 特征（行=特征）
    df = _select_top_var(df, topk)
    return df

# ---------- clinicalMatrix → 标签 ----------

def _pick_subtype_row_auto_by_rows(cm_by_rows: pd.DataFrame) -> str:
    index_raw = list(cm_by_rows.index.astype(str))
    def score_line(name: str, ser: pd.Series):
        vals = pd.Series(ser.astype(str).str.strip().str.lower())
        vals_nz = vals[~vals.isin({"", "na", "nan", "none"})]
        nuniq = vals_nz.nunique(dropna=True)
        tokens_long = {"proneural","classical","neural","mesenchymal","g-cimp","gcimp"}
        tokens_short = {"pn","cl","ne","mes"}
        hit_long = vals_nz.isin(tokens_long).sum()
        hit_short = vals_nz.isin(tokens_short).sum()
        name_l = name.lower()
        has_subtype = "subtype" in name_l
        has_expr = any(k in name_l for k in ["mrna","gene","expr","expression","transcript"])
        alphabetic_ratio = float(vals_nz.str.fullmatch(r"[a-zA-Z\-\+_/\. ]+").mean())
        score = (hit_long * 2.0 + hit_short * 0.8 +
                 (1.0 if has_subtype else 0.0) + (0.5 if has_expr else 0.0) +
                 (0.3 if 2 <= nuniq <= 15 else -0.3) + 0.3 * alphabetic_ratio)
        return score
    best_name, best_score = None, -1e18
    for name in index_raw:
        sc = score_line(name, cm_by_rows.loc[name])
        if sc > best_score:
            best_name, best_score = name, sc
    if best_name is None:
        raise KeyError("auto 模式未能在“行=变量名”的布局里找到子型行")
    return best_name

def _pick_subtype_col_auto_by_cols(cm_by_cols: pd.DataFrame) -> str:
    cols_raw = list(cm_by_cols.columns.astype(str))
    def score_col(name: str, ser: pd.Series):
        vals = pd.Series(ser.astype(str).str.strip().str.lower())
        vals_nz = vals[~vals.isin({"", "na", "nan", "none"})]
        nuniq = vals_nz.nunique(dropna=True)
        tokens_long = {"proneural","classical","neural","mesenchymal","g-cimp","gcimp"}
        tokens_short = {"pn","cl","ne","mes"}
        hit_long = vals_nz.isin(tokens_long).sum()
        hit_short = vals_nz.isin(tokens_short).sum()
        name_l = name.lower()
        has_subtype = "subtype" in name_l
        has_expr = any(k in name_l for k in ["mrna","gene","expr","expression","transcript"])
        alphabetic_ratio = float(vals_nz.str.fullmatch(r"[a-zA-Z\-\+_/\. ]+").mean())
        score = (hit_long * 2.0 + hit_short * 0.8 +
                 (1.0 if has_subtype else 0.0) + (0.5 if has_expr else 0.0) +
                 (0.3 if 2 <= nuniq <= 15 else -0.3) + 0.3 * alphabetic_ratio)
        return score
    best_name, best_score = None, -1e18
    for name in cols_raw:
        sc = score_col(name, cm_by_cols[name])
        if sc > best_score:
            best_name, best_score = name, sc
    if best_name is None:
        raise KeyError("auto 模式未能在“行=样本ID”的布局里找到子型列")
    return best_name

def _labels_from_clinical_matrix(path: str,
                                 row: str,
                                 primary_only: bool = True) -> pd.DataFrame:
    cm_raw = read_table_auto(path)

    # 判定布局
    col_like_sample = sum(_looks_like_tcga(str(c)) for c in cm_raw.columns) >= max(3, int(0.6 * len(cm_raw.columns)))
    if col_like_sample:
        # 行=变量，列=样本
        cm = cm_raw.set_index(cm_raw.columns[0])
        new_cols = _normalize_sample_ids(cm.columns.astype(str).tolist())
        cm.columns = new_cols
        cm = _aggregate_duplicate_columns(cm)
        if row and row.lower() != "auto":
            lower_map = {r.lower(): r for r in cm.index.astype(str)}
            chosen = lower_map.get(row.lower())
            if chosen is None:
                def norm(s): return re.sub(r"[\s_\-]+", "", s.strip().lower())
                norm2raw = {norm(r): r for r in cm.index.astype(str)}
                chosen = norm2raw.get(norm(row))
            if chosen is None:
                cand = [r for r in cm.index.astype(str) if row.lower() in r.lower()]
                chosen = cand[0] if len(cand) > 0 else None
            if chosen is None:
                raise KeyError(f"Row '{row}' not found in clinical matrix (by rows)")
        else:
            chosen = _pick_subtype_row_auto_by_rows(cm)
        ser = cm.loc[chosen]
        df = ser.rename("label").to_frame()
        df["sample_id"] = df.index.astype(str)
    else:
        # 试探行=样本
        first_col = cm_raw.columns[0]
        first_col_vals = cm_raw[first_col].astype(str).tolist()
        row_like_sample = sum(_looks_like_tcga(v) for v in first_col_vals) >= max(3, int(0.6 * len(first_col_vals)))
        if not row_like_sample:
            # 两种方向都试
            try:
                cm_try = cm_raw.set_index(cm_raw.columns[0])
                new_cols = _normalize_sample_ids(cm_try.columns.astype(str).tolist())
                cm_try.columns = new_cols
                cm_try = _aggregate_duplicate_columns(cm_try)
                chosen = _pick_subtype_row_auto_by_rows(cm_try) if (not row or row.lower()=="auto") else row
                ser = cm_try.loc[chosen]
                df = ser.rename("label").to_frame()
                df["sample_id"] = df.index.astype(str)
            except Exception:
                cm_try2 = cm_raw.copy()
                cm_try2 = cm_try2.rename(columns={cm_try2.columns[0]: "sample_id"}).set_index("sample_id")
                cm_try2.index = _normalize_sample_ids(cm_try2.index.astype(str).tolist())
                cm_try2 = cm_try2.groupby(level=0).first()
                chosen = _pick_subtype_col_auto_by_cols(cm_try2) if (not row or row.lower()=="auto") else row
                ser = cm_try2[chosen]
                df = ser.rename("label").to_frame()
                df["sample_id"] = df.index.astype(str)
        else:
            cm = cm_raw.rename(columns={cm_raw.columns[0]: "sample_id"}).set_index("sample_id")
            cm.index = _normalize_sample_ids(cm.index.astype(str).tolist())
            cm = cm.groupby(level=0).first()
            if row and row.lower() != "auto":
                col_lower = {c.lower(): c for c in cm.columns.astype(str)}
                chosen = col_lower.get(row.lower())
                if chosen is None:
                    def norm(s): return re.sub(r"[\s_\-]+", "", s.strip().lower())
                    norm2raw = {norm(c): c for c in cm.columns.astype(str)}
                    chosen = norm2raw.get(norm(row))
                if chosen is None:
                    cand = [c for c in cm.columns.astype(str) if row.lower() in c.lower()]
                    chosen = cand[0] if len(cand) > 0 else None
                if chosen is None:
                    raise KeyError(f"Column '{row}' not found in clinical matrix (by columns)")
            else:
                chosen = _pick_subtype_col_auto_by_cols(cm)
            ser = cm[chosen]
            df = ser.rename("label").to_frame()
            df["sample_id"] = df.index.astype(str)

    df["sample_id"] = df["sample_id"].map(to_short_sample_id)
    df = (df.groupby("sample_id")["label"]
            .apply(lambda s: s.dropna().iloc[0] if s.dropna().size > 0 else np.nan)
            .reset_index())
    if primary_only:
        df = df[df["sample_id"].map(is_primary_tumor)]
    df = df.dropna(subset=["label"])
    print(f"[clinicalMatrix] 选用标签字段/行：'{chosen}'（如需固定，可将其填入 YAML 的 DATA.LABELS_FROM_CLINICAL.ROW）")
    return df[["sample_id", "label"]]

# ---------- 主入口 ----------

def align_and_standardize(modality_paths: Dict[str, str],
                          labels_path: Optional[str] = None,
                          clinical_cfg: Optional[Dict] = None,
                          take_intersection: bool = True,
                          zscore: bool = True,
                          select_top_var: Optional[Dict] = None
                          ) -> Tuple[List[np.ndarray], np.ndarray, List[str], LabelEncoder]:
    """
    支持标签两种来源：
      - labels_path: 传统 CSV（sample_id,label）
      - clinical_cfg: {PATH, ROW, PRIMARY_ONLY}
    select_top_var（可选）示例：
      {"enabled": true, "per_modality_k": {"rna": 5000, "meth": 20000, "cnv": 2000}}
    """
    # 1) 读取模态
    top_enable = False
    kmap = {}
    if isinstance(select_top_var, dict) and select_top_var.get("enabled", False):
        top_enable = True
        kmap = select_top_var.get("per_modality_k", {}) or {}

    mod_dfs: Dict[str, pd.DataFrame] = {}
    for name, p in modality_paths.items():
        if p is None or str(p).strip() == "":
            continue
        topk = int(kmap.get(name, 0)) if top_enable else None
        mod_dfs[name] = _load_modality(p, topk=topk)
    if len(mod_dfs) == 0:
        raise ValueError("No valid modality files provided.")

    # 2) 标签
    if clinical_cfg is not None:
        lab = _labels_from_clinical_matrix(
            path=clinical_cfg["PATH"],
            row=clinical_cfg.get("ROW", "auto"),
            primary_only=clinical_cfg.get("PRIMARY_ONLY", True)
        )
    else:
        if labels_path is None:
            raise ValueError("No labels provided: set DATA.LABELS or DATA.LABELS_FROM_CLINICAL in YAML.")
        lab = read_labels(labels_path)
        lab["sample_id"] = lab["sample_id"].map(to_short_sample_id)

    # 3) 样本交集
    for df in mod_dfs.values():
        df.columns = df.columns.astype(str)
    sample_sets = [set(df.columns) for df in mod_dfs.values()]
    sample_all = set.union(*sample_sets)
    sample_common = (set.intersection(*sample_sets, set(lab["sample_id"]))
                     if take_intersection else
                     sample_all.intersection(set(lab["sample_id"])))
    sample_ids = sorted(list(sample_common))
    if len(sample_ids) < 10:
        tips = (f"Too few intersected samples ({len(sample_ids)}). 建议排查：\n"
                f"  1) 将 DATA.LABELS_FROM_CLINICAL.PRIMARY_ONLY 改为 false 测试；\n"
                f"  2) 确认各模态列名可标准化为短条码（01A/01B 会自动聚合为 01）；\n"
                f"  3) 先只用 rna 跑通，再逐个加模态定位。")
        raise ValueError(tips)

    # 4) 对齐与标准化
    X_list: List[np.ndarray] = []
    for name, df in mod_dfs.items():
        df = df.loc[:, sample_ids]
        X = df.T.values  # samples × features
        X = _safe_zscore_samples_by_feature(X) if zscore else np.asarray(X, dtype=np.float32)
        X_list.append(X)

    # 5) 标签编码
    le = LabelEncoder()
    lab = lab[lab["sample_id"].isin(sample_ids)].copy()
    lab = lab.set_index("sample_id").loc[sample_ids]
    y = le.fit_transform(lab["label"].astype(str).values)

    return X_list, y.astype(int), sample_ids, le
