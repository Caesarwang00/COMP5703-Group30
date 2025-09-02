# preprocess_with_subtype.py
# -*- coding: utf-8 -*-
import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ======================
# 基础I/O与规范化
# ======================

def _read_expression_any(path: str) -> pd.DataFrame:
    """
    读取表达矩阵（TSV/CSV/真正xlsx/“伪装xlsx的TSV”皆可）。
    期望第一列是基因名，其余列为样本ID。
    返回：DataFrame，行=基因，列=样本
    """
    p = Path(path)
    # 尝试Excel
    try:
        xls = pd.ExcelFile(p, engine="openpyxl")
        df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
    except Exception:
        # 先探测列数，1列多半是TSV被错读
        probe = pd.read_csv(p, nrows=5, header=0)
        if probe.shape[1] == 1:
            df = pd.read_csv(p, sep="\t")
        else:
            df = pd.read_csv(p)

    first_col = df.columns[0]
    df = df.rename(columns={first_col: "gene"}).dropna(subset=["gene"])
    # 重复基因聚合（取均值）
    df = df.groupby("gene", as_index=False).mean(numeric_only=True)
    df = df.set_index("gene")
    return df


def _norm_tcga_15(s: str) -> str:
    """TCGA样本名统一到前15位（含样本类型两位），便于跨表对齐。"""
    s = str(s).strip()
    # 常见模式：TCGA-XX-XXXX-YYZ... -> 取前15字符足够区分到 -YY
    return s[:15]


# ======================
# 临床矩阵读取与subtype提取
# ======================

def _read_any_table(path: str) -> pd.DataFrame:
    """宽容地读取任意表格：先试xlsx，再试csv/tsv。"""
    p = Path(path)
    try:
        xls = pd.ExcelFile(p, engine="openpyxl")
        df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
        return df
    except Exception:
        # 尝试CSV/TSV
        probe = pd.read_csv(p, nrows=5, header=0)
        if probe.shape[1] == 1:
            return pd.read_csv(p, sep="\t")
        return pd.read_csv(p)


def _guess_orientation_and_extract_subtype(df: pd.DataFrame) -> Dict[str, str]:
    """
    兼容两种常见临床矩阵布局：
    A) 行=样本, 列=临床字段（包含 'subtype' 等）
    B) 行=临床字段, 列=样本（TCGA有时是这种“clinicalMatrix”）
    自动发现“subtype”相关字段并返回 {sample_id_15: subtype_label}
    """
    # 关键词：尽量覆盖 GBM 常见命名
    key_patterns = [
        r"subtype", r"molecular[_\s]*subtype", r"class", r"cluster",
        r"gene[\s_]*expression[\s_]*subtype", r"methylation[\s_]*subtype"
    ]
    key_regex = re.compile("|".join(key_patterns), flags=re.IGNORECASE)

    # Helper: 清洗列/行名
    def _clean(s): return str(s).strip()

    # 情况A：列里有subtype样式字段
    cols = [c for c in df.columns if key_regex.search(str(c))]
    if len(cols) >= 1:
        # 期望此时 行=样本
        # 找一个最像subtype的列（优先包含“subtype”的）
        cols_sorted = sorted(cols, key=lambda x: (0 if re.search(r"subtype", str(x), re.I) else 1, str(x)))
        col_sub = cols_sorted[0]
        tmp = df[[col_sub]].copy()
        # 尝试找到样本ID列；若无，就把index当作样本
        sample_col = None
        for cand in df.columns:
            if re.search(r"^(sample|barcode|tcga|patient)", str(cand), re.I):
                sample_col = cand
                break
        if sample_col is not None and sample_col != col_sub:
            tmp.index = df[sample_col].map(_norm_tcga_15)
        else:
            tmp.index = df.index.map(_clean).map(_norm_tcga_15)
        tmp = tmp.dropna()
        return {idx: str(v) for idx, v in zip(tmp.index, tmp[col_sub]) if pd.notna(v)}

    # 情况B：行里有subtype样式字段
    idx = [_clean(i) for i in df.index] if df.index.name or df.index.tolist() else None
    if idx is None or (len(idx) == 0 or isinstance(df.index, pd.RangeIndex)):
        # 尝试把第一列作为“行名”
        first_col = df.columns[0]
        dfB = df.set_index(first_col)
    else:
        dfB = df.copy()
    # 在行索引里找subtype行
    cand_rows = [r for r in map(str, dfB.index) if key_regex.search(r)]
    if len(cand_rows) >= 1:
        # 同样优先含“subtype”的
        cand_rows_sorted = sorted(cand_rows, key=lambda x: (0 if re.search(r"subtype", x, re.I) else 1, x))
        row_sub = cand_rows_sorted[0]
        series = dfB.loc[row_sub].copy()
        # 列名应是样本ID
        series.index = [ _norm_tcga_15(c) for c in series.index ]
        return {idx: str(v) for idx, v in series.items() if pd.notna(v)}

    raise ValueError("未在临床矩阵中检测到 'subtype' 相关字段，请检查文件结构或手动指定。")


def load_subtype_labels_from_clinical(clinical_path: str) -> Dict[str, str]:
    """从临床矩阵中抽取 {sample15: subtype} 标签。"""
    raw = _read_any_table(clinical_path)
    # 若第一列是无名列，先赋名避免后续混乱
    if raw.columns[0] is None:
        raw = raw.rename(columns={raw.columns[0]: "key0"})
    # 尝试两种方向
    try:
        labels = _guess_orientation_and_extract_subtype(raw)
        return labels
    except Exception:
        # 有些xlsx首行是字段名但整体向右/下偏移，尝试转置重试
        labels = _guess_orientation_and_extract_subtype(raw.T)
        return labels


# ======================
# 特征过滤/选择、切分、标准化
# ======================

def filter_and_select_genes(
    expr_g_by_s: pd.DataFrame,
    min_nonzero_pct: float = 0.2,
    topk_by_variance: int = 2000
) -> pd.DataFrame:
    """
    输入：行=基因，列=样本
    过滤低非零占比基因 + 选Top-K方差基因
    """
    n = expr_g_by_s.shape[1]
    nonzero_pct = (expr_g_by_s != 0).sum(axis=1) / n
    kept = expr_g_by_s.loc[nonzero_pct >= min_nonzero_pct]
    if kept.empty:
        raise ValueError("过滤后没有基因，请降低 min_nonzero_pct。")
    var = kept.var(axis=1)
    k = min(topk_by_variance, kept.shape[0])
    top_genes = var.sort_values(ascending=False).head(k).index
    return kept.loc[top_genes]


def stratified_split(
    X: pd.DataFrame, y: pd.Series,
    test_size: float = 0.2, val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    val_rel = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_rel, stratify=y_trainval, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def zscore_fit_transform(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(X_train.values)
    X_train_z = pd.DataFrame(scaler.transform(X_train.values), index=X_train.index, columns=X_train.columns)
    X_val_z   = pd.DataFrame(scaler.transform(X_val.values),   index=X_val.index,   columns=X_val.columns)
    X_test_z  = pd.DataFrame(scaler.transform(X_test.values),  index=X_test.index,  columns=X_test.columns)
    return X_train_z, X_val_z, X_test_z, scaler


# ======================
# 主流程
# ======================

def preprocess_pipeline_with_subtype(
    expr_path: str,
    clinical_path: str,
    min_nonzero_pct: float = 0.2,
    topk_by_variance: int = 2000,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    subtype_normalization: Optional[Dict[str, str]] = None
):
    """
    读表达矩阵 + 临床矩阵，抽取subtype，完成特征选择/标准化与分层切分。
    返回：
      X_train/X_val/X_test（行=样本、列=基因，已z-score）
      y_train/y_val/y_test（subtype标签，pd.Series）
      genes（选中的基因列表） scaler（标准化器）
    """
    # 1) 读表达：行=基因，列=样本
    expr_g_by_s = _read_expression_any(expr_path)
    expr_g_by_s.columns = [ _norm_tcga_15(c) for c in expr_g_by_s.columns ]

    # 2) 读临床 -> subtype字典
    sub_map = load_subtype_labels_from_clinical(clinical_path)

    # 可选：统一subtype命名（大小写/同义）
    if subtype_normalization:
        sub_map = {k: subtype_normalization.get(str(v).strip(), str(v).strip()) for k, v in sub_map.items()}
    else:
        # 默认做一些常见GBM归一化
        norm = {
            "classical": "Classical",
            "mesenchymal": "Mesenchymal",
            "proneural": "Proneural",
            "neural": "Neural",
        }
        sub_map = {k: norm.get(str(v).strip().lower(), str(v).strip()) for k, v in sub_map.items()}

    # 3) 对齐：只保留有subtype的样本
    labeled_cols = [c for c in expr_g_by_s.columns if c in sub_map]
    if not labeled_cols:
        raise ValueError("表达矩阵与临床矩阵未找到可对齐的样本ID（前15位）。请检查两表的样本命名。")
    expr_g_by_s = expr_g_by_s[labeled_cols]
    y = pd.Series([sub_map[c] for c in labeled_cols], index=labeled_cols, name="subtype")

    # 4) 过滤与Top-K
    expr_g_by_s = filter_and_select_genes(expr_g_by_s, min_nonzero_pct=min_nonzero_pct, topk_by_variance=topk_by_variance)

    # 5) 转成 行=样本、列=基因
    X = expr_g_by_s.T.copy()

    # 6) 分层切分（基于subtype）
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(
        X, y, test_size=test_size, val_size=val_size, random_state=random_state
    )

    # 7) 标准化（仅在训练集拟合）
    X_train, X_val, X_test, scaler = zscore_fit_transform(X_train, X_val, X_test)

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "genes": list(X.columns),
        "scaler": scaler
    }


if __name__ == "__main__":
    # 你的两份文件路径（根据你的实际路径修改）：
    expr_path = r"D:/毕设数据/HiSeqV2.xlsx"  # 实为TSV
    clinical_path = r"D:/毕设数据/TCGA.GBM.sampleMap_GBM_clinicalMatrix.xlsx"

    results = preprocess_pipeline_with_subtype(
        expr_path=expr_path,
        clinical_path=clinical_path,
        min_nonzero_pct=0.2,
        topk_by_variance=2000,
        test_size=0.2,
        val_size=0.1,
        random_state=42,
    )

    # 简要查看形状
    for k in ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]:
        v = results[k]
        print(k, getattr(v, "shape", (len(v),)))
    print("Num genes:", len(results["genes"]))
