# train_gbm_subtype.py
# -*- coding: utf-8 -*-
"""
端到端：读取表达矩阵 + 临床subtype -> 预处理(z-score/TopK方差) -> 分层划分 -> 训练(LogReg/SVM) -> 评估
"""
import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd


try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
except ModuleNotFoundError as e:
    raise SystemExit(
        "缺少依赖：scikit-learn\n"
        "请在命令行执行：D:/python/python.exe -m pip install scikit-learn pandas numpy openpyxl\n"
        "然后重新运行本脚本。"
    )

# 读表达矩阵、临床、抽取subtype
def _read_expression_any(path: str) -> pd.DataFrame:
    p = Path(path)
    try:
        xls = pd.ExcelFile(p, engine="openpyxl")
        df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
    except Exception:
        # 探测分隔符
        probe = pd.read_csv(p, nrows=5, header=0)
        if probe.shape[1] == 1:
            df = pd.read_csv(p, sep="\t")
        else:
            df = pd.read_csv(p)
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "gene"}).dropna(subset=["gene"])
    df = df.groupby("gene", as_index=False).mean(numeric_only=True)  # 重复基因聚合
    df = df.set_index("gene")
    return df

def _read_any_table(path: str) -> pd.DataFrame:
    p = Path(path)
    try:
        xls = pd.ExcelFile(p, engine="openpyxl")
        return pd.read_excel(xls, sheet_name=xls.sheet_names[0])
    except Exception:
        probe = pd.read_csv(p, nrows=5, header=0)
        if probe.shape[1] == 1:
            return pd.read_csv(p, sep="\t")
        return pd.read_csv(p)

def _norm_tcga_15(s: str) -> str:
    return str(s).strip()[:15]

def _guess_orientation_and_extract_subtype(df: pd.DataFrame) -> Dict[str, str]:
    key_patterns = [
        r"subtype", r"molecular[_\s]*subtype", r"class", r"cluster",
        r"gene[\s_]*expression[\s_]*subtype", r"methylation[\s_]*subtype"
    ]
    key_regex = re.compile("|".join(key_patterns), flags=re.IGNORECASE)

    # 列名里直接有subtype
    cols = [c for c in df.columns if key_regex.search(str(c))]
    if cols:
        cols_sorted = sorted(cols, key=lambda x: (0 if re.search(r"subtype", str(x), re.I) else 1, str(x)))
        col_sub = cols_sorted[0]
        tmp = df[[col_sub]].copy()
        # 尝试找样本列；没有就用索引
        sample_col = None
        for cand in df.columns:
            if re.search(r"^(sample|barcode|tcga|patient)", str(cand), re.I):
                sample_col = cand; break
        if sample_col and sample_col != col_sub:
            tmp.index = df[sample_col].map(_norm_tcga_15)
        else:
            tmp.index = df.index.map(lambda x: _norm_tcga_15(str(x)))
        tmp = tmp.dropna()
        return {idx: str(v) for idx, v in zip(tmp.index, tmp[col_sub]) if pd.notna(v)}

    # subtype在行索引
    if not df.index.name and isinstance(df.index, pd.RangeIndex):
        
        dfB = df.set_index(df.columns[0])
    else:
        dfB = df
    cand_rows = [str(r) for r in dfB.index if key_regex.search(str(r))]
    if cand_rows:
        cand_rows_sorted = sorted(cand_rows, key=lambda x: (0 if re.search(r"subtype", x, re.I) else 1, x))
        row_sub = cand_rows_sorted[0]
        series = dfB.loc[row_sub]
        series.index = [ _norm_tcga_15(c) for c in series.index ]
        return {idx: str(v) for idx, v in series.items() if pd.notna(v)}

    raise ValueError("未在临床矩阵中检测到 'subtype' 相关字段；请检查文件。")

def load_subtype_labels_from_clinical(clinical_path: str) -> Dict[str, str]:
    raw = _read_any_table(clinical_path)
    try:
        return _guess_orientation_and_extract_subtype(raw)
    except Exception:
        return _guess_orientation_and_extract_subtype(raw.T)

# 预处理：过滤/选择/切分/标准化
def filter_and_select_genes(expr_g_by_s: pd.DataFrame, min_nonzero_pct=0.2, topk_by_variance=2000) -> pd.DataFrame:
    n = expr_g_by_s.shape[1]
    nonzero_pct = (expr_g_by_s != 0).sum(axis=1) / n
    kept = expr_g_by_s.loc[nonzero_pct >= min_nonzero_pct]
    if kept.empty:
        raise ValueError("过滤后没有基因，请降低 min_nonzero_pct。")
    var = kept.var(axis=1)
    k = min(topk_by_variance, kept.shape[0])
    top_genes = var.sort_values(ascending=False).head(k).index
    return kept.loc[top_genes]

def stratified_split(X: pd.DataFrame, y: pd.Series, test_size=0.2, val_size=0.1, random_state=42):
    X_trv, X_te, y_trv, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    val_rel = val_size / (1 - test_size)
    X_tr, X_va, y_tr, y_va = train_test_split(X_trv, y_trv, test_size=val_rel, stratify=y_trv, random_state=random_state)
    return X_tr, X_va, X_te, y_tr, y_va, y_te

def zscore_fit_transform(Xtr: pd.DataFrame, Xva: pd.DataFrame, Xte: pd.DataFrame):
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(Xtr.values)
    Xtr = pd.DataFrame(scaler.transform(Xtr.values), index=Xtr.index, columns=Xtr.columns)
    Xva = pd.DataFrame(scaler.transform(Xva.values), index=Xva.index, columns=Xva.columns)
    Xte = pd.DataFrame(scaler.transform(Xte.values), index=Xte.index, columns=Xte.columns)
    return Xtr, Xva, Xte, scaler

def preprocess_pipeline_with_subtype(expr_path: str,
                                     clinical_path: str,
                                     min_nonzero_pct=0.2,
                                     topk_by_variance=2000,
                                     test_size=0.2,
                                     val_size=0.1,
                                     random_state=42,
                                     subtype_normalization: Optional[Dict[str, str]] = None):
    # 1) 表达矩阵：行=基因，列=样本
    expr = _read_expression_any(expr_path)
    expr.columns = [_norm_tcga_15(c) for c in expr.columns]

    # 2) 临床 -> subtype
    sub_map = load_subtype_labels_from_clinical(clinical_path)
    if subtype_normalization:
        sub_map = {k: subtype_normalization.get(str(v).strip(), str(v).strip()) for k, v in sub_map.items()}
    else:
        norm = {"classical": "Classical", "mesenchymal": "Mesenchymal",
                "proneural": "Proneural", "neural": "Neural"}
        sub_map = {k: norm.get(str(v).strip().lower(), str(v).strip()) for k, v in sub_map.items()}

    # 3) 对齐样本
    labeled_cols = [c for c in expr.columns if c in sub_map]
    if not labeled_cols:
        raise ValueError("表达矩阵与临床矩阵未找到可对齐的样本（前15位）。")
    expr = expr[labeled_cols]
    y = pd.Series([sub_map[c] for c in labeled_cols], index=labeled_cols, name="subtype")

    # 4) 过滤 + TopK方差
    expr = filter_and_select_genes(expr, min_nonzero_pct=min_nonzero_pct, topk_by_variance=topk_by_variance)

    # 5) 转置为 行=样本、列=基因
    X = expr.T.copy()

    # 6) 分层划分
    Xtr, Xva, Xte, ytr, yva, yte = stratified_split(X, y, test_size=test_size, val_size=val_size, random_state=random_state)

    # 7) z-score（仅训练集拟合）
    Xtr, Xva, Xte, scaler = zscore_fit_transform(Xtr, Xva, Xte)

    return {"X_train": Xtr, "X_val": Xva, "X_test": Xte,
            "y_train": ytr, "y_val": yva, "y_test": yte,
            "genes": list(X.columns), "scaler": scaler}

# ========== 训练与评估 ==========
def train_and_eval_multiclass(Xtr, ytr, Xva, yva, Xte, yte):
    print("\n=== 多分类：Logistic Regression (saga, L2, class_weight=balanced) ===")
    clf = LogisticRegression(max_iter=5000, solver="saga", class_weight="balanced")
    clf.fit(Xtr, ytr)

    def _report(split, X, y):
        pred = clf.predict(X)
        print(f"\n[{split}] acc:", accuracy_score(y, pred))
        print(f"[{split}] macro-F1:", f1_score(y, pred, average="macro"))
        print(f"\n[{split}] per-class report:\n{classification_report(y, pred)}")
        print(f"[{split}] confusion matrix:\n{confusion_matrix(y, pred)}")

    _report("VAL", Xva, yva)
    _report("TEST", Xte, yte)
    return clf

def train_and_eval_binary(Xtr, ytr, Xva, yva, Xte, yte, keep: set):
    # 子集化到两类
    def _subset(X, y):
        m = y.isin(keep)
        return X.loc[m], y.loc[m]
    Xtr2, ytr2 = _subset(Xtr, ytr)
    Xva2, yva2 = _subset(Xva, yva)
    Xte2, yte2 = _subset(Xte, yte)

    print(f"\n=== 二分类：Linear SVM (classes={sorted(keep)}) ===")
    clf = LinearSVC(class_weight="balanced")
    clf.fit(Xtr2, ytr2)

    def _report(split, X, y):
        pred = clf.predict(X)
        print(f"\n[{split}] acc:", accuracy_score(y, pred))
        print(f"[{split}] macro-F1:", f1_score(y, pred, average="macro"))
        print(f"\n[{split}] report:\n{classification_report(y, pred)}")
        print(f"[{split}] confusion matrix:\n{confusion_matrix(y, pred)}")

    _report("VAL", Xva2, yva2)
    _report("TEST", Xte2, yte2)
    return clf

#  主函数 
if __name__ == "__main__":

    EXPR_PATH = r"D:/毕设数据/HiSeqV2.xlsx"  
    CLIN_PATH = r"D:/毕设数据/TCGA.GBM.sampleMap_GBM_clinicalMatrix.xlsx"

    # 预处理参数（可调）
    MIN_NONZERO_PCT = 0.2
    TOPK_BY_VAR = 2000
    TEST_SIZE = 0.1
    VAL_SIZE = 0.1
    SEED = 42

    # 运行预处理
    data = preprocess_pipeline_with_subtype(
        expr_path=EXPR_PATH,
        clinical_path=CLIN_PATH,
        min_nonzero_pct=MIN_NONZERO_PCT,
        topk_by_variance=TOPK_BY_VAR,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        random_state=SEED
    )
    Xtr, Xva, Xte = data["X_train"], data["X_val"], data["X_test"]
    ytr, yva, yte = data["y_train"], data["y_val"], data["y_test"]

    print("Shapes:",
          "\n  X_train:", Xtr.shape,
          "\n  X_val:  ", Xva.shape,
          "\n  X_test: ", Xte.shape)
    print("Classes in y_train:", sorted(ytr.unique()))

    #  多分类（按subtype全部类别）
    clf_multi = train_and_eval_multiclass(Xtr, ytr, Xva, yva, Xte, yte)

    # 二分类：只挑两类，例如 Classical vs Mesenchymal
    # keep_two = {"Classical", "Mesenchymal"}
    # clf_bin = train_and_eval_binary(Xtr, ytr, Xva, yva, Xte, yte, keep=keep_two)
