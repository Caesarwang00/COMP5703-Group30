#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mutation-only MULTICLASS training for TCGA-GBM (single modality) — fast tuned.

- 模型：logreg / rf / xgb（默认 xgb）
- 特征选择：SelectKBest，score_func ∈ {chi2, f_classif}
- k 自适应：不会超过实际特征数，候选默认 [200, 400, 800, 1200]
- 评估：外层分层CV + 多指标；OOF classification_report + 混淆矩阵
- 速度优化：较小网格、XGB n_jobs=1、GridSearchCV(verbose=2)
"""

import argparse, json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV, cross_val_predict
from sklearn.preprocessing import MaxAbsScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, make_scorer, accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix
)
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

# ---- 安静掉 k>n_features 的警告刷屏 ----
warnings.filterwarnings("ignore", message=".*k is greater than n_features.*")

# ---- 可选：XGBoost ----
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

# ---- JSON 序列化修复：函数/NumPy 标量等 ----
def _json_default(o):
    import numpy as _np
    if callable(o):
        return getattr(o, "__name__", str(o))
    if isinstance(o, (_np.integer,)):
        return int(o)
    if isinstance(o, (_np.floating,)):
        return float(o)
    if isinstance(o, (_np.bool_,)):
        return bool(o)
    return str(o)

# ---------------------- Utils ----------------------
def norm_barcode(s: str) -> str:
    s = str(s)
    return s[:12] if s.startswith("TCGA-") else s

def _find_clinical_file(data_dir: Path) -> Path:
    candidates = [
        data_dir / "TCGA.GBM.sampleMap_GBM_clinicalMatrix",
        data_dir / "TCGA.GBM.sampleMap_GBM_clinicalMatrix.txt",
        data_dir / "TCGA.GBM.sampleMap_GBM_clinicalMatrix.tsv",
    ]
    for p in candidates:
        if p.exists():
            return p
    matches = list(data_dir.glob("*clinicalMatrix*"))
    hint = f"。可选候选：{[str(m) for m in matches]}" if matches else ""
    raise FileNotFoundError("未找到临床文件，期望文件名接近 'TCGA.GBM.sampleMap_GBM_clinicalMatrix'" + hint)

def load_clinical(clin_path: Path, label_col: str) -> pd.DataFrame:
    clin = pd.read_csv(clin_path, sep="\t", low_memory=False)
    if "sampleID" not in clin.columns:
        raise ValueError("clinicalMatrix 缺少 'sampleID' 列")
    clin["sampleID"] = clin["sampleID"].map(norm_barcode)
    if label_col not in clin.columns:
        raise ValueError(f"Label 列 '{label_col}' 不在临床文件中")
    ydf = clin[["sampleID", label_col]].dropna().copy()
    return ydf

def load_mut_genelevel(path: Path) -> pd.DataFrame:
    mat = pd.read_csv(path, sep="\t", low_memory=False)
    gene_col = next((c for c in ["Hugo_Symbol","Gene","Gene Symbol","gene","symbol"] if c in mat.columns), mat.columns[0])
    mat = mat.rename(columns={gene_col:"Gene"}).set_index("Gene")
    bin_mat = (mat.fillna(0).astype(float) >= 1).astype(np.uint8)
    bin_mat.columns = [c[:12] if str(c).startswith("TCGA-") else str(c) for c in bin_mat.columns]
    X = bin_mat.T
    X.index.name = "sampleID"
    return X

def load_mut_from_maf(path: Path) -> pd.DataFrame:
    keep = {
        "Missense_Mutation","Nonsense_Mutation","Frame_Shift_Del","Frame_Shift_Ins",
        "Splice_Site","In_Frame_Del","In_Frame_Ins","Nonstop_Mutation","Translation_Start_Site"
    }
    maf = pd.read_csv(path, sep="\t", low_memory=False, compression="infer")
    req = {"Variant_Classification","Hugo_Symbol","Tumor_Sample_Barcode"}
    if not req.issubset(set(maf.columns)):
        raise ValueError("MAF 需包含 'Variant_Classification','Hugo_Symbol','Tumor_Sample_Barcode'")
    maf = maf[maf["Variant_Classification"].isin(keep)].copy()
    maf["sampleID"] = maf["Tumor_Sample_Barcode"].map(lambda s: str(s)[:12])
    maf["val"] = 1
    X = (maf[["sampleID","Hugo_Symbol","val"]]
            .drop_duplicates()
            .pivot_table(index="sampleID", columns="Hugo_Symbol", values="val", fill_value=0)
            .astype(np.uint8))
    return X

# --------- Transformer ----------
class PrevalenceFilter(BaseEstimator, TransformerMixin):
    """仅保留在 [min_mut_samples, n-min_mut_samples] 区间内出现的基因列"""
    def __init__(self, min_mut_samples=3):
        self.min_mut_samples = int(min_mut_samples)
        self.keep_cols_ = None
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            col_sums = X.sum(axis=0).astype(int)
            n = X.shape[0]
            keep = (col_sums >= self.min_mut_samples) & (col_sums <= (n - self.min_mut_samples))
            if keep.sum() == 0:
                keep.iloc[col_sums.sort_values(ascending=False).index[:1]] = True
            self.keep_cols_ = col_sums.index[keep].tolist()
        return self
    def transform(self, X):
        if self.keep_cols_ is None:
            return X
        return X[self.keep_cols_]

# ---------------------- Main ----------------------
def main(args):
    data_dir = Path(args.data_dir)
    clin_path = _find_clinical_file(data_dir)

    # 临床标签
    ydf = load_clinical(clin_path, args.label_col)
    uniq = ydf[args.label_col].astype(str).unique()
    if len(uniq) < 3:
        raise ValueError(f"'{args.label_col}' 只有 {len(uniq)} 个类别，至少3类才支持。")

    # 突变矩阵
    if args.gene_level_file:
        gpath = Path(args.gene_level_file)
        if not gpath.exists():
            gpath = data_dir / args.gene_level_file
        Xraw = load_mut_genelevel(gpath)
    elif args.maf_file:
        mpath = Path(args.maf_file)
        if not mpath.exists():
            mpath = data_dir / args.maf_file
        Xraw = load_mut_from_maf(mpath)
    else:
        raise ValueError("请提供 --gene_level_file 或 --maf_file")

    # 对齐
    data = ydf.merge(Xraw.reset_index(), on="sampleID", how="inner")
    if data.shape[0] < 30:
        print("[WARN] 对齐后样本很少，请检查条形码是否一致", file=sys.stderr)

    # 标签编码
    le = LabelEncoder()
    y = le.fit_transform(data[args.label_col].astype(str))
    label_mapping = {int(i): cls for i, cls in enumerate(le.classes_)}

    # 特征矩阵（非负）
    Xmat = data.drop(columns=["sampleID", args.label_col]).astype(np.uint8)

    # ===== 自适应 k_grid + score_funcs（更快）=====
    n_features = Xmat.shape[1]
    default_ks = [200, 400, 800, 1200]
    k_grid = sorted(set([min(k, n_features) for k in default_ks if min(k, n_features) > 0]))
    if not k_grid:
        k_grid = [min(200, n_features)]
    score_funcs = [chi2, f_classif]

    # ===== Pipeline & Param Grid =====
    base_steps = [
        ("pf", PrevalenceFilter(min_mut_samples=args.min_mut_samples)),
        ("sel", SelectKBest(score_func=chi2, k=k_grid[0])),  # k/score_func 走网格
    ]

    if args.model == "logreg":
        clf = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            class_weight="balanced",
            multi_class="multinomial",
            max_iter=8000,
            random_state=args.seed
        )
        pipe = make_pipeline(*(s for _, s in base_steps), MaxAbsScaler(), clf)
        param_grid = {
            "selectkbest__k": k_grid,
            "selectkbest__score_func": score_funcs,
            "logisticregression__C": [0.5, 1.0, 2.0],
            "logisticregression__l1_ratio": [0.3, 0.5, 0.7],
        }

    elif args.model == "rf":
        clf = RandomForestClassifier(
            n_estimators=600,
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=1
        )
        pipe = make_pipeline(*(s for _, s in base_steps), clf)
        param_grid = {
            "selectkbest__k": k_grid,
            "selectkbest__score_func": score_funcs,
            "randomforestclassifier__max_depth": [None, 10, 20],
            "randomforestclassifier__max_features": ["sqrt", "log2"],
        }

    elif args.model == "xgb":
        if not _HAS_XGB:
            raise RuntimeError("未安装 xgboost，请先执行: pip install xgboost")
        clf = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=1,               # 让外层并行去占核
            random_state=args.seed
        )
        pipe = make_pipeline(*(s for _, s in base_steps), clf)
        param_grid = {
            "selectkbest__k": k_grid,
            "selectkbest__score_func": score_funcs,
            "xgbclassifier__n_estimators": [300, 500],
            "xgbclassifier__max_depth": [4, 6],
            "xgbclassifier__learning_rate": [0.03, 0.1],
            "xgbclassifier__subsample": [0.9],
            "xgbclassifier__colsample_bytree": [0.8],
        }
    else:
        raise ValueError("--model 必须是 logreg / rf / xgb")

    # ===== 外层CV =====
    counts = np.bincount(y)
    min_cls = counts[counts > 0].min()
    k = min(args.folds, max(2, int(min_cls)))
    outer_cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=args.seed)

    # ===== 内层GS（用于外层）=====
    gs = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=3,
        scoring=make_scorer(f1_score, average="macro", zero_division=0),
        n_jobs=-1,
        refit=True,
        verbose=2   # 展示进度
    )

    # ===== 外层CV多指标 =====
    scoring = {
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
        "accuracy": make_scorer(accuracy_score),
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
    }

    # 1) 外层CV指标
    res = cross_validate(gs, Xmat, y, cv=outer_cv, scoring=scoring, error_score=0)
    metrics = {k.replace("test_", ""): float(np.nanmean(v)) for k, v in res.items() if k.startswith("test_")}
    print("[RESULT]", json.dumps(metrics, indent=2, ensure_ascii=False))

    # 2) 外层CV OOF 预测 -> 分类报告
    y_oof = cross_val_predict(gs, Xmat, y, cv=outer_cv, n_jobs=-1, method="predict")
    target_names = list(le.classes_)
    report_txt = classification_report(
        y_true=y, y_pred=y_oof,
        target_names=target_names, digits=2, zero_division=0
    )
    print("\n[CLASSIFICATION_REPORT]\n" + report_txt)

    cm = confusion_matrix(y, y_oof)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    print("[CONFUSION_MATRIX]\n", cm_df)

    # 3) 拟合全量数据，得到最终最佳模型
    gs.fit(Xmat, y)
    best_params = gs.best_params_
    best_payload = {
        "best_score_macro_f1_cv3": float(gs.best_score_),
        "best_params": best_params
    }
    print("[BEST]", json.dumps(best_payload, indent=2, ensure_ascii=False, default=_json_default))

    # 4) 保存输出
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(gs.best_estimator_, out_dir/"model.pkl")

    with open(out_dir/"cv_metrics.json","w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(out_dir/"label_mapping.json","w") as f:
        json.dump(label_mapping, f, indent=2, ensure_ascii=False)
    with open(out_dir/"best_params.json","w") as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False, default=_json_default)
    with open(out_dir/"best_summary.json","w") as f:
        json.dump(best_payload, f, indent=2, ensure_ascii=False, default=_json_default)
    with open(out_dir/"classification_report.txt","w", encoding="utf-8") as f:
        f.write(report_txt)
    cm_df.to_csv(out_dir/"confusion_matrix.csv")

    print(f"[SAVED] model={out_dir/'model.pkl'}  "
          f"metrics=cv_metrics.json  label_map=label_mapping.json  "
          f"report=classification_report.txt  cm=confusion_matrix.csv  "
          f"best=best_summary.json")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--label_col", type=str, default="GeneExp_Subtype")
    p.add_argument("--gene_level_file", type=str, default="GBM_mc3_gene_level.txt")
    p.add_argument("--maf_file", type=str, default=None)
    p.add_argument("--min_mut_samples", type=int, default=3)
    p.add_argument("--chi2_k", type=int, default=1000)      # 初始占位；真实 k 由网格决定
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="out_mut_subtype")
    p.add_argument("--model", type=str, default="xgb",
                   choices=["logreg","rf","xgb"], help="选择模型: logreg / rf / xgb")
    args = p.parse_args()
    main(args)

