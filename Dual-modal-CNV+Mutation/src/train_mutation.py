#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mutation-only MULTICLASS training for TCGA-GBM (single modality).
- 支持 --model logreg 或 --model rf
- logreg: ElasticNet Logistic 回归 + chi2 特征选择
- rf: RandomForestClassifier + chi2 特征选择
"""

import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import MaxAbsScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer, accuracy_score, balanced_accuracy_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

# ---------------------- Utils ----------------------
def norm_barcode(s: str) -> str:
    s = str(s)
    return s[:12] if s.startswith("TCGA-") else s

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
    if "Variant_Classification" not in maf.columns or "Hugo_Symbol" not in maf.columns:
        raise ValueError("MAF 需包含 'Variant_Classification' 与 'Hugo_Symbol'")
    maf = maf[maf["Variant_Classification"].isin(keep)].copy()
    maf["sampleID"] = maf["Tumor_Sample_Barcode"].map(lambda s: str(s)[:12])
    maf["val"] = 1
    X = (maf[["sampleID","Hugo_Symbol","val"]]
            .drop_duplicates()
            .pivot_table(index="sampleID", columns="Hugo_Symbol", values="val", fill_value=0)
            .astype(np.uint8))
    return X

# --------- Transformers ----------
class PrevalenceFilter(BaseEstimator, TransformerMixin):
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
    clin_path = data_dir / "TCGA.GBM.sampleMap_GBM_clinicalMatrix"
    if not clin_path.exists():
        raise FileNotFoundError(f"未找到临床文件: {clin_path}")

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

    # 特征
    Xmat = data.drop(columns=["sampleID", args.label_col]).astype(np.uint8)
    chi2_k = min(args.chi2_k if args.chi2_k else Xmat.shape[1], Xmat.shape[1])

    # 模型选择
    if args.model == "logreg":
        clf = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            class_weight="balanced",
            multi_class="multinomial",
            max_iter=8000,
            random_state=args.seed
        )
        pipe = make_pipeline(
            PrevalenceFilter(min_mut_samples=args.min_mut_samples),
            SelectKBest(score_func=chi2, k=chi2_k),
            MaxAbsScaler(),
            clf
        )
        param_grid = {
            "logisticregression__C": [0.1, 0.5, 1.0, 2.0, 5.0],
            "logisticregression__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        }
    elif args.model == "rf":
        clf = RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=-1
        )
        pipe = make_pipeline(
            PrevalenceFilter(min_mut_samples=args.min_mut_samples),
            SelectKBest(score_func=chi2, k=chi2_k),
            clf
        )
        param_grid = {
            "randomforestclassifier__max_depth": [None, 10, 20, 30],
            "randomforestclassifier__max_features": ["sqrt", "log2", None],
        }
    else:
        raise ValueError("--model 必须是 logreg 或 rf")

    # CV
    counts = np.bincount(y)
    min_cls = counts[counts > 0].min()
    k = min(args.folds, max(2, int(min_cls)))
    outer_cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=args.seed)

    gs = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=3,
        scoring=make_scorer(f1_score, average="macro", zero_division=0),
        n_jobs=-1,
        refit=True
    )

    scoring = {
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
        "accuracy": make_scorer(accuracy_score),
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
    }

    res = cross_validate(gs, Xmat, y, cv=outer_cv, scoring=scoring, error_score=0)
    metrics = {k.replace("test_", ""): float(np.nanmean(v)) for k, v in res.items() if k.startswith("test_")}
    print("[RESULT]", json.dumps(metrics, indent=2, ensure_ascii=False))

    # 最优模型再拟合
    gs.fit(Xmat, y)
    best_params = gs.best_params_
    print("[BEST_PARAMS]", json.dumps(best_params, indent=2, ensure_ascii=False))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(gs.best_estimator_, out_dir/"model.pkl")

    with open(out_dir/"cv_metrics.json","w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(out_dir/"label_mapping.json","w") as f:
        json.dump(label_mapping, f, indent=2, ensure_ascii=False)
    with open(out_dir/"best_params.json","w") as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)

    print(f"[SAVED] model={out_dir/'model.pkl'}  metrics=cv_metrics.json  label_map=label_mapping.json")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--label_col", type=str, default="GeneExp_Subtype")
    p.add_argument("--gene_level_file", type=str, default="GBM_mc3_gene_level.txt")
    p.add_argument("--maf_file", type=str, default=None)
    p.add_argument("--min_mut_samples", type=int, default=3)
    p.add_argument("--chi2_k", type=int, default=1000)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="out_mut_subtype")
    p.add_argument("--model", type=str, default="logreg", choices=["logreg","rf"], help="选择模型: logreg 或 rf")
    args = p.parse_args()
    main(args)
