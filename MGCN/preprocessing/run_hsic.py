# -*- coding: utf-8 -*-
"""
run_hsic.py
对 rnaseq_clean.tsv 的“所有基因”进行特征筛选：
1) 优先使用 pyHSICLasso（HSIC Lasso，监督式）
2) 若未安装，则自动回退到 sklearn 的 mutual_info_classif

输入文件（与脚本既有相对路径保持不变）：
- ../../data_RNA/rnaseq_clean.tsv        行=基因（第一列为基因名），列=样本ID
- ../data_RNA/subtype_labels.tsv         两列：sampleID, GeneExp_Subtype

输出文件（统一写到 MGCN/data_RNA/）：
- hsic_all_genes.tsv / hsic_top200.tsv     （HSIC 模式）
- mi_all_genes.tsv   / mi_top200.tsv       （回退模式）
- aligned_samples.tsv                       （可选：如需记录对齐样本，见下方注释行）
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# === 统一输出目录：…/MGCN/data_RNA ===
OUT = Path(__file__).resolve().parents[1] / "data_RNA"
OUT.mkdir(parents=True, exist_ok=True)

# 尝试导入 HSICLasso；失败则使用 sklearn 回退
USE_HSIC = True
try:
    from pyHSICLasso import HSICLasso  # pip install git+https://github.com/riken-aip/pyHSICLasso.git
except Exception as e:
    print("⚠️  未检测到 pyHSICLasso，将回退为 sklearn 的 mutual_info 特征选择。")
    print(f"   详情：{repr(e)}")
    USE_HSIC = False

# ---------- 1) 读入表达矩阵 ----------
expr_path = "../../data/CNV"
if not os.path.exists(expr_path):
    sys.exit(f"❌ 未找到表达矩阵文件：{expr_path}")

expr_df = pd.read_csv(expr_path, sep="\t")
if expr_df.shape[1] < 2:
    sys.exit("❌ rnaseq_clean.tsv 列数异常，至少需要 2 列（基因名 + 样本）。")

# 第一列作为基因名索引
expr_df = expr_df.set_index(expr_df.columns[0])
genes = expr_df.index.to_numpy()

# ---------- 2) 读入标签 ----------
labels_path = "../data/subtype_labels.tsv"
if not os.path.exists(labels_path):
    sys.exit(f"❌ 未找到标签文件：{labels_path}（请先生成 sampleID 与 GeneExp_Subtype 对照）")

labels_df = pd.read_csv(labels_path, sep="\t")
# 容错：大小写/空白
labels_df.columns = [c.strip() for c in labels_df.columns]
needed = {"sampleID", "GeneExp_Subtype"}
if not needed.issubset(set(labels_df.columns)):
    sys.exit("❌ subtype_labels.tsv 需要包含列：sampleID, GeneExp_Subtype")

# ---------- 3) 对齐样本 ----------
sample_ids = labels_df["sampleID"].astype(str)
common = expr_df.columns.intersection(sample_ids)
if len(common) < 5:
    sys.exit(f"❌ 可对齐的样本太少（{len(common)}）。请检查两边样本ID是否一致。")

# # 如需记录最终参与计算的样本，请取消注释（输出到 MGCN/data_RNA）
# pd.DataFrame({"sampleID": common}).to_csv(OUT / "aligned_samples.tsv", sep="\t", index=False)

# 构造 X, y
X = expr_df[common].T.values            # (n_samples, n_genes)
y = labels_df.set_index("sampleID").loc[common, "GeneExp_Subtype"].astype(str).values

print(f"✅ 数据就绪：样本数={X.shape[0]}，基因数={X.shape[1]}")

# ---------- 4) 运行特征选择 ----------
if USE_HSIC:
    print("🚀 使用 HSIC Lasso（pyHSICLasso）对所有基因打分…")
    try:
        import numpy as np
        import pandas as pd

        hsic = HSICLasso()
        hsic.input(X, y, n_jobs=-1)               # 多核
        ret = hsic.classification(X.shape[1])     # K=所有基因；有些版本会把索引作为返回值给你

        # === 兼容垫片：尽一切可能拿到排序（order） ===
        order = None

        # 1) 直接属性 .order
        if hasattr(hsic, "order"):
            try:
                order = np.array(hsic.order, dtype=int)
            except Exception:
                order = None

        # 2) 方法 get_order()
        if order is None and hasattr(hsic, "get_order"):
            try:
                order = np.array(hsic.get_order(), dtype=int)
            except Exception:
                order = None

        # 3) 方法/属性：get_index() / selected / index
        if order is None and hasattr(hsic, "get_index"):
            try:
                order = np.array(hsic.get_index(), dtype=int)
            except Exception:
                order = None
        if order is None and hasattr(hsic, "selected"):
            try:
                order = np.array(hsic.selected, dtype=int)
            except Exception:
                order = None
        if order is None and hasattr(hsic, "index"):
            try:
                order = np.array(hsic.index, dtype=int)
            except Exception:
                order = None

        # 4) 其它命名：ranking / get_ranking()
        if order is None and hasattr(hsic, "ranking"):
            try:
                order = np.array(hsic.ranking, dtype=int)
            except Exception:
                order = None
        if order is None and hasattr(hsic, "get_ranking"):
            try:
                order = np.array(hsic.get_ranking(), dtype=int)
            except Exception:
                order = None

        # 5) 直接用 classification(...) 的返回值（有些版本返回索引或 (idx,score)）
        if order is None and ret is not None:
            try:
                if isinstance(ret, (list, tuple, np.ndarray)):
                    if len(ret) == 2 and all(hasattr(ret[i], "__len__") for i in (0, 1)):
                        order = np.array(ret[0], dtype=int)
                    else:
                        order = np.array(ret, dtype=int)
            except Exception:
                order = None

        if order is None:
            raise AttributeError("HSICLasso: cannot obtain feature order from this version/API")

        # === 分数（尽力对齐；没有也不影响 topK 名单） ===
        relevance = None
        if hasattr(hsic, "relevance"):
            try:
                relevance = np.array(hsic.relevance, dtype=float)
            except Exception:
                relevance = None
        if relevance is None and hasattr(hsic, "get_index_score"):
            try:
                idx_score = hsic.get_index_score()
                if isinstance(idx_score, tuple) and len(idx_score) == 2:
                    idxs, scores = idx_score
                    m = {int(i): float(s) for i, s in zip(idxs, scores)}
                    relevance = np.array([m.get(int(i), np.nan) for i in order], dtype=float)
                elif isinstance(idx_score, list):
                    m = {int(i): float(s) for i, s in idx_score}
                    relevance = np.array([m.get(int(i), np.nan) for i in order], dtype=float)
            except Exception:
                relevance = None
        if relevance is None and hasattr(hsic, "score"):
            try:
                sc = np.array(hsic.score, dtype=float)
                if sc.shape[0] == len(order):
                    relevance = sc
            except Exception:
                relevance = None

        # === 输出到 MGCN/data_RNA ===
        genes_arr = np.array(genes)
        out = pd.DataFrame({"gene": genes_arr[order]})
        if relevance is not None:
            out["score"] = relevance[:len(out)]
        # out.to_csv(OUT / "hsic_all_genes.tsv", sep="\t", index=False)
        out.head(200).to_csv(OUT / "hsic_top200.tsv", sep="\t", index=False)

        print(f"✅ 完成 HSIC：{OUT/'hsic_all_genes.tsv'}（全基因打分），{OUT/'hsic_top200.tsv'}（前200基因）。")

    except Exception as e:
        print("⚠️ HSIC 运行失败，将自动回退为 sklearn 互信息方法。")
        print(f"   详情：{repr(e)}")
        USE_HSIC = False

if not USE_HSIC:
    print("🚀 使用 sklearn 的 mutual_info_classif 回退方案对所有基因打分…")
    try:
        from sklearn.feature_selection import mutual_info_classif
        # 将字符串标签编码为整数
        y_codes = pd.Categorical(y).codes
        mi_scores = mutual_info_classif(X, y_codes, discrete_features=False, random_state=0)
        out = pd.DataFrame({"gene": genes, "score": mi_scores}).sort_values("score", ascending=False)

        # === 输出到 MGCN/data_RNA ===
        out.to_csv(OUT / "mi_all_genes.tsv", sep="\t", index=False)
        out.head(200).to_csv(OUT / "mi_top200.tsv", sep="\t", index=False)
        print(f"✅ 完成 MI：{OUT/'mi_all_genes.tsv'}（全基因打分），{OUT/'mi_top200.tsv'}（前200基因）。")
    except Exception as e:
        sys.exit(f"❌ 回退方案也失败：{repr(e)}")

print("🎉 全流程完成。你现在可以用结果做后续构图 / 融合 / 建模了。")
