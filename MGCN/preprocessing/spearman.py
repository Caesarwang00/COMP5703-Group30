#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 CNV(GISTIC by_genes) 构建“样本-样本相似图”，输出与原 edge_list.csv 完全同口径：
- 相关：Spearman rho
- 阈值：tau（默认 0.65）
- ID：保留 14 位条码（不截断为 12 位）
- 输出：MGCN/data_RNA/edge_list.csv（列：u,v,rho,pval），边为上三角(i<j)顺序
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# 路径约定（零参数可跑）
ROOT = Path(__file__).resolve().parents[2]                              # .../COMP5703-Group30
CNV  = ROOT / "data_RNA" / "CNV"
OUTD = ROOT / "MGCN" / "data_RNA"
OUTD.mkdir(parents=True, exist_ok=True)
OUTF = OUTD / "edge_list.csv"

TAU = 0.60  # 若你原始图更稠/更稀，可把这里改成当时的阈值

def load_cnv_by_genes_keep14(path: Path) -> pd.DataFrame:
    """
    读取 GISTIC thresholded by genes：返回 (genes x samples)。
    - 自动识别“基因名列”（优先匹配含 'gene' 的列，否则取首列）
    - 去掉说明/注释列（如 Cytoband/Locus ID 等）
    - **不做 12 位截断**，保留原始列名（通常是 14 位条码）
    """
    df = pd.read_csv(path, sep=None, engine="python")
    # 基因名列
    gene_col = df.columns[0]
    for c in df.columns[:3]:
        if 'gene' in c.lower():
            gene_col = c
            break
    df = df.set_index(gene_col)

    # 丢掉明显的注释列
    drop_cols = [c for c in df.columns if c.lower() in ("cytoband", "gene id", "locus id")]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    # 如果文件里包含非样本列，优先仅保留以 TCGA- 开头的列；若都不是，则保留除了索引外的所有列
    sample_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("TCGA-")]
    if sample_cols:
        df = df[sample_cols]

    # 数值化
    df = df.apply(pd.to_numeric, errors="coerce").astype(float)
    # GISTIC 阈值矩阵通常没有缺失；若有极少量缺失，Spearman 不接受 NaN，可用 0 填充避免偏差最小
    if df.isna().values.any():
        df = df.fillna(0.0)

    # 统一列名去空格（但**不截断**）
    df.columns = [str(c).strip() for c in df.columns]
    return df

def main():
    if not CNV.exists():
        raise FileNotFoundError(f"未找到 CNV 文件：{CNV}")
    cnv = load_cnv_by_genes_keep14(CNV)
    # 转为样本×基因
    M = cnv.T.copy()  # (n_samples x n_genes)
    samples = M.index.to_list()
    n = len(samples)
    if n < 3:
        raise RuntimeError("样本过少，无法构图。")
    print(f"[CNV] genes={M.shape[1]} samples={n}")

    # 一次性计算 Spearman rho/pval（按样本维度）
    rho, pval = spearmanr(M.values, axis=1)
    rho, pval = rho[:n, :n], pval[:n, :n]
    # 主对角置零（无自环）；p 值对角置 1
    np.fill_diagonal(rho, 0.0)
    np.fill_diagonal(pval, 1.0)

    # 取上三角(i<j)且 rho>=TAU 的边，保持上三角自然顺序（与你示例一致）
    iu, ju = np.triu_indices(n, k=1)
    keep = rho[iu, ju] >= TAU
    u_idx, v_idx = iu[keep], ju[keep]

    edges = pd.DataFrame({
        "u":    [samples[i] for i in u_idx],     # 14 位条码，不截断
        "v":    [samples[j] for j in v_idx],
        "rho":  rho[iu, ju][keep],
        "pval": pval[iu, ju][keep],
    })

    edges.to_csv(OUTF, index=False)
    print(f"[OK] nodes={n} edges={len(edges)}  -> {OUTF}")

if __name__ == "__main__":
    main()
