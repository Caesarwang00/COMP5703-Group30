# -*- coding: utf-8 -*-
"""
融合三模态：RNA + CNV + SNV
用法示例：
  python fuse_multiomics_snv.py --cnv "Gistic2_CopyNumber_by_genes.tsv" --snv "GBM_mc3.one_level.txt"
注意：若文件名有空格，请用引号括起来。
"""

import os, sys, argparse, numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# 输出目录：…/MGCN/data_RNA
OUT = Path(__file__).resolve().parents[1] / "data_RNA"
OUT.mkdir(parents=True, exist_ok=True)

def read_matrix(path):
    # 自动分隔符检测：优先 tab，否则逗号
    try:
        df = pd.read_csv(path, sep="\t")
        if df.shape[1] == 1:
            df = pd.read_csv(path, sep=",")
    except Exception:
        df = pd.read_csv(path, sep=",")
    df = df.set_index(df.columns[0])   # 第一列为基因名
    return df

def to_samples_by_features(df):
    # 输入：基因×样本；输出：样本×基因
    return df.T.copy()

def safe_pca(df, n_components=50, prefix="X", treat_as_binary=False):
    X = df.values
    if not treat_as_binary:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X = scaler.fit_transform(X)
    # 组件数不超过 min(样本-1, 特征)
    k = max(1, min(X.shape[0]-1, X.shape[1], n_components))
    pca = PCA(n_components=k, svd_solver="auto", random_state=0)
    Z = pca.fit_transform(X)
    return pd.DataFrame(Z, index=df.index, columns=[f"{prefix}_{i}" for i in range(Z.shape[1])])

def maybe_filter_rna_by_topk(rna_gene_by_sample, topk_file="mi_top200.tsv"):
    if os.path.exists(topk_file):
        top = pd.read_csv(topk_file, sep="\t")
        keep = rna_gene_by_sample.index.intersection(top["gene"])
        if len(keep) >= 50:
            print(f"🧪 使用 {topk_file} 过滤 RNA 基因，保留 {len(keep)} 个。")
            return rna_gene_by_sample.loc[keep]
        else:
            print("⚠️ 过滤后基因过少，忽略过滤，使用全部 RNA 基因。")
    return rna_gene_by_sample

def main():
    from pathlib import Path  # 只需在文件顶部或这段前面加这行

    # 以脚本为基准定位到项目根目录 …/COMP5703-Group30
    ROOT = Path(__file__).resolve().parents[2]
    DATA = ROOT / "data_RNA"

    ap = argparse.ArgumentParser()
    ap.add_argument("--rna",
                    default=str((DATA / "rnaseq_clean.tsv" ).resolve()),
                    help="RNA 基因×样本矩阵（默认 data_RNA/RNA/rnaseq_clean.tsv）")
    ap.add_argument("--cnv",
                    default=str((DATA / "CNV" ).resolve()),
                    help="CNV 基因×样本矩阵（GISTIC by_genes）")
    ap.add_argument("--snv",
                    default=str((DATA / "MUT").resolve()),
                    help="SNV 基因×样本矩阵（0/1 或 -1/0/1）")
    ap.add_argument("--k", type=int, default=100,
                    help="每模态降维后的维度上限（默认50）")
    args = ap.parse_args()

    # （可选）打印一下实际使用的路径，便于排错
    print("[I] RNA:", args.rna)
    print("[I] CNV:", args.cnv)
    print("[I] SNV:", args.snv)
    print("[I] k  :", args.k)

    # 读取三模态（基因×样本）
    if not os.path.exists(args.rna): sys.exit(f"❌ 找不到 RNA：{args.rna}")
    if not os.path.exists(args.cnv): sys.exit(f"❌ 找不到 CNV：{args.cnv}")
    if not os.path.exists(args.snv): sys.exit(f"❌ 找不到 SNV：{args.snv}")

    rna_gx = read_matrix(args.rna)
    cnv_gx = read_matrix(args.cnv)
    snv_gx = read_matrix(args.snv)

    # SNV 统一成 0/1
    snv_gx = snv_gx.apply(pd.to_numeric, errors="coerce").fillna(0)
    snv_gx = (snv_gx.values > 0).astype(np.int8)
    snv_gx = pd.DataFrame(snv_gx, index=read_matrix(args.snv).index, columns=read_matrix(args.snv).columns)

    # # 可选：RNA 用 top200 过滤
    # rna_gx = maybe_filter_rna_by_topk(rna_gx, "../data_RNA/hsic_top200.tsv")

    # 转成 样本×基因
    rna = to_samples_by_features(rna_gx)
    cnv = to_samples_by_features(cnv_gx)
    snv = to_samples_by_features(snv_gx)

    # 对齐样本
    common = rna.index.intersection(cnv.index).intersection(snv.index)
    if len(common) < 20:
        sys.exit(f"❌ 三模态交集样本过少：{len(common)}。请检查样本ID一致性。")
    rna, cnv, snv = rna.loc[common], cnv.loc[common], snv.loc[common]
    print(f"✅ 对齐样本：{len(common)}")

    # 各模态降维
    k = args.k
    rna_z = safe_pca(rna, n_components=k, prefix="RNA", treat_as_binary=False)
    cnv_z = safe_pca(cnv, n_components=k, prefix="CNV", treat_as_binary=False)
    snv_z = safe_pca(snv.astype(np.float32), n_components=k, prefix="SNV", treat_as_binary=True)

    # # 保存各模态潜在表示到 MGCN/data_RNA
    # rna_z.to_csv(OUT / "rna_latent.tsv", sep="\t")
    # cnv_z.to_csv(OUT / "cnv_latent.tsv", sep="\t")
    # snv_z.to_csv(OUT / "snv_latent.tsv", sep="\t")

    # 融合并保存到 MGCN/data_RNA
    fused = pd.concat([rna_z, cnv_z, snv_z], axis=1)
    fused.to_csv(OUT / "multiomics_fused_features.tsv", sep="\t")
    print(f"🎉 完成融合：样本={fused.shape[0]}，维度={fused.shape[1]} → {OUT / 'multiomics_fused_features.tsv'}")

if __name__ == "__main__":
    main()
