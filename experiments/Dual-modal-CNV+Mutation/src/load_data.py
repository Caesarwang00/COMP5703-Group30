
# -*- coding: utf-8 -*-
"""
数据加载与模态对齐
- 读取 CNV thresholded、突变基因矩阵、临床标签（GeneExp_Subtype）
- 过滤原发瘤样本 (sample_type == "primary tumor")（如果存在该列）
- 返回：cnv_df (genes x samples), mut_df (genes x samples), y_series (samples -> label), sample_order
"""
import os
import pandas as pd

try:
    import config
except Exception:
    from . import config

# ==== 常见文件名 ====
CNV_CANDIDATES = [
    "CNV",
    "gistic2_all_thresholded.by_genes.txt",
]
MUT_CANDIDATES = [
    "MUT",
    "Mut_MC3_gene_level.txt",
]
CLINICAL_CANDIDATES = [
    "GBM_clinicalMatrix",
    "phenotype - Curated survival data_RNA.txt",
]

def _find_file(dirpath, names):
    for n in names:
        p = os.path.join(dirpath, n)
        if os.path.exists(p):
            return p
    # fallback: 部分平台会去掉空格或扩展名不同，尝试模糊匹配
    files = os.listdir(dirpath)
    for n in names:
        for f in files:
            if n.replace(" ", "").split(".")[0].lower() in f.replace(" ", "").lower():
                return os.path.join(dirpath, f)
    return None

def short_barcode(x):
    """把 TCGA 条形码化为短码：'TCGA-XX-XXXX'"""
    if pd.isna(x): return None
    s = str(x).strip()
    parts = s.split("-")
    if len(parts) >= 3:
        return "-".join(parts[:3])
    return s

def _load_cnv(path):
    df = pd.read_csv(path, sep="\t", header=0, dtype=str)
    # 第一列 gene symbol
    gene_col = df.columns[0]
    df = df.set_index(gene_col)
    # 转 float
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(0.0)
    # 列名统一为短码（保留原始列同时去重）
    df.columns = [short_barcode(c) for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def _load_mut(path):
    df = pd.read_csv(path, sep="\t", header=0, dtype=str)
    gene_col = df.columns[0]
    df = df.set_index(gene_col)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df.columns = [short_barcode(c) for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def _load_clinical(path):
    # 支持两种格式：制表符或空格分隔
    try:
        df = pd.read_csv(path, sep="\t", header=0, dtype=str)
    except Exception:
        df = pd.read_csv(path, sep=r"\s+", header=0, dtype=str, engine="python")
    # 标准化列名
    df.columns = [c.strip() for c in df.columns]
    # 标签列
    label_col = None
    for c in ["GeneExp_Subtype", "geneexp_subtype", "gene_expression_subtype"]:
        if c in df.columns:
            label_col = c; break
    # sample 标识列
    cand_id = [c for c in df.columns if "sample" in c.lower() or "id" in c.lower() or "barcode" in c.lower()]
    sid_col = cand_id[0] if cand_id else df.columns[0]
    df["short"] = df[sid_col].map(short_barcode)
    # 过滤原发瘤（若存在 sample_type）
    st_col = None
    for c in df.columns:
        if c.lower().replace(" ", "") in ["sample_type", "sampletype"]:
            st_col = c; break
    if st_col is not None:
        mask_primary = df[st_col].str.lower().str.contains("primary", na=False)
        df = df.loc[mask_primary]
    # 保留标签
    if label_col is None:
        y = pd.Series(index=[], dtype=str)
    else:
        y = df.set_index("short")[label_col].dropna().astype(str)
    return y

def align_modalities():
    data_dir = config.DATA_DIR
    cnv_path = _find_file(data_dir, CNV_CANDIDATES)
    mut_path = _find_file(data_dir, MUT_CANDIDATES)
    clin_path = _find_file(data_dir, CLINICAL_CANDIDATES)
    if cnv_path is None or mut_path is None:
        raise FileNotFoundError(f"未在 {data_dir} 找到 CNV/Mutation 文件，请检查文件名")

    cnv = _load_cnv(cnv_path)
    mut = _load_mut(mut_path)
    y = _load_clinical(clin_path) if clin_path is not None else pd.Series(dtype=str)

    # 交集（样本短码）
    samples = sorted(set(cnv.columns) & set(mut.columns))
    if len(y) > 0:
        samples = [s for s in samples if s in y.index]
    if not samples:
        raise RuntimeError("CNV 与 Mutation 与 标签的交集为空，请检查样本命名是否一致")

    # 子集化
    cnv_sub = cnv.loc[:, samples]
    mut_sub = mut.loc[:, samples]
    y_series = y.loc[samples] if len(y) > 0 else pd.Series(index=samples, dtype=str)

    return cnv_sub, mut_sub, y_series, samples
