# -*- coding: utf-8 -*-
"""
数据加载与对齐：CNV + Methyl450 + RNAseq + 标签（原发）
- 自动寻找常见文件名；
- 自动识别矩阵朝向（样本在行/列）并转置；
- 自动尝试 3段 与 4段 条形码截断（择优交集更大的方案）；
- 若临床含 sample_type，则过滤 primary tumor；
- 详细调试日志（由 config.DEBUG_LOG 控制）。
"""
import os, re, pandas as pd, numpy as np

try:
    from . import config
except Exception:
    import config

# 常见文件名候选
CNV_CANDIDATES = [
    "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes",
    "gistic2_all_thresholded.by_genes.txt",
]
METH_CANDIDATES = [
    "HumanMethylation450",
    "methyl450",
    "Methylation450",
    "TCGA.GBM.sampleMap_HumanMethylation450",
]
RNA_CANDIDATES = [
    "TCGA.GBM.sampleMap_HiSeqV2",
    "TCGA.GBM.sampleMap_HiSeq",
    "rnaseq",
    "RNAseq",
    "RNAseq_gene_tpm",
]
CLINICAL_CANDIDATES = [
    "TCGA.GBM.sampleMap_GBM_clinicalMatrix",
    "phenotype - Curated survival data.txt",
]

def _find_file_or_dir(dirpath, names):
    if not dirpath or not os.path.isdir(dirpath): return None
    # 精确命中
    for n in names:
        p = os.path.join(dirpath, n)
        if os.path.exists(p): return p
    # 模糊匹配
    files = os.listdir(dirpath)
    for n in names:
        key = n.replace(" ", "").split(".")[0].lower()
        for f in files:
            if key in f.replace(" ", "").lower():
                return os.path.join(dirpath, f)
    return None

def to_short_barcode(x, groups=3):
    if pd.isna(x): return None
    s = str(x).strip()
    if not s: return None
    parts = s.split("-")
    if len(parts) >= groups:
        return "-".join(parts[:groups])
    return s

def _looks_like_barcode(vals, sample=200):
    """抽样检查有多少比例像 TCGA- 开头，用于判断样本是否在列/行"""
    import random
    arr = list(vals)
    if len(arr) == 0: return 0.0
    k = min(len(arr), sample)
    pick = random.sample(arr, k)
    m = 0
    for v in pick:
        if str(v).startswith("TCGA-"):
            m += 1
    return m / k

def _load_table_firstcol_as_index(path):
    # 读入，初步以首列为索引
    try:
        df = pd.read_csv(path, sep="\t", header=0, dtype=str)
    except Exception:
        df = pd.read_csv(path, sep=r"\s+", header=0, dtype=str, engine="python")
    df = df.set_index(df.columns[0])

    # 自动识别朝向（样本应在列）
    col_ratio = _looks_like_barcode(df.columns)
    idx_ratio = _looks_like_barcode(df.index)
    need_transpose = idx_ratio > col_ratio
    if getattr(config, "DEBUG_LOG", False):
        print(f"[DEBUG] load '{os.path.basename(path)}': shape={df.shape} | barcode_in_cols={col_ratio:.2f} | barcode_in_rows={idx_ratio:.2f} | transpose={need_transpose}")
    if need_transpose:
        df = df.T

    # 转数值、缺失填0、去重列
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def _load_cnv(path):  return _load_table_firstcol_as_index(path)
def _load_rna(path):  return _load_table_firstcol_as_index(path)

def _load_meth(path):
    # 支持目录：从目录中选择体积最大的 .txt/.tsv 作为合并矩阵
    if os.path.isdir(path):
        cands = []
        for f in os.listdir(path):
            full = os.path.join(path, f)
            if os.path.isfile(full) and f.lower().endswith((".txt", ".tsv")):
                cands.append((os.path.getsize(full), full))
        if not cands:
            raise FileNotFoundError("HumanMethylation450 目录下未发现合并矩阵（.txt/.tsv）。")
        cands.sort(reverse=True)
        path = cands[0][1]
    return _load_table_firstcol_as_index(path)

def _load_labels(path):
    try:
        df = pd.read_csv(path, sep="\t", header=0, dtype=str)
    except Exception:
        df = pd.read_csv(path, sep=r"\s+", header=0, dtype=str, engine="python")
    df.columns = [c.strip() for c in df.columns]

    # 选择标签列
    label_col = None
    for c in ["GeneExp_Subtype", "geneexp_subtype", "gene_expression_subtype"]:
        if c in df.columns:
            label_col = c; break

    # 选择样本ID列
    sid_col = None
    for c in df.columns:
        lc = c.lower()
        if "sample" in lc or "id" in lc or "barcode" in lc:
            sid_col = c; break
    if sid_col is None:
        sid_col = df.columns[0]

    # 过滤原发瘤（若存在 sample_type）
    st_col = None
    for c in df.columns:
        if c.lower().replace(" ", "") in ["sample_type", "sampletype"]:
            st_col = c; break
    if st_col is not None:
        mask_primary = df[st_col].str.lower().str.contains("primary", na=False)
        df = df.loc[mask_primary]

    df["sid_raw"] = df[sid_col].astype(str)
    if label_col is None:
        y = pd.Series(index=[], dtype=str)
    else:
        y = df.set_index("sid_raw")[label_col].dropna().astype(str)
    return y

# ...（前面与你当前版本一致，这里只展示需要替换的函数片段） ...

def _apply_barcode_groups(df_or_series, groups):
    """把样本名/索引规范为短条形码，并对重复样本进行聚合（取均值）"""
    if isinstance(df_or_series, pd.Series):
        y = df_or_series.copy()
        y.index = [to_short_barcode(i, groups=groups) for i in y.index]
        y = y.groupby(level=0).first()
        return y
    else:
        df = df_or_series.copy()
        df.columns = [to_short_barcode(c, groups=groups) for c in df.columns]
        if df.columns.duplicated().any():
            # 修复 FutureWarning: 用转置后按列名分组再转回
            df = df.T.groupby(level=0).mean().T
        return df

def _align_once(groups, cnv, meth, rna, y):
    cnv2  = _apply_barcode_groups(cnv, groups)  if cnv  is not None else None
    meth2 = _apply_barcode_groups(meth, groups) if meth is not None else None
    rna2  = _apply_barcode_groups(rna, groups)  if rna  is not None else None
    y2    = _apply_barcode_groups(y, groups)    if y is not None and len(y)>0 else pd.Series(dtype=str)

    sets = []
    if cnv2  is not None: sets.append(set(cnv2.columns))
    if meth2 is not None: sets.append(set(meth2.columns))
    if rna2  is not None: sets.append(set(rna2.columns))
    if len(y2)>0:         sets.append(set(y2.index))
    inter = sorted(set.intersection(*sets)) if sets else []
    return inter, cnv2, meth2, rna2, y2

def align_modalities():
    data_dir = config.DATA_DIR
    cnv_path  = _find_file_or_dir(data_dir, CNV_CANDIDATES)  if config.USE_CNV  else None
    meth_path = _find_file_or_dir(data_dir, METH_CANDIDATES) if config.USE_METH else None
    rna_path  = _find_file_or_dir(data_dir, RNA_CANDIDATES)  if config.USE_RNA  else None
    clin_path = _find_file_or_dir(data_dir, CLINICAL_CANDIDATES)

    if config.USE_CNV  and not cnv_path:  raise FileNotFoundError("未找到 CNV 文件")
    if config.USE_METH and not meth_path: raise FileNotFoundError("未找到 Methyl450 文件或目录")
    if config.USE_RNA  and not rna_path:  raise FileNotFoundError("未找到 RNAseq 文件")

    cnv  = _load_cnv(cnv_path)   if cnv_path  else None
    meth = _load_meth(meth_path) if meth_path else None
    rna  = _load_rna(rna_path)   if rna_path  else None
    y    = _load_labels(clin_path) if clin_path else pd.Series(dtype=str)

    # 尝试 3 段与 4 段，选择交集更大的方案
    g_pref = int(getattr(config, "SHORT_BARCODE_GROUPS", 3))
    candidates = [g_pref] + ([4] if g_pref != 4 else []) + ([3] if g_pref != 3 else [])
    best = None
    for g in candidates:
        inter, cnv2, meth2, rna2, y2 = _align_once(g, cnv, meth, rna, y)
        if getattr(config, "DEBUG_LOG", False):
            def n(x): return 0 if x is None else x.shape[1] if hasattr(x, "shape") else len(x)
            print(f"[DEBUG] groups={g} | cnv_cols={n(cnv2)} meth_cols={n(meth2)} rna_cols={n(rna2)} y={len(y2)} | intersection={len(inter)}")
        if best is None or len(inter) > len(best[0]):
            best = (inter, cnv2, meth2, rna2, y2, g)
    samples, cnv, meth, rna, y, gsel = best

    # 打印两两交集规模，便于定位谁在掉样本
    if getattr(config, "DEBUG_LOG", False):
        sets = {}
        if cnv is not None:  sets["CNV"]  = set(cnv.columns)
        if meth is not None: sets["METH"] = set(meth.columns)
        if rna is not None:  sets["RNA"]  = set(rna.columns)
        if len(y)>0:         sets["Y"]    = set(y.index)
        keys = list(sets.keys())
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                a,b = keys[i], keys[j]
                print(f"[DEBUG] |{a}∩{b}|={len(sets[a] & sets[b])}")
        print(f"[DEBUG] 使用条形码段数 groups={gsel}，最终交集样本数={len(samples)}")

    if not samples:
        raise RuntimeError("各模态与标签的样本交集为空：请检查样本命名或文件朝向（已自动尝试3/4段条形码与自动转置）。")

    cnv_sub  = cnv.loc[:, samples]  if cnv  is not None else None
    meth_sub = meth.loc[:, samples] if meth is not None else None
    rna_sub  = rna.loc[:, samples]  if rna  is not None else None
    y_series = y.loc[samples] if len(y) > 0 else pd.Series(index=samples, dtype=str)
    return cnv_sub, meth_sub, rna_sub, y_series, samples
