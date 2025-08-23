# -*- coding: utf-8 -*-
"""
GBM 多模态数据体检（只print；修复原发瘤=01判断 & 清理非样本列）
放到: data_preprocessing/data_analyse.py
运行:  python data_preprocessing/data_analyse.py
"""

import os
import re
import numpy as np
import pandas as pd

# ========= 路径（相对）=========
HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.normpath(os.path.join(HERE, "..", "data"))

FILES = {
    "mutation":  os.path.join(BASE_DIR, "GBM_mc3_gene_level.txt"),
    "cnv_thres": os.path.join(BASE_DIR, "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes"),
    "methyl450": os.path.join(BASE_DIR, "HumanMethylation450"),
    "mirna":     os.path.join(BASE_DIR, "miRNA_HiSeq_gene"),
    "clinical":  os.path.join(BASE_DIR, "TCGA.GBM.sampleMap_GBM_clinicalMatrix"),
}

# ========= 工具函数 =========
SHORT_RE = re.compile(r'^(TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4}-(\d{2}))')

def read_tsv(path, nrows=None):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t", low_memory=False, nrows=nrows)

def to_sample_short(s: str):
    """从任意TCGA条形码提取短样本号: TCGA-XX-XXXX-YY"""
    if not isinstance(s, str):
        return None
    m = SHORT_RE.match(s.strip())
    return m.group(1) if m else None

def is_primary_by_short(short_id: str):
    """通过短样本号末段 YY 判断原发瘤（01=Primary Tumor）"""
    if not isinstance(short_id, str):
        return False
    m = SHORT_RE.match(short_id)
    if not m:
        return False
    yy = m.group(2)  # 两位
    return yy == "01"

def guess_matrix_orientation(df: pd.DataFrame, name: str):
    """
    统一为: 行=特征(基因/探针), 列=样本(短样本号或条形码)
    并尽量清理非样本列（如 Gene Symbol / sample）
    """
    notes = []
    df0 = df.copy()

    # 1) 处理显式的特征列名
    first_lower = df0.columns[0].lower()
    special_first_cols = {
        "gene", "genes", "hugo", "symbol", "gene symbol",
        "id", "probe", "probes", "composite element ref"
    }
    if (first_lower in special_first_cols) or (df0.columns[0] in ["Gene Symbol", "sample"]):
        df0 = df0.set_index(df0.columns[0])
        notes.append(f"{name}: 第一列 {df.columns[0]} 设为索引")

    # 2) 判断样本是否在列；若不在列且第一列是TCGA，则转置
    col_has_tcga = any(str(c).startswith("TCGA-") for c in df0.columns)
    if not col_has_tcga:
        first_col = df0.columns[0]
        if df0[first_col].astype(str).str.startswith("TCGA-").any():
            df0 = df0.set_index(first_col).transpose()
            notes.append(f"{name}: 检测到样本在行上，已转置")
        else:
            # 3) 只保留以 TCGA- 开头的列（若>=3列）
            tcga_cols = [c for c in df0.columns if str(c).startswith("TCGA-")]
            if len(tcga_cols) >= 3:
                drop_cols = [c for c in df0.columns if c not in tcga_cols][:5]
                if drop_cols:
                    notes.append(f"{name}: 丢弃非样本列示例: {drop_cols}")
                df0 = df0[tcga_cols]

    # 4) 再次强制删除常见非样本列名
    drop_explicit = [c for c in df0.columns if str(c).lower() in {"gene symbol", "sample"}]
    if drop_explicit:
        df0 = df0.drop(columns=drop_explicit, errors="ignore")
        notes.append(f"{name}: 显式删除非样本列: {drop_explicit}")

    # 尽量转为数值
    df_num = df0.apply(pd.to_numeric, errors="coerce")
    return df_num, notes

def basic_checks(df_num: pd.DataFrame):
    rep = {}
    total = df_num.size if df_num.size else 1
    rep["shape"] = df_num.shape
    rep["nan_total"] = int(df_num.isna().sum().sum())
    rep["nan_pct"] = round(df_num.isna().sum().sum() / total * 100, 4)
    rep["all_zero_rows"] = int((df_num.fillna(0).abs().sum(axis=1) == 0).sum())
    rep["all_zero_cols"] = int((df_num.fillna(0).abs().sum(axis=0) == 0).sum())
    rep["zero_pct_overall"] = round((df_num.fillna(0) == 0).sum().sum() / total * 100, 4)
    rep["worst_cols_by_na"] = df_num.isna().mean().sort_values(ascending=False).head(5).round(4).to_dict()
    rep["worst_rows_by_na"] = df_num.isna().mean(axis=1).sort_values(ascending=False).head(5).round(4).to_dict()
    return rep

def rule_checks(name: str, df_num: pd.DataFrame):
    abn = {}
    vals = df_num.values.flatten()
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return abn

    if name == "cnv_thres":
        allowed = {-2, -1, 0, 1, 2}
        uniq = set(np.unique(vals).tolist())
        abn["allowed_values"] = sorted(list(uniq))[:10]
        abn["invalid_value_count"] = int(np.sum(~np.isin(vals, list(allowed))))
        if abn["invalid_value_count"] > 0:
            abn["invalid_values_example"] = [v for v in uniq if v not in allowed][:10]

    elif name == "mutation":
        allowed = {0, 1}
        uniq = set(np.unique(vals).tolist())
        abn["allowed_values"] = sorted(list(uniq))[:10]
        abn["invalid_value_count"] = int(np.sum(~np.isin(vals, list(allowed))))
        if abn["invalid_value_count"] > 0:
            abn["invalid_values_example"] = [v for v in uniq if v not in allowed][:10]

    elif name == "methyl450":
        below = int((vals < -1e-6).sum())
        above = int((vals > 1 + 1e-6).sum())
        abn["below_0_count"] = below
        abn["above_1_count"] = above
        if below or above:
            abn["range_warning"] = True

    return abn

def clinical_labels(clin: pd.DataFrame):
    cols = {c.lower(): c for c in clin.columns}
    def pick(*names):
        for n in names:
            c = cols.get(n.lower())
            if c: return c
        return None

    subtype_col = pick("GeneExp_Subtype")
    gcimp_col   = pick("G_CIMP_STATUS")
    stypeid_col = pick("sample_type_id")
    stype_col   = pick("sample_type")
    sid_col     = pick("sampleID", "bcr_sample_barcode", "bcr_sample_barcode")
    patient_col = pick("bcr_patient_barcode")

    # 分布
    subtype_vc = clin[subtype_col].dropna().astype(str).value_counts().to_dict() if subtype_col else {}
    gcimp_vc   = clin[gcimp_col].dropna().astype(str).value_counts().to_dict() if gcimp_col else {}

    # 短样本号
    if sid_col:
        sample_short = clin[sid_col].astype(str).apply(to_sample_short)
    else:
        sample_short = pd.Series([None]*len(clin), index=clin.index)

    # 用短样本号直接判断原发瘤=01（最稳妥）
    is_primary = sample_short.apply(is_primary_by_short)

    # 退路：如果短样本号缺失，再用 sample_type_id / sample_type
    if (~is_primary).all():
        if stypeid_col and clin[stypeid_col].notna().any():
            is_primary = clin[stypeid_col].astype(str).str.zfill(2).str[:2].eq("01")
        elif stype_col:
            is_primary = clin[stype_col].astype(str).str.lower().str.contains("primary")

    labelled_mask = pd.Series(False, index=clin.index)
    if subtype_col: labelled_mask |= clin[subtype_col].notna()
    if gcimp_col:   labelled_mask |= clin[gcimp_col].notna()

    labelled_primary = labelled_mask & is_primary
    n_labelled_primary = int(labelled_primary.sum())
    labelled_primary_shorts = set(sample_short[labelled_primary].dropna().unique().tolist())

    # 调试信息
    print("\n[DEBUG] 原发瘤判断：")
    print(" - 有效短样本号数量:", int(sample_short.notna().sum()))
    if stypeid_col:
        vals = clin[stypeid_col].dropna().astype(str).str[:2].value_counts().to_dict()
        print(" - sample_type_id 前两位分布示例:", {k: vals[k] for k in list(vals)[:6]})
    if stype_col:
        vals2 = clin[stype_col].dropna().astype(str).str.lower().value_counts().head(5).to_dict()
        print(" - sample_type 频次Top5:", vals2)

    return {
        "subtype_vc": subtype_vc,
        "gcimp_vc": gcimp_vc,
        "n_labelled_primary": n_labelled_primary,
        "labelled_primary_shorts": labelled_primary_shorts
    }

def extract_shorts_from_cols(df_num: pd.DataFrame):
    shorts = set()
    for c in df_num.columns:
        s = to_sample_short(str(c))
        if s: shorts.add(s)
    return shorts

# ========= 主流程 =========
def main():
    print(f"[INFO] BASE_DIR = {BASE_DIR}")

    # 1) 临床与标签
    clin = read_tsv(FILES["clinical"])
    lab = clinical_labels(clin)
    print("\n=== 临床/标签分布 ===")
    print("GeneExp_Subtype:", lab["subtype_vc"])
    print("G_CIMP_STATUS  :", lab["gcimp_vc"])
    print("带标签且原发瘤的样本数:", lab["n_labelled_primary"])

    # 为“按亚型计数”准备映射：short -> GeneExp_Subtype
    sid_col = "sampleID" if "sampleID" in clin.columns else ("bcr_sample_barcode" if "bcr_sample_barcode" in clin.columns else None)
    if sid_col is None:
        raise RuntimeError("clinical 中缺少 sampleID / bcr_sample_barcode")
    clin["__short"] = clin[sid_col].astype(str).apply(to_sample_short)
    has_sub = clin["GeneExp_Subtype"].notna()
    map_short2sub = dict(zip(clin.loc[has_sub, "__short"], clin.loc[has_sub, "GeneExp_Subtype"]))

    def count_by_sub(short_set, title):
        # 只统计有 GeneExp_Subtype 的样本
        subs = pd.Series([map_short2sub[s] for s in short_set if s in map_short2sub])
        print(f"\n[{title}] n={len(short_set)}；其中有亚型标签的样本数={len(subs)}")
        if len(subs) > 0:
            vc = subs.value_counts().sort_index()
            for k, v in vc.items():
                print(f"  {k}: {v}")
        else:
            print("  (无亚型标签可统计)")

    # 2) 各模态检查
    sample_sets = {}
    for key in ["cnv_thres", "mutation", "methyl450", "mirna"]:
        path = FILES[key]
        if not os.path.exists(path):
            print(f"\n[WARN] 缺失文件: {key} -> {path}")
            continue

        df_raw = read_tsv(path)
        df_num, notes = guess_matrix_orientation(df_raw, key)
        print(f"\n=== {key} ===")
        for n in notes:
            print(" -", n)

        rep = basic_checks(df_num)
        abn = rule_checks(key, df_num)

        print(f"形状: {rep['shape']}")
        print(f"缺失值比例: {rep['nan_pct']}% | 全零行: {rep['all_zero_rows']} | 全零列: {rep['all_zero_cols']} | 0值比例: {rep['zero_pct_overall']}%")
        if rep["all_zero_cols"] > 0:
            zero_cols = df_num.columns[(df_num.fillna(0).abs().sum(axis=0) == 0)].tolist()[:5]
            print("全零列示例:", zero_cols)
        if abn:
            print("规则检查:", abn)
        print("最差样本(按缺失率Top5):", rep["worst_cols_by_na"])

        # 收集样本短ID
        sample_sets[key] = extract_shorts_from_cols(df_num)
        print("可解析短样本号数:", len(sample_sets[key]))

    # 3) 与“带标签原发瘤”的交集（单模态）
    print("\n=== 单模态与(标签∩原发)交集 ===")
    label_set = lab["labelled_primary_shorts"]
    for k, s in sample_sets.items():
        inter = s & label_set
        print(f"{k}: 模态样本={len(s)}, 交集={len(inter)}")
        # 按亚型统计（单模态）
        count_by_sub(inter, f"{k} + 标签(原发)")

    # 4) 双/三模态联合交集
    S_cnv  = sample_sets.get("cnv_thres", set())
    S_mut  = sample_sets.get("mutation", set())
    S_meth = sample_sets.get("methyl450", set())

    print("\n=== 双/三模态与(标签∩原发)交集 ===")
    inter_cnv_mut  = (S_cnv & S_mut  & label_set)
    inter_cnv_meth = (S_cnv & S_meth & label_set)
    inter_mut_meth = (S_mut & S_meth & label_set)
    inter_all3     = (S_cnv & S_mut & S_meth & label_set)

    print("CNV + Mutation + 标签(原发):", len(inter_cnv_mut))
    print("CNV + Methyl450 + 标签(原发):", len(inter_cnv_meth))
    print("Mutation + Methyl450 + 标签(原发):", len(inter_mut_meth))
    print("CNV + Mutation + Methyl450 + 标签(原发):", len(inter_all3))

    # 各组合的亚型分布
    count_by_sub(inter_cnv_mut,  "CNV + Mutation + 标签(原发)")
    count_by_sub(inter_cnv_meth, "CNV + Methyl450 + 标签(原发)")
    count_by_sub(inter_mut_meth, "Mutation + Methyl450 + 标签(原发)")
    count_by_sub(inter_all3,     "CNV + Mutation + Methyl450 + 标签(原发)")

if __name__ == "__main__":
    main()

