# -*- coding: utf-8 -*-
import os, re, pandas as pd
from typing import Tuple, Dict
import config

SHORT_RE = re.compile(r'^(TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4}-(\d{2}))')

def to_short(s: str):
    if not isinstance(s,str): return None
    m = SHORT_RE.match(s.strip());
    return m.group(1) if m else None

def is_primary_short(short_id: str) -> bool:
    m = SHORT_RE.match(short_id) if isinstance(short_id,str) else None
    return bool(m and m.group(2)=="01")

def read_tsv(path): return pd.read_csv(path, sep="\t", low_memory=False)

def load_clinical_and_labels() -> pd.DataFrame:
    clin_path = os.path.join(config.DATA_DIR, config.FILES["clinical"])
    clin = read_tsv(clin_path)
    assert "sampleID" in clin.columns and "GeneExp_Subtype" in clin.columns, "clinical列缺失"
    clin["short"] = clin["sampleID"].astype(str).apply(to_short)
    # 任务：仅原发
    is_primary = clin["short"].apply(is_primary_short)
    # 标签模式
    y = clin.loc[is_primary & clin["GeneExp_Subtype"].notna(), ["short", "GeneExp_Subtype"]].copy()
    if config.SUBTYPE_MODE == "3class_merge_neural":
        y["GeneExp_Subtype"] = y["GeneExp_Subtype"].replace({"Neural": "Proneural"})
    elif config.SUBTYPE_MODE == "3class_drop_neural":
        y = y[y["GeneExp_Subtype"]!="Neural"]
    return y  # columns: short, GeneExp_Subtype

def load_cnv_thres() -> pd.DataFrame:
    p = os.path.join(config.DATA_DIR, config.FILES["cnv_thres"])
    df = read_tsv(p)
    # 第一列通常是Gene Symbol
    first = df.columns[0].lower()
    if first in {"gene symbol","gene","symbol","id"}:
        df = df.set_index(df.columns[0])
    cols = [c for c in df.columns if str(c).startswith("TCGA-")]
    df = df[cols]
    df.columns = [c[:16] for c in df.columns]
    return df  # rows=genes, cols=short sample

def load_mutation() -> pd.DataFrame:
    p = os.path.join(config.DATA_DIR, config.FILES["mutation"])
    df = read_tsv(p)
    if df.columns[0].lower()=="sample":
        df = df.set_index(df.columns[0])
    cols = [c for c in df.columns if str(c).startswith("TCGA-")]
    df = df[cols]
    df.columns = [c[:16] for c in df.columns]
    return df  # rows=genes, cols=short sample

def align_modalities() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, list]:
    """返回: CNV_df, MUT_df, y(Series), sample_order(list) —— 均仅保留(原发+有亚型)∩(CNV∩MUT)"""
    ydf = load_clinical_and_labels()
    cnv = load_cnv_thres()
    mut = load_mutation()

    # 交集
    inter = set(ydf["short"]) & set(cnv.columns) & set(mut.columns)
    sample_order = sorted(list(inter))
    assert len(sample_order) > 0, "交集为空，请检查数据"

    y = ydf.set_index("short").loc[sample_order, "GeneExp_Subtype"]

    cnv = cnv.loc[:, sample_order]
    mut = mut.loc[:, sample_order]

    return cnv, mut, y, sample_order
