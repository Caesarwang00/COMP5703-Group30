# src/labels.py
# -*- coding: utf-8 -*-
import os
import pandas as pd
from .dataio import to_short, looks_like_sample

def _dedup_label_index(ser: pd.Series, name: str = "GeneExp_Subtype") -> pd.Series:
    """按短样本号索引去重（保留首次出现），并打印去重统计。"""
    before = len(ser)
    # 仅保留看起来像样本ID的索引
    ser = ser[[looks_like_sample(ix) for ix in ser.index]]
    # 去重
    dup_mask = ser.index.duplicated(keep="first")
    dups = int(dup_mask.sum())
    if dups > 0:
        print(f"[labels] 发现 {name} 标签重复样本 {dups} 个（按短码去重，保留首次出现）。")
    ser = ser[~dup_mask]
    # 清理空白
    ser = ser.dropna().astype(str).str.strip()
    after = len(ser)
    if after == 0:
        print(f"[labels][WARN] 去重后标签为空；请检查 clinical。")
    return ser

def load_labels_from_clinical(path: str, label_key: str = "GeneExp_Subtype") -> pd.Series:
    """
    适配 UCSC Xena clinicalMatrix：
      A) 行=字段, 列=样本（常见）。第一列名如 sample/field，行名包含 label_key。
      B) 行=样本, 列含 label_key。需找到样本ID列。
    返回: Series(index=短样本号, values=标签字符串)，索引已去重。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[labels] clinical 文件不存在: {path}")

    df = pd.read_csv(path, sep="\t", low_memory=False)
    first_col = df.columns[0]

    # A: 行=字段
    if str(first_col).strip().lower() in {"sample", "id", "field", "feature"}:
        df = df.set_index(first_col)
        if label_key not in df.index:
            raise RuntimeError(f"[labels] clinical 中不存在行 '{label_key}'")
        # 列名转短码
        cols = [to_short(str(c)) or str(c) for c in df.columns]
        df.columns = cols
        ser = df.loc[label_key]
        ser.index = [ix for ix in ser.index]  # 保持列名（已转短码）
        return _dedup_label_index(ser, name=label_key)

    # B: 行=样本
    if label_key in df.columns:
        cand = [c for c in ["sampleID","bcr_sample_barcode","bcr_patient_barcode","sample","id"] if c in df.columns]
        if cand:
            sid_col = cand[0]
            ser = df.set_index(sid_col)[label_key]
            ser.index = ser.index.map(lambda x: to_short(str(x)) or str(x))
            return _dedup_label_index(ser, name=label_key)
        else:
            # 尝试索引即样本ID
            df = df.copy()
            df.index = df.index.map(lambda x: to_short(str(x)) or str(x))
            ser = df[label_key]
            return _dedup_label_index(ser, name=label_key)

    raise RuntimeError("[labels] 无法在 clinical 中找到 GeneExp_Subtype（既不在行也不在列）。")
