import os
import pandas as pd
import yaml

def read_table_auto(path: str) -> pd.DataFrame:
    """
    自动识别分隔符读取表格。
    约定：第一列为特征名，后续列为样本ID；行=特征，列=样本。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        # fallback
        if path.endswith(".tsv") or path.endswith(".txt"):
            df = pd.read_csv(path, sep="\t")
        else:
            df = pd.read_csv(path)
    return df

def read_labels(path: str) -> pd.DataFrame:
    """
    读取标签文件，需包含列: sample_id, label
    """
    df = read_table_auto(path)
    cols = [c.lower() for c in df.columns]
    rename_map = {}
    if "sample_id" not in cols:
        for c in df.columns:
            if c.lower() in ("sample", "id", "sampleid", "case_id"):
                rename_map[c] = "sample_id"
    if "label" not in cols:
        for c in df.columns:
            if c.lower() in ("subtype", "class", "y", "group", "phenotype"):
                rename_map[c] = "label"
    if rename_map:
        df = df.rename(columns=rename_map)
    assert "sample_id" in [c.lower() for c in df.columns], "labels file must contain 'sample_id' column"
    assert "label" in [c.lower() for c in df.columns], "labels file must contain 'label' column"
    df = df.rename(columns={c: c.lower() for c in df.columns})
    return df[["sample_id", "label"]]

def read_yaml(path: str):
    """
    读取 YAML 并返回 dict
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
