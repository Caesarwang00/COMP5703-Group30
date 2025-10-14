import pandas as pd
from pathlib import Path

# 脚本所在目录
BASE = Path(__file__).resolve().parent

# 输入与输出路径
in_path  = (BASE / "../../data_RNA/GBM_clinicalMatrix").resolve()
out_dir  = BASE / "../data_RNA"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "subtype_labels.tsv"

clinical = pd.read_csv(in_path, sep="\t")
labels = clinical[["sampleID", "GeneExp_Subtype"]].dropna()
labels.to_csv(out_path, sep="\t", index=False)

print(f"标签文件已生成: {out_path}")
