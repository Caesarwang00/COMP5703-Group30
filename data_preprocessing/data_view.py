import os
import re
import glob
import pandas as pd

# ===== 基本路径 =====
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

# ===== 你当前的实际文件名（按截图填写）=====
PATHS = {
    "Mut_MC3_genelevel": os.path.join(BASE_DIR, "GBM_mc3_gene_level.txt"),
    "CNV_GISTIC2_thresholded": os.path.join(BASE_DIR, "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes"),
    "Meth_450K": os.path.join(BASE_DIR, "HumanMethylation450"),
    "Clinical": os.path.join(BASE_DIR, "TCGA.GBM.sampleMap_GBM_clinicalMatrix"),
    "Phenotype": os.path.join(BASE_DIR, "phenotype - Curated survival data.txt"),

    # RNA-seq 目前不在你的目录里；等你下载好，改成真实路径或文件夹即可：
    # 例如：os.path.join(BASE_DIR, "TCGA.GBM.sampleMap_HiSeqV2.txt")
    "Expr_RNAseq": os.path.join(BASE_DIR, "HiSeqV2"),
}

# ===== 小工具：在给定路径上解析“真实数据文件” =====
def resolve_file(p: str):
    """
    p 可以是：
      - 真实文件（返回自身）
      - 一个目录（返回其中最大的 .txt/.tsv/.gz）
      - 一个无扩展名的真实文件
    若找不到，返回 None
    """
    if p is None:
        return None
    # 直接是文件
    if os.path.isfile(p):
        return p
    # 是目录：找里面的候选
    if os.path.isdir(p):
        cand = []
        for ext in ("*.txt", "*.tsv", "*.csv", "*.gz", "*"):
            cand.extend(glob.glob(os.path.join(p, ext)))
        cand = [c for c in cand if os.path.isfile(c) and "__MACOSX" not in c and not os.path.basename(c).startswith(("._", "~$"))]
        if cand:
            cand.sort(key=lambda x: (-os.path.getsize(x), x))
            return cand[0]
        return None
    # 不是现成文件也不是目录：尝试在 BASE_DIR 里递归匹配“以 p 的basename 开头”的真实文件
    base = os.path.basename(p)
    matches = glob.glob(os.path.join(BASE_DIR, "**", base + "*"), recursive=True)
    matches = [m for m in matches if os.path.isfile(m)]
    if matches:
        matches.sort(key=lambda x: (-os.path.getsize(x), x))
        return matches[0]
    return None

# ===== 读取任意表格（优先制表符）=====
def read_any_table(path):
    if path is None:
        return None
    try:
        return pd.read_csv(path, sep="\t", header=0, index_col=0, low_memory=False)
    except Exception:
        try:
            return pd.read_csv(path, sep=",", header=0, index_col=0, low_memory=False)
        except Exception as e:
            print(f"[读取失败] {path}: {e}")
            return None

# ===== 识别样本轴（样本在列？）=====
_TCGA = re.compile(r"^TCGA-\w{2}-\w{4}-\d{2}", re.IGNORECASE)

def extract_sample_ids(df: pd.DataFrame):
    if df is None:
        return set(), None
    cols_like = sum(1 for c in df.columns[:200] if isinstance(c, str) and _TCGA.match(c))
    idxs_like = sum(1 for i in df.index[:200]   if isinstance(i, str) and _TCGA.match(i))
    if cols_like >= idxs_like:
        return set(map(str, df.columns)), True   # 样本在列
    else:
        return set(map(str, df.index)), False    # 样本在行

def show_preview(df):
    if df is not None:
        print(df.iloc[:5, :5])

# ===== 主流程：逐模态解析、读取、汇总 =====
loaded = {}

def load_one(tag, raw_path):
    path = resolve_file(raw_path)
    if path is None:
        print(f"[未找到文件] {tag} -> {raw_path}")
        loaded[tag] = {"path": None, "df": None, "samples": set(), "in_cols": None}
        return
    df = read_any_table(path)
    sids, in_cols = extract_sample_ids(df)
    print(f"\n=== {tag} ===")
    print("路径:", path)
    if df is not None:
        print("形状:", df.shape, "| 样本在列?", in_cols)
        print("样本数:", len(sids))
        show_preview(df)
    loaded[tag] = {"path": path, "df": df, "samples": sids, "in_cols": in_cols}

for k, v in PATHS.items():
    load_one(k, v)

# ===== 交集分析 =====
def report_intersections(keys):
    present = [k for k in keys if loaded.get(k, {}).get("df") is not None]
    if len(present) < 2:
        print("\n[交集] 可用模态不足 2 个，跳过。")
        return
    print("\n=== 两两交集样本数 ===")
    for i in range(len(present)):
        for j in range(i+1, len(present)):
            a, b = present[i], present[j]
            inter = loaded[a]["samples"] & loaded[b]["samples"]
            print(f"{a} ∩ {b}: {len(inter)}")

    def inter_of(names):
        sets = [loaded[n]["samples"] for n in names if loaded[n]["df"] is not None]
        return set.intersection(*sets) if len(sets) >= 2 else set()

    combos = [
        ("Expr∩Meth",           ["Expr_RNAseq", "Meth_450K"]),
        ("Expr∩CNV",            ["Expr_RNAseq", "CNV_GISTIC2_thresholded"]),
        ("Expr∩Mut",            ["Expr_RNAseq", "Mut_MC3_genelevel"]),
        ("Expr∩Meth∩CNV",       ["Expr_RNAseq", "Meth_450K", "CNV_GISTIC2_thresholded"]),
        ("Expr∩CNV∩Mut",        ["Expr_RNAseq", "CNV_GISTIC2_thresholded", "Mut_MC3_genelevel"]),
        ("Expr∩Meth∩Mut",       ["Expr_RNAseq", "Meth_450K", "Mut_MC3_genelevel"]),
        ("Expr∩Meth∩CNV∩Mut",   ["Expr_RNAseq", "Meth_450K", "CNV_GISTIC2_thresholded", "Mut_MC3_genelevel"]),
    ]
    print("\n=== 常用多模态交集（以表达为核心） ===")
    for label, names in combos:
        s = inter_of(names)
        print(f"{label}: {len(s)}" + (f" | 例: {list(s)[:8]}" if len(s) > 0 else ""))

report_intersections(list(PATHS.keys()))

# ===== 适用性结论（简单规则）=====
present = [k for k, v in loaded.items() if v["df"] is not None]
expr_ok = "Expr_RNAseq" in present
expr_meth = len((loaded.get("Expr_RNAseq", {}).get("samples", set())) &
                (loaded.get("Meth_450K", {}).get("samples", set())))
expr_cnv  = len((loaded.get("Expr_RNAseq", {}).get("samples", set())) &
                (loaded.get("CNV_GISTIC2_thresholded", {}).get("samples", set())))
expr_mut  = len((loaded.get("Expr_RNAseq", {}).get("samples", set())) &
                (loaded.get("Mut_MC3_genelevel", {}).get("samples", set())))

print("\n=== 结论 & 建议 ===")
if not expr_ok:
    print("- 检测到缺少 RNA-seq（HiSeqV2 基因表达）。GBM 亚型的主力模态是表达谱，建议尽快下载：")
    print("  -> TCGA.GBM.sampleMap/HiSeqV2.gz（Xena）解压后放到 data/，并把 PATHS['Expr_RNAseq'] 填上。")
else:
    print(f"- 表达谱已就绪；与甲基化交集样本 ≈ {expr_meth}，与 CNV 交集 ≈ {expr_cnv}，与突变交集 ≈ {expr_mut}。")
    if expr_meth >= 120 or expr_cnv >= 120:
        print("  -> 交集规模良好，支持做“表达+甲基化/表达+CNV”的多模态建模。")
    else:
        print("  -> 交集较小，建议先做表达单模态基线，再采用晚期融合（专家模型 stacking）。")
