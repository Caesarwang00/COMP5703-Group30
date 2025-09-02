import math
from collections import Counter

import numpy as np
import pandas as pd


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


from train import preprocess_pipeline_with_subtype


EXPR_PATH = r"D:/毕设数据/HiSeqV2.xlsx"
CLIN_PATH = r"D:/毕设数据/TCGA.GBM.sampleMap_GBM_clinicalMatrix.xlsx"


def pct(v): return f"{100*v:.1f}%"

def class_ratios(y: pd.Series) -> dict:
    c = Counter(y.tolist())
    total = sum(c.values())
    return {k: c[k] / total for k in sorted(c)}

def ratio_diff(a: dict, b: dict) -> float:
    """最大类别占比差异（越小越好）"""
    keys = set(a) | set(b)
    return max(abs(a.get(k,0) - b.get(k,0)) for k in keys)

def print_header(title: str):
    print("\n" + "="*10 + f" {title} " + "="*10)

def assert_true(cond: bool, msg: str):
    if cond:
        print(f"[PASS] {msg}")
    else:
        print(f"[FAIL] {msg}")
        raise AssertionError(msg)

# ===== 运行预处理 =====
print_header("Load & Preprocess")
res = preprocess_pipeline_with_subtype(
    expr_path=EXPR_PATH,
    clinical_path=CLIN_PATH,
    min_nonzero_pct=0.2,
    topk_by_variance=2000,   # 可按需调整
    test_size=0.2,
    val_size=0.1,
    random_state=42
)

Xtr, Xva, Xte = res["X_train"], res["X_val"], res["X_test"]
ytr, yva, yte = res["y_train"], res["y_val"], res["y_test"]
genes = res["genes"]

print(f"Shapes:\n  X_train: {Xtr.shape}\n  X_val:   {Xva.shape}\n  X_test:  {Xte.shape}")
print("Classes:", sorted(ytr.unique()))

# 测试1：数据完整性
print_header("Test 1: Data integrity")
assert_true(Xtr.shape[0] > 0 and Xtr.shape[1] > 0, "训练集非空")
assert_true(Xva.shape[0] > 0 and Xva.shape[1] == Xtr.shape[1], "验证集列数与训练集一致")
assert_true(Xte.shape[0] > 0 and Xte.shape[1] == Xtr.shape[1], "测试集列数与训练集一致")
assert_true(len(genes) == Xtr.shape[1], "基因特征数与列数一致")
assert_true(not pd.isna(Xtr.values).any(), "训练集无 NaN")
assert_true(not pd.isna(Xva.values).any(), "验证集无 NaN")
assert_true(not pd.isna(Xte.values).any(), "测试集无 NaN")

# 拆分互斥/并集正确
idx_tr, idx_va, idx_te = set(Xtr.index), set(Xva.index), set(Xte.index)
assert_true(idx_tr.isdisjoint(idx_va) and idx_tr.isdisjoint(idx_te) and idx_va.isdisjoint(idx_te),
            "train/val/test 样本互不重叠")
union_size = len(idx_tr | idx_va | idx_te)
assert_true(union_size == (len(idx_tr) + len(idx_va) + len(idx_te)), "三集并集大小正确")

# 测试2：分层划分效果 
print_header("Test 2: Stratification check")
all_y = pd.concat([ytr, yva, yte])
rat_all = class_ratios(all_y)
rat_tr  = class_ratios(ytr)
rat_va  = class_ratios(yva)
rat_te  = class_ratios(yte)

print("Overall ratios:", {k: pct(v) for k, v in rat_all.items()})
print("Train   ratios:", {k: pct(v) for k, v in rat_tr.items()})
print("Val     ratios:", {k: pct(v) for k, v in rat_va.items()})
print("Test    ratios:", {k: pct(v) for k, v in rat_te.items()})

# 允许一点浮动（样本少时波动更大）
tol = 0.18
assert_true(ratio_diff(rat_tr, rat_all) <= tol, f"训练集类别占比与总体接近(≤{pct(tol)})")
assert_true(ratio_diff(rat_va, rat_all) <= tol, f"验证集类别占比与总体接近(≤{pct(tol)})")
assert_true(ratio_diff(rat_te, rat_all) <= tol, f"测试集类别占比与总体接近(≤{pct(tol)})")

# 测试3：标准化效果（仅检查训练集）
print_header("Test 3: Standardization (z-score) on train only")
means = Xtr.mean(axis=0).abs().mean()    # 所有特征均值的绝对值的平均
stds  = (Xtr.std(axis=0, ddof=0) - 1).abs().mean()  # 方差应≈1
print(f"Avg |mean| over features: {means:.4f}")
print(f"Avg |std-1| over features: {stds:.4f}")

assert_true(means < 0.1, "训练集 z-score 后特征均值≈0（平均偏差 < 0.1）")
assert_true(stds  < 0.1, "训练集 z-score 后特征方差≈1（平均偏差 < 0.1）")

# 测试4：训练与评估（Logistic Regression 基线）
print_header("Test 4: Train & Evaluate (LogReg)")
clf = LogisticRegression(max_iter=5000, solver="saga", class_weight="balanced")
clf.fit(Xtr, ytr)

def report(split, X, y):
    pred = clf.predict(X)
    acc = accuracy_score(y, pred)
    f1m = f1_score(y, pred, average="macro")
    print(f"\n[{split}] acc={acc:.3f}  macroF1={f1m:.3f}")
    print(classification_report(y, pred))
    print("Confusion matrix:\n", confusion_matrix(y, pred, labels=sorted(y.unique())))
    return acc, f1m

acc_va, f1_va = report("VAL", Xva, yva)
acc_te, f1_te = report("TEST", Xte, yte)

# 不设置苛刻阈值，做一个“烟囱测试”保证不是随机
assert_true(acc_va > 0.45, "验证集准确率高于随机水平(>0.45)")
assert_true(acc_te > 0.45, "测试集准确率高于随机水平(>0.45)")

print_header("All tests passed ✅")
