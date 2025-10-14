# -*- coding: utf-8 -*-
"""
train_search.py — 自动超参搜索 (GraphSAGE)
搜索 hid / lr / dropout / weight_decay 组合，并保存每组的最佳 Val/Test 准确率。

需要的文件：
  - multiomics_fused_features.tsv   (样本 × 特征)
  - edge_list.csv                   (两列：source/target 或类似命名)
  - subtype_labels.tsv              (至少含 sample / subtype 或 sampleid / geneexp_subtype)

输出：
  - search_results.csv              (每组超参的 best_val / best_test)
  - 控制台最后打印最优超参
"""

import os
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# -------------------------
# 读取特征
# -------------------------
FEAT = "../data_RNA/multiomics_fused_features.tsv"
if not os.path.exists(FEAT):
    raise FileNotFoundError(f"缺少 {FEAT}")

X = pd.read_csv(FEAT, sep="\t", index_col=0)
X.index = X.index.astype(str).str.strip()
node_ids = X.index.tolist()
x = torch.tensor(X.values, dtype=torch.float32)
id2idx = {sid: i for i, sid in enumerate(node_ids)}
print(f"✅ 读取特征：samples={x.shape[0]}, dims={x.shape[1]}")

# -------------------------
# 读取边 (更健壮：自动识别列名，映射后 dropna，保证 src/dst 等长)
# -------------------------
EL = "../data_RNA/edge_list.csv"
if not os.path.exists(EL):
    raise FileNotFoundError(f"缺少 {EL}")

edges = pd.read_csv(EL)
edges.columns = [str(c).strip().lower() for c in edges.columns]

# 自动识别两列
cand_src = [c for c in ["source","src","from","u","node1"] if c in edges.columns]
cand_dst = [c for c in ["target","dst","to","v","node2"]  if c in edges.columns]
if not cand_src or not cand_dst:
    if edges.shape[1] >= 2:
        edges = edges.iloc[:, :2]
        edges.columns = ["source","target"]
    else:
        raise ValueError("edge_list.csv 至少需要两列（source/target）")

edges = edges[[cand_src[0] if cand_src else "source",
               cand_dst[0]  if cand_dst  else "target"]].copy()
edges.columns = ["source","target"]

# 映射到节点索引，并丢弃未对齐的行
edges["source"] = edges["source"].astype(str).str.strip().map(id2idx)
edges["target"] = edges["target"].astype(str).str.strip().map(id2idx)
edges = edges.dropna()
src = edges["source"].astype(int).to_numpy()
dst = edges["target"].astype(int).to_numpy()

# 组装 edge_index（保证两个数组等长）
if len(src) == 0 or len(dst) == 0 or len(src) != len(dst):
    raise ValueError(f"边数据不合法：src={len(src)} dst={len(dst)}（需相等且 >0）")
edge_index = torch.tensor([src, dst], dtype=torch.long)
print(f"✅ 读取边：edges={edge_index.shape[1]}")

# -------------------------
# 读取标签 & 分层划分 (train/val/test = 70/15/15)
# -------------------------
LAB = "../data_RNA/subtype_labels.tsv"
if not os.path.exists(LAB):
    raise FileNotFoundError(f"缺少 {LAB}")

labels = pd.read_csv(LAB, sep="\t")
labels.columns = [str(c).strip().lower() for c in labels.columns]
name_col = "sample" if "sample" in labels.columns else ("sampleid" if "sampleid" in labels.columns else None)
sub_col  = "subtype" if "subtype" in labels.columns else ("geneexp_subtype" if "geneexp_subtype" in labels.columns else None)
if name_col is None or sub_col is None:
    raise ValueError("标签文件需要包含 sample/subtype 或 sampleid/geneexp_subtype 两列")

labels = labels.rename(columns={name_col: "sample", sub_col: "subtype"})
labels["sample"] = labels["sample"].astype(str).str.strip()
labels = labels.set_index("sample").reindex(node_ids)

keep = ~labels["subtype"].isna()
labeled_nodes = np.where(keep.values)[0]
if len(labeled_nodes) == 0:
    raise ValueError("没有可用的标签（全部为缺失）")

le = LabelEncoder()
y_full = np.full(len(node_ids), -1, dtype=int)
y_full[labeled_nodes] = le.fit_transform(labels.loc[keep, "subtype"].values)
classes = list(le.classes_)
y = torch.tensor(y_full, dtype=torch.long)
print(f"✅ 标签：{len(classes)} 类 → {classes}")

# 分层划分
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
tr_idx, vt_idx = next(sss1.split(np.zeros(len(labeled_nodes)), y_full[labeled_nodes]))
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
v_idx, te_idx = next(sss2.split(np.zeros(len(vt_idx)), y_full[labeled_nodes][vt_idx]))

train_nodes = labeled_nodes[tr_idx]
val_nodes   = labeled_nodes[vt_idx[v_idx]]
test_nodes  = labeled_nodes[vt_idx[te_idx]]

n = len(node_ids)
train_mask = torch.zeros(n, dtype=torch.bool); train_mask[train_nodes] = True
val_mask   = torch.zeros(n, dtype=torch.bool); val_mask[val_nodes]    = True
test_mask  = torch.zeros(n, dtype=torch.bool); test_mask[test_nodes]  = True
print(f"✅ 划分：train={train_mask.sum().item()}, val={val_mask.sum().item()}, test={test_mask.sum().item()}")

# -------------------------
# 定义模型与工具函数
# -------------------------
class SageNet(nn.Module):
    def __init__(self, in_ch, hid=64, out_ch=4, drop=0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_ch, hid)
        self.bn1   = nn.BatchNorm1d(hid)
        self.conv2 = SAGEConv(hid, hid)
        self.bn2   = nn.BatchNorm1d(hid)
        self.out   = nn.Linear(hid, out_ch)
        self.drop  = drop
    def forward(self, x, edge_index):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.drop, training=self.training)
        return self.out(x)

def accuracy(logits, labels, mask):
    if mask.sum() == 0: return float("nan")
    pred = logits[mask].argmax(dim=1)
    return (pred == labels[mask]).float().mean().item()

# -------------------------
# 超参网格
# -------------------------
param_grid = {
    "hid":  [48, 64],                           # 贴近你最佳的 64，也试 48
    "lr":   [3e-4, 5e-4, 7e-4, 1e-3],           # 以 5e-4 为中心的窄域
    "drop": [0.3, 0.4, 0.5],                    # 围绕 0.4
    "wd":   [0.0, 1e-5, 5e-5, 1e-4],            # 关键：加入 0
}

combos = list(itertools.product(param_grid["hid"], param_grid["lr"], param_grid["drop"], param_grid["wd"]))

results = []
print(f"🔍 开始搜索，共 {len(combos)} 组…")

# -------------------------
# 训练循环（每组 200 epoch，取 best Val/Test）
# -------------------------
for hid, lr, drop, wd in combos:
    model = SageNet(x.shape[1], hid=hid, out_ch=len(classes), drop=drop)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_val, best_test = 0.0, 0.0
    for epoch in range(1, 201):
        model.train()
        opt.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[train_mask], y[train_mask])
        loss.backward()
        opt.step()

        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(x, edge_index)
                val_acc = accuracy(logits, y, val_mask)
                test_acc = accuracy(logits, y, test_mask)
                if val_acc > best_val:
                    best_val = val_acc
                    best_test = test_acc

    results.append({
        "hid": hid, "lr": lr, "drop": drop, "wd": wd,
        "best_val": float(best_val), "best_test": float(best_test)
    })
    print(f"✅ hid={hid}, lr={lr}, drop={drop}, wd={wd} | Val={best_val:.3f} Test={best_test:.3f}")

# -------------------------
# 保存结果并输出最优组合
# -------------------------
df = pd.DataFrame(results)
df.to_csv("search_results.csv", index=False)
best = df.sort_values("best_val", ascending=False).iloc[0]
print("\n🎉 最优超参:")
print(best)



