# -*- coding: utf-8 -*-
"""
train_mgcn_rna.py — 最终版 GraphSAGE 训练脚本（支持外部 SEED、多文件健壮读取、早停、类别权重、GNNGuard）

输入（同目录）：
  - multiomics_fused_features.tsv   行=样本；列=融合后的多组学特征
  - edge_list.csv                   两列（自动识别 source/target/src/dst 等命名）
  - subtype_labels.tsv              至少包含 sample/subtype（或 sampleid/geneexp_subtype）

可选环境变量：
  - SEED：随机种子（默认 42）

输出：
  - node_embeddings.tsv             全量样本的节点表示（用于可视化/下游任务）
  - predictions_test.csv            测试集预测明细（含真实/预测及每类概率）
  - metrics_summary.txt             JSON 文本，包含 Train/Val/Test 指标、分类报告、混淆矩阵
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# ========================
# 随机种子（可被环境变量覆盖）
# ========================
SEED = int(os.environ.get("SEED", "42"))
torch.manual_seed(SEED)
np.random.seed(SEED)

# ========================
# 固定最优超参（来自你的搜索）
# ========================
HID_DIM       = 64
LR            = 0.0005
DROPOUT       = 0.4
WEIGHT_DECAY  = 5e-4
MAX_EPOCHS    = 800
PATIENCE      = 30   # 以验证集准确率为准的早停耐心

# ========================
# 读取特征
# ========================
FEAT = "../data_RNA/multiomics_fused_features.tsv"
if not os.path.exists(FEAT):
    raise FileNotFoundError(f"缺少 {FEAT}")
X = pd.read_csv(FEAT, sep="\t", index_col=0)
X.index = X.index.astype(str).str.strip()
node_ids = X.index.tolist()
x = torch.tensor(X.values, dtype=torch.float32)
id2idx = {sid: i for i, sid in enumerate(node_ids)}
print(f"✅ 读取特征：samples={x.shape[0]}, dims={x.shape[1]}")

# ========================
# 读取边（自动识别列名；映射后 dropna；包含 GNNGuard 重加权）
# ========================
EL = "../data_RNA/edge_list.csv"
if not os.path.exists(EL):
    raise FileNotFoundError(f"缺少 {EL}")
edges = pd.read_csv(EL)
edges.columns = [str(c).strip().lower() for c in edges.columns]
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
edges["source"] = edges["source"].astype(str).str.strip().map(id2idx)
edges["target"] = edges["target"].astype(str).str.strip().map(id2idx)
edges = edges.dropna()
src = edges["source"].astype(int).to_numpy()
dst = edges["target"].astype(int).to_numpy()
if len(src) == 0 or len(src) != len(dst):
    raise ValueError(f"边数据不合法：src={len(src)} dst={len(dst)}（需相等且>0）")

# 更快地创建张量，避免 “list of numpy.ndarrays is slow” 的告警
edge_index = torch.from_numpy(np.vstack([src, dst]).astype(np.int64))
print(f"✅ 使用 edge_list.csv 构图：edges={edge_index.shape[1]}")

# ---- 简化版 GNNGuard：按余弦相似度重加权并对每个节点保留 Top-p 邻居 ----
def gnnguard_reweight(x_feat: torch.Tensor, edge_index: torch.Tensor,
                      p_keep: float = 0.7, alpha: float = 2.0):
    row, col = edge_index
    h = F.normalize(x_feat, dim=1)                 # 归一化特征
    s = (h[row] * h[col]).sum(dim=1).clamp(min=0)  # 余弦相似度（截负）
    w = (s ** alpha)                                # 幂次强化

    E = edge_index.size(1)
    keep = torch.zeros(E, dtype=torch.bool)
    idx_by_src = {}
    for e in range(E):
        i = int(row[e]); idx_by_src.setdefault(i, []).append(e)
    for i, eidxs in idx_by_src.items():
        k = max(1, int(len(eidxs) * p_keep))
        topk = torch.topk(w[eidxs], k).indices
        keep_eidx = [eidxs[int(t)] for t in topk]
        keep[keep_eidx] = True

    new_ei = edge_index[:, keep]
    new_w  = w[keep]
    # 镜像反向边（增强无向性）
    new_ei = torch.cat([new_ei, new_ei.flip(0)], dim=1)
    new_w  = torch.cat([new_w,  new_w], dim=0)
    return new_ei, new_w

edge_index, edge_weight = gnnguard_reweight(x, edge_index, p_keep=0.7, alpha=2.0)
print(f"✅ GNNGuard 后：edges={edge_index.shape[1]}")

# ========================
# 读取标签 & 分层划分 (70/15/15)
# ========================
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
    raise ValueError("没有可用标签（全部缺失）")

le = LabelEncoder()
y_full = np.full(len(node_ids), -1, dtype=int)
y_full[labeled_nodes] = le.fit_transform(labels.loc[keep, "subtype"].values)
classes = list(le.classes_)
y = torch.tensor(y_full, dtype=torch.long)
print(f"✅ 读取标签：{len(classes)} 类 → {classes}")

# 分层划分（70/15/15）
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
print(f"✅ 监督训练（分层划分）：train={train_mask.sum().item()}, val={val_mask.sum().item()}, test={test_mask.sum().item()}")

# 类别权重（按训练集频次的反比，防止长尾）
train_labels = y[train_mask].numpy()
cls, cnt = np.unique(train_labels, return_counts=True)
weights = np.zeros(len(classes), dtype=np.float32) + 1.0
for c, c_cnt in zip(cls, cnt):
    weights[c] = float(len(train_labels)) / (len(cls) * c_cnt)
class_weights = torch.tensor(weights, dtype=torch.float32)
print(f"✅ 类别权重：{class_weights.numpy()}")

# ========================
# GraphSAGE 模型
# ========================
class SageNet(nn.Module):
    def __init__(self, in_ch, hid=48, out_ch=4, drop=0.6):
        super().__init__()
        self.conv1 = SAGEConv(in_ch, hid)
        self.bn1   = nn.BatchNorm1d(hid)
        self.conv2 = SAGEConv(hid, hid)
        self.bn2   = nn.BatchNorm1d(hid)
        self.out   = nn.Linear(hid, out_ch)
        self.drop  = drop
    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.drop, training=self.training)
        return self.out(x)
    def embeddings(self, x, edge_index):
        with torch.no_grad():
            h = F.relu(self.bn1(self.conv1(x, edge_index)))
            h = F.relu(self.bn2(self.conv2(h, edge_index)))
        return h

def accuracy(logits, labels, mask):
    if mask.sum()==0: return float("nan")
    pred = logits[mask].argmax(dim=1)
    return (pred == labels[mask]).float().mean().item()
# === 新增：逐类准确率 ===
def per_class_accuracy(logits, labels, mask, classes):
    """
    各类别上的“正确率”（= 该类被判对的比例，等价于 classification_report 的 recall）。
    当某个切分里没有该类样本时，返回 NaN。
    """
    if mask.sum() == 0:
        return {c: float("nan") for c in classes}
    pred = logits[mask].argmax(dim=1).cpu().numpy()
    true = labels[mask].cpu().numpy()
    acc = {}
    for ci, cname in enumerate(classes):
        sel = (true == ci)
        if sel.sum() == 0:
            acc[cname] = float("nan")
        else:
            acc[cname] = float((pred[sel] == ci).mean())
    return acc

def _fmt_pc(pc_dict):
    import numpy as np
    return " | ".join(
        [f"{k}:{(v if not np.isnan(v) else float('nan')):.3f}" for k, v in pc_dict.items()]
    )

# ========================
# 训练（早停）
# ========================
model = SageNet(x.shape[1], hid=HID_DIM, out_ch=len(classes), drop=DROPOUT)
opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
best_val, best_state, wait = 0.0, None, 0

for epoch in range(1, MAX_EPOCHS+1):
    model.train()
    opt.zero_grad()
    out  = model(x, edge_index, None)
    loss = F.cross_entropy(out[train_mask], y[train_mask], weight=class_weights)
    loss.backward(); opt.step()

    if epoch % 10 == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            logits = model(x, edge_index, None)
            tr = accuracy(logits, y, train_mask)
            va = accuracy(logits, y, val_mask)
            print(f"Epoch {epoch:03d} | Loss {loss.item():.4f} | Train {tr:.3f} | Val {va:.3f} | BestVal {best_val:.3f}")
            if va > best_val:
                best_val  = va
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= PATIENCE:
                    print(f"⏹ 早停于 Epoch {epoch}（Best Val={best_val:.3f}）")
                    break

# 加载最佳权重并最终评估
if best_state is not None:
    model.load_state_dict(best_state)

model.eval()
with torch.no_grad():
    logits = model(x, edge_index, None)
    train_acc = accuracy(logits, y, train_mask)
    val_acc   = accuracy(logits, y, val_mask)
    test_acc  = accuracy(logits, y, test_mask)

    # 逐类准确率（=各类别召回率）
    pc_train = per_class_accuracy(logits, y, train_mask, classes)
    pc_val   = per_class_accuracy(logits, y, val_mask,   classes)
    pc_test  = per_class_accuracy(logits, y, test_mask,  classes)

print(f"✅ 最终评估 | Train {train_acc:.3f} | Val {val_acc:.3f} | Test {test_acc:.3f}")
print("📊 各类准确率（Train）:", _fmt_pc(pc_train))
print("📊 各类准确率（Val）  :", _fmt_pc(pc_val))
print("📊 各类准确率（Test） :", _fmt_pc(pc_test))


# ========================
# 保存输出
# ========================
# 节点嵌入
# emb = models.embeddings(x, edge_index).cpu().numpy()
# pd.DataFrame(emb, index=node_ids).to_csv("../other/node_embeddings.tsv", sep="\t")
# print("💾 已保存：node_embeddings.tsv")
#
# # 测试集预测明细（含各类概率）
# proba = F.softmax(logits, dim=1).cpu().numpy()
# pred  = logits.argmax(dim=1).cpu().numpy()
# inv_labels = np.array(classes)
#
# test_rows = np.where(test_mask.cpu().numpy())[0]
# pred_df = pd.DataFrame({
#     "sample":   [node_ids[i] for i in test_rows],
#     "true":     [inv_labels[y[i].item()]   for i in test_rows],
#     "pred":     [inv_labels[pred[i]]       for i in test_rows],
# })
# for ci, cname in enumerate(classes):
#     pred_df[f"prob_{cname}"] = proba[test_rows, ci]
# pred_df.to_csv("predictions_test.csv", index=False)
# print("💾 已保存：predictions_test.csv")
#
# # 汇总指标 + 分类报告 + 混淆矩阵（测试集）
# y_true = [inv_labels[y[i].item()] for i in test_rows]
# y_pred = [inv_labels[pred[i]]     for i in test_rows]
# report = classification_report(y_true, y_pred, labels=classes, digits=3, output_dict=True)
# cm = confusion_matrix(y_true, y_pred, labels=classes).tolist()
#
# summary = {
#     "seed": SEED,
#     "classes": classes,
#     "train_acc": float(train_acc),
#     "val_acc":   float(val_acc),
#     "test_acc":  float(test_acc),
#     "classification_report": report,
#     "confusion_matrix": cm
# }
# with open("../other/metrics_summary.txt", "w", encoding="utf-8") as f:
#     f.write(json.dumps(summary, ensure_ascii=False, indent=2))
# print("💾 已保存：metrics_summary.txt")
# print("🎉 完成！")
