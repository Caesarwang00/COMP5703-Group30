# -*- coding: utf-8 -*-
"""
train_torch.py
- PyTorch GPU 版：CNV + Mutation 早期融合，多类分类
- 5 折分层交叉验证，AdamW + CosineLR + EarlyStopping
- 兼容：若未定义的超参从 config 读取，否则使用合理默认值
"""

import os, sys, time, json, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- 兼容导入 config 与项目内模块 ----
try:
    import config
except ModuleNotFoundError:
    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.dirname(HERE)
    if ROOT not in sys.path:
        sys.path.append(ROOT)
    import config

try:
    from .load_data import align_modalities
    from . import preprocess as PP
    from .utils import ensure_dir, print_and_save_report, save_json
except Exception:
    from load_data import align_modalities
    import preprocess as PP
    from utils import ensure_dir, print_and_save_report, save_json

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
import joblib

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

# ============ 默认/回落超参 ============
SEED           = getattr(config, "SEED", 42)
N_SPLITS       = getattr(config, "N_SPLITS", 5)
CNV_ENCODING   = getattr(config, "CNV_ENCODING", "numeric")
CNV_NZ_FRAC    = getattr(config, "CNV_NONZERO_FRAC", 0.05)
MUT_MIN_FRAC   = getattr(config, "MUT_MIN_FRAC", 0.01)
MUT_MIN_ABS    = getattr(config, "MUT_MIN_ABS", 3)
BW_CNV         = getattr(config, "BLOCK_WEIGHT_CNV", getattr(config, "BLOCK_WEIGHT_CNV", 1.0))
BW_MUT         = getattr(config, "BLOCK_WEIGHT_MUT", getattr(config, "BLOCK_WEIGHT_MUT", 1.0))
TOPK_TOTAL     = getattr(config, "TOPK_TOTAL", 0)     # 0 表示不做全局TopK
MAX_EPOCHS     = getattr(config, "EPOCHS", 60)
BATCH_SIZE     = getattr(config, "BATCH_SIZE", 64)
LR             = getattr(config, "LR", 1e-3)
WEIGHT_DECAY   = getattr(config, "WEIGHT_DECAY", 1e-4)
PATIENCE       = getattr(config, "PATIENCE", 8)
DROPOUT        = getattr(config, "DROPOUT", 0.2)
HIDDEN_DIM     = getattr(config, "HIDDEN_DIM", 0)     # 0=纯Logistic；>0=一层MLP
LABEL_SMOOTH   = getattr(config, "LABEL_SMOOTH", 0.0)
SCHEDULER      = getattr(config, "SCHEDULER", "cos")  # 'cos' 或 None
DEVICE_CFG     = getattr(config, "DEVICE", "auto")     # 'auto' / 'cuda' / 'cpu'
USE_TQDM       = getattr(config, "USE_TQDM", True)

# ============ 小工具 ============
def get_device():
    if DEVICE_CFG == "cpu":
        return torch.device("cpu")
    if DEVICE_CFG == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class MLPClassifier(nn.Module):
    """一层隐藏层（可选）的多类分类器；hidden_dim=0 则退化为 Logistic 回归"""
    def __init__(self, in_dim, num_classes, hidden_dim=0, dropout=0.2):
        super().__init__()
        if hidden_dim and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.net = nn.Linear(in_dim, num_classes)
    def forward(self, x):
        return self.net(x)

def make_class_weight(y, n_class):
    counts = np.bincount(y, minlength=n_class).astype(np.float32)
    counts[counts == 0] = 1.0
    w = counts.sum() / counts
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)

def to_tensor(x, device):
    return torch.from_numpy(x.astype(np.float32)).to(device, non_blocking=True)

def train_one_fold(Xtr, ytr, Xva, yva, labels, fold_dir, device):
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    n_class = len(labels)
    in_dim  = Xtr.shape[1]

    # --- 1) 先在 CPU 上构造张量（不要 .to(device)） ---
    Xtr_t = torch.from_numpy(Xtr.astype(np.float32))           # CPU
    ytr_t = torch.from_numpy(ytr.astype(np.int64))             # CPU
    Xva_t = torch.from_numpy(Xva.astype(np.float32))           # CPU
    yva_t = torch.from_numpy(yva.astype(np.int64))             # CPU

    # --- 2) DataLoader：GPU 时打开 pin_memory，但数据仍在 CPU ---
    pin = (device.type == "cuda")
    train_ds = TensorDataset(Xtr_t, ytr_t)
    val_ds   = TensorDataset(Xva_t, yva_t)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=pin)

    # --- 3) 模型与优化器 ---
    model = MLPClassifier(in_dim, n_class, hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(device)
    class_weight = make_class_weight(ytr, n_class).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=float(LABEL_SMOOTH))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
                 if SCHEDULER == "cos" else None)

    # --- 4) 训练循环（在循环里把每个 batch 搬到 GPU） ---
    best_f1, best_state, no_improve = -1.0, None, 0
    iterator = range(1, MAX_EPOCHS + 1)
    if USE_TQDM and tqdm is not None:
        iterator = tqdm(iterator, desc="train", leave=False)

    for epoch in iterator:
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)  # ← 此处搬到 GPU
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        if scheduler is not None:
            scheduler.step()

        # 验证
        model.eval()
        preds, y_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)          # ← 验证也搬
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                pred = torch.argmax(logits, dim=1)
                preds.append(pred.cpu().numpy())
                y_true.append(yb.cpu().numpy())
        y_pred = np.concatenate(preds); y_true = np.concatenate(y_true)
        acc = (y_pred == y_true).mean()
        f1m = f1_score(y_true, y_pred, average="macro")

        if USE_TQDM and tqdm is not None:
            iterator.set_postfix({"loss": f"{total_loss/len(train_ds):.4f}",
                                  "acc": f"{acc:.4f}", "mF1": f"{f1m:.4f}"})
        else:
            print(f"  [epoch {epoch:03d}] loss={total_loss/len(train_ds):.4f} acc={acc:.4f} mF1={f1m:.4f}")

        # early stopping
        if f1m > best_f1 + 1e-6:
            best_f1 = f1m
            no_improve = 0
            best_state = { "model": model.state_dict(),
                           "meta": { "in_dim": in_dim, "n_class": n_class,
                                     "hidden_dim": HIDDEN_DIM, "dropout": DROPOUT } }
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    # 用最佳状态重算验证集预测并保存
    if best_state is not None:
        model.load_state_dict(best_state["model"])
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    y_pred = np.concatenate(preds)

    ensure_dir(fold_dir)
    torch.save(best_state if best_state is not None else {"model": model.state_dict()},
               os.path.join(fold_dir, "model.pt"))
    return y_pred

def train_eval_cv():
    set_seed()
    device = get_device()
    print(f"[INFO] 使用设备: {device}")

    # 1) 读取对齐
    cnv_df, mut_df, y_series, sample_order = align_modalities()
    print(f"[INFO] 对齐后样本数: {len(sample_order)} | CNV形状={cnv_df.shape} | MUT形状={mut_df.shape}")

    # 2) 标签编码
    le = LabelEncoder()
    y_enc = le.fit_transform(y_series.values)
    labels = list(le.classes_)
    print("[INFO] 类别顺序:", labels)

    # 3) 分层 CV
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.arange(len(sample_order)), y_enc))
    oof_pred = np.zeros_like(y_enc)

    ensure_dir(config.MODEL_DIR); ensure_dir(config.OUTPUT_DIR)

    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        print(f"\n========== [Torch] Fold {fold} / {N_SPLITS} ==========")
        tr_ids = [sample_order[i] for i in tr_idx]
        va_ids = [sample_order[i] for i in va_idx]

        # 4) 训练折内筛特征
        keep_mut = PP.filter_mutation_lowfreq(mut_df, tr_ids, min_frac=MUT_MIN_FRAC, min_abs=MUT_MIN_ABS)
        keep_cnv = PP.filter_cnv_nonzero_frac(cnv_df, tr_ids, nz_frac=CNV_NZ_FRAC)
        print(f"[F{fold}] 保留突变基因: {len(keep_mut)} | 保留CNV基因: {len(keep_cnv)}")

        # 5) 编码
        # CNV
        if CNV_ENCODING == "onehot":
            Xtr_cnv, _ = PP.encode_cnv_onehot(cnv_df, tr_ids, keep_cnv)
            Xva_cnv, _ = PP.encode_cnv_onehot(cnv_df, va_ids, keep_cnv)
        else:
            Xtr_cnv, _ = PP.encode_cnv_numeric(cnv_df, tr_ids, keep_cnv)
            Xva_cnv, _ = PP.encode_cnv_numeric(cnv_df, va_ids, keep_cnv)
        # MUT
        Xtr_mut = mut_df.loc[keep_mut, tr_ids].T.values.astype(np.float32)
        Xva_mut = mut_df.loc[keep_mut, va_ids].T.values.astype(np.float32)

        # 6) 标准化（按块）
        Xtr_cnv_s, sc_cnv = PP.standardize_fit(Xtr_cnv)
        Xva_cnv_s = PP.standardize_apply(Xva_cnv, sc_cnv)
        Xtr_mut_s, sc_mut = PP.standardize_fit(Xtr_mut)
        Xva_mut_s = PP.standardize_apply(Xva_mut, sc_mut)

        # 7) 分块加权
        Xtr_cnv_w = PP.apply_block_weight(Xtr_cnv_s, BW_CNV)
        Xtr_mut_w = PP.apply_block_weight(Xtr_mut_s, BW_MUT)
        Xva_cnv_w = PP.apply_block_weight(Xva_cnv_s, BW_CNV)
        Xva_mut_w = PP.apply_block_weight(Xva_mut_s, BW_MUT)

        # 8) 早期融合
        Xtr = PP.concat_blocks(Xtr_cnv_w, Xtr_mut_w)
        Xva = PP.concat_blocks(Xva_cnv_w, Xva_mut_w)
        ytr = y_enc[tr_idx]; yva = y_enc[va_idx]
        print(f"[F{fold}] 融合后维度={Xtr.shape[1]} | 训练/验证={Xtr.shape[0]}/{Xva.shape[0]}")

        # 9) 可选：融合后全局 Top-K（如果你在 preprocess 里实现了 select_kbest_global）
        skb = getattr(PP, "select_kbest_global", None)
        sel = None
        if skb is not None and TOPK_TOTAL and TOPK_TOTAL > 0:
            Xtr, Xva, sel = PP.select_kbest_global(Xtr, ytr, Xva, TOPK_TOTAL)
            print(f"[F{fold}] 全局TopK选择后维度={Xtr.shape[1]} (K={TOPK_TOTAL})")

        # 10) 训练（GPU）
        fold_dir = os.path.join(config.MODEL_DIR, f"fold{fold}")
        y_pred = train_one_fold(Xtr, ytr, Xva, yva, labels, fold_dir, device)
        oof_pred[va_idx] = y_pred

        # 11) 保存这折的预处理器
        joblib.dump(sc_cnv, os.path.join(fold_dir, "scaler_cnv.joblib"))
        joblib.dump(sc_mut, os.path.join(fold_dir, "scaler_mut.joblib"))
        if sel is not None:
            joblib.dump(sel, os.path.join(fold_dir, "selector_topk.joblib"))

        # 12) 本折报告
        print_and_save_report(yva, y_pred, labels, config.OUTPUT_DIR, f"fold{fold}")

    # 13) OOF 总体
    print_and_save_report(y_enc, oof_pred, labels, config.OUTPUT_DIR, "OOF")

if __name__ == "__main__":
    train_eval_cv()
