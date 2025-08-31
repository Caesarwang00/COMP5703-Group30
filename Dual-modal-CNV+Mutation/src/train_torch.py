
# -*- coding: utf-8 -*-
"""
PyTorch 版：CNV + Mutation 早期融合，多类分类
- 5 折分层交叉验证，AdamW + CosineLR + EarlyStopping
- hidden_dim=0 => Logistic；>0 => 一层 MLP
"""
import os, sys, numpy as np
import torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import joblib

try:
    from . import config
    from .load_data import align_modalities
    from . import preprocess as PP
    from .utils import ensure_dir, print_and_save_report
except Exception:
    import config
    from load_data import align_modalities
    import preprocess as PP
    from utils import ensure_dir, print_and_save_report

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

# ---- 超参 ----
SEED           = getattr(config, "SEED", 42)
N_SPLITS       = getattr(config, "N_SPLITS", 5)
CNV_ENCODING   = getattr(config, "CNV_ENCODING", "numeric")
CNV_NZ_FRAC    = getattr(config, "CNV_NONZERO_FRAC", 0.05)
MUT_MIN_FRAC   = getattr(config, "MUT_MIN_FRAC", 0.01)
MUT_MIN_ABS    = getattr(config, "MUT_MIN_ABS", 3)
BW_CNV         = getattr(config, "BLOCK_WEIGHT_CNV", 1.0)
BW_MUT         = getattr(config, "BLOCK_WEIGHT_MUT", 1.0)
TOPK_TOTAL     = getattr(config, "TOPK_TOTAL", 0)
MAX_EPOCHS     = getattr(config, "EPOCHS", 60)
BATCH_SIZE     = getattr(config, "BATCH_SIZE", 64)
LR             = getattr(config, "LR", 1e-3)
WEIGHT_DECAY   = getattr(config, "WEIGHT_DECAY", 1e-4)
PATIENCE       = getattr(config, "PATIENCE", 8)
DROPOUT        = getattr(config, "DROPOUT", 0.2)
HIDDEN_DIM     = getattr(config, "HIDDEN_DIM", 0)
LABEL_SMOOTH   = getattr(config, "LABEL_SMOOTH", 0.0)
SCHEDULER      = getattr(config, "SCHEDULER", "cos")
DEVICE_CFG     = getattr(config, "DEVICE", "auto")
USE_TQDM       = getattr(config, "USE_TQDM", True)

# ---- 工具 ----
def get_device():
    if DEVICE_CFG == "cpu":
        return torch.device("cpu")
    if DEVICE_CFG == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=SEED):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class MLPClassifier(nn.Module):
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

# ---- 训练单折 ----
def train_one_fold(Xtr, ytr, Xva, yva, labels, fold_dir, device):
    from torch.utils.data import TensorDataset, DataLoader
    n_class = len(labels); in_dim = Xtr.shape[1]

    # 张量（先在 CPU 上）
    Xtr_t = torch.from_numpy(Xtr.astype(np.float32))
    ytr_t = torch.from_numpy(ytr.astype(np.int64))
    Xva_t = torch.from_numpy(Xva.astype(np.float32))
    yva_t = torch.from_numpy(yva.astype(np.int64))

    pin = (device.type == "cuda")
    train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(TensorDataset(Xva_t, yva_t), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin)

    model = MLPClassifier(in_dim, n_class, hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss(weight=make_class_weight(ytr, n_class).to(device), label_smoothing=float(LABEL_SMOOTH))
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=MAX_EPOCHS) if SCHEDULER == "cos" else None

    best_f1, best_state, no_improve = -1.0, None, 0
    iterator = range(1, MAX_EPOCHS + 1)
    if USE_TQDM and tqdm is not None: iterator = tqdm(iterator, desc="train", leave=False)

    for epoch in iterator:
        model.train(); tot = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step(); tot += loss.item() * xb.size(0)
        if sched is not None: sched.step()

        # 验证
        model.eval(); preds=[]; y_true=[]
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
                pred = torch.argmax(model(xb), dim=1)
                preds.append(pred.cpu().numpy()); y_true.append(yb.cpu().numpy())
        y_pred = np.concatenate(preds); y_true=np.concatenate(y_true)
        f1m = f1_score(y_true, y_pred, average="macro")

        if USE_TQDM and tqdm is not None: iterator.set_postfix({"loss": f"{tot/len(train_loader.dataset):.4f}", "mF1": f"{f1m:.4f}"})
        else: print(f"[epoch {epoch:03d}] loss={tot/len(train_loader.dataset):.4f} mF1={f1m:.4f}")

        # early stop
        if f1m > best_f1 + 1e-6:
            best_f1 = f1m; no_improve = 0
            best_state = {"model": model.state_dict(), "meta": {"in_dim": in_dim, "n_class": n_class, "hidden_dim": HIDDEN_DIM}}
        else:
            no_improve += 1
            if no_improve >= PATIENCE: break

    if best_state is not None: model.load_state_dict(best_state["model"])

    # 输出验证集预测
    model.eval(); preds=[]
    with torch.no_grad():
        for xb,_ in val_loader:
            xb = xb.to(device, non_blocking=True)
            preds.append(torch.argmax(model(xb), dim=1).cpu().numpy())
    y_pred = np.concatenate(preds)
    torch.save(best_state if best_state is not None else {"model": model.state_dict()}, os.path.join(fold_dir, "model.pt"))
    return y_pred

# ---- 主流程：对齐 -> 筛特征 -> 编码 -> 标准化 -> 融合 -> 训练CV ----
def train_eval_cv():
    # 设备与随机种子
    set_seed(); device = get_device()
    print(f"[INFO] 使用设备: {device}")

    # 1) 读取并对齐
    cnv_df, mut_df, y_series, sample_order = align_modalities()
    print(f"[INFO] 样本数={len(sample_order)} | CNV={cnv_df.shape} | MUT={mut_df.shape}")

    # 2) 标签编码
    le = LabelEncoder()
    if y_series is None or len(y_series) == 0:
        raise RuntimeError("未找到 GeneExp_Subtype 标签，请检查临床文件")
    y_enc = le.fit_transform(y_series.values)
    labels = list(le.classes_)
    print("[INFO] 类别顺序:", labels)

    # 3) 分层 CV
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.arange(len(sample_order)), y_enc))
    oof_pred = np.zeros_like(y_enc)
    ensure_dir(config.MODEL_DIR); ensure_dir(config.OUTPUT_DIR)

    for fold,(tr_idx, va_idx) in enumerate(splits, 1):
        print(f"\n========== [Torch] Fold {fold}/{N_SPLITS} ==========")
        tr_ids = [sample_order[i] for i in tr_idx]
        va_ids = [sample_order[i] for i in va_idx]

        # 4) 筛特征（只在训练折上统计）
        keep_mut = PP.filter_mutation_lowfreq(mut_df, tr_ids, min_frac=MUT_MIN_FRAC, min_abs=MUT_MIN_ABS)
        keep_cnv = PP.filter_cnv_nonzero_frac(cnv_df, tr_ids, nz_frac=CNV_NZ_FRAC)
        print(f"[F{fold}] 保留突变基因: {len(keep_mut)} | 保留CNV基因: {len(keep_cnv)}")

        # 5) 编码
        # CNV
        if CNV_ENCODING == "onehot":
            Xtr_cnv, enc = PP.encode_cnv_onehot(cnv_df, tr_ids, keep_cnv)
            Xva_cnv, _   = PP.encode_cnv_onehot(cnv_df, va_ids, keep_cnv)
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

        # 7) 块加权
        Xtr_cnv_w = PP.apply_block_weight(Xtr_cnv_s, BW_CNV)
        Xtr_mut_w = PP.apply_block_weight(Xtr_mut_s, BW_MUT)
        Xva_cnv_w = PP.apply_block_weight(Xva_cnv_s, BW_CNV)
        Xva_mut_w = PP.apply_block_weight(Xva_mut_s, BW_MUT)

        # 8) 早期融合
        Xtr = PP.concat_blocks(Xtr_cnv_w, Xtr_mut_w)
        Xva = PP.concat_blocks(Xva_cnv_w, Xva_mut_w)
        ytr = y_enc[tr_idx]; yva = y_enc[va_idx]
        print(f"[F{fold}] 融合后维度={Xtr.shape[1]} | 训练/验证={Xtr.shape[0]}/{Xva.shape[0]}")

        # 9) 可选：融合后全局 Top-K
        skb = getattr(PP, "select_kbest_global", None)
        sel = None
        if skb is not None and TOPK_TOTAL and TOPK_TOTAL > 0:
            Xtr, Xva, sel = PP.select_kbest_global(Xtr, ytr, Xva, TOPK_TOTAL)
            print(f"[F{fold}] 全局TopK选择后维度={Xtr.shape[1]} (K={TOPK_TOTAL})")

        # 10) 训练
        fold_dir = os.path.join(config.MODEL_DIR, f"fold{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        y_pred = train_one_fold(Xtr, ytr, Xva, yva, labels, fold_dir, device)
        oof_pred[va_idx] = y_pred

        # 11) 保存预处理器
        joblib.dump(sc_cnv, os.path.join(fold_dir, "scaler_cnv.joblib"))
        joblib.dump(sc_mut, os.path.join(fold_dir, "scaler_mut.joblib"))
        if sel is not None:
            joblib.dump(sel, os.path.join(fold_dir, "selector_topk.joblib"))

        # 12) 报告
        print_and_save_report(yva, y_pred, labels, config.OUTPUT_DIR, f"fold{fold}")

    # 13) OOF 总体
    print_and_save_report(y_enc, oof_pred, labels, config.OUTPUT_DIR, "OOF")

if __name__ == "__main__":
    train_eval_cv()
