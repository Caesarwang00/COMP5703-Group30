# -*- coding: utf-8 -*-
"""
三模态早期融合（CNV + Methyl450 + RNAseq）
- 单模态：筛选→标准化→(可选)PCA降维→加权
- 融合：拼接后（可选）全局TopK
- 训练：5折CV + AdamW + CosineLR + EarlyStopping + 均衡采样 + (可选)FocalLoss
"""
import os, numpy as np, torch, torch.nn as nn
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

from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

# ===== 超参 =====
SEED          = getattr(config, "SEED", 42)
N_SPLITS      = getattr(config, "N_SPLITS", 5)
MAX_EPOCHS    = getattr(config, "EPOCHS", 80)
BATCH_SIZE    = getattr(config, "BATCH_SIZE", 64)
LR            = getattr(config, "LR", 1e-3)
WEIGHT_DECAY  = getattr(config, "WEIGHT_DECAY", 5e-4)
PATIENCE      = getattr(config, "PATIENCE", 10)
DROPOUT       = getattr(config, "DROPOUT", 0.5)
HIDDEN_DIM    = getattr(config, "HIDDEN_DIM", 512)
SCHEDULER     = getattr(config, "SCHEDULER", "cos")
DEVICE_CFG    = getattr(config, "DEVICE", "auto")
USE_TQDM      = getattr(config, "USE_TQDM", True)

USE_CNV       = getattr(config, "USE_CNV", True)
USE_METH      = getattr(config, "USE_METH", True)
USE_RNA       = getattr(config, "USE_RNA", True)

CNV_ENCODING  = getattr(config, "CNV_ENCODING", "numeric")
BW_CNV        = getattr(config, "BLOCK_WEIGHT_CNV", 1.0)
BW_METH       = getattr(config, "BLOCK_WEIGHT_METH", 1.0)
BW_RNA        = getattr(config, "BLOCK_WEIGHT_RNA", 1.0)

CNV_NZ_FRAC   = getattr(config, "CNV_NONZERO_FRAC", 0.05)
METH_MIN_STD  = getattr(config, "METH_MIN_STD", 0.02)
METH_TOPK_VAR = getattr(config, "METH_TOPK_VAR", 20000)
RNA_MIN_STD   = getattr(config, "RNA_MIN_STD", 0.1)
TOPK_TOTAL    = getattr(config, "TOPK_TOTAL", 0)

USE_DIMRED    = getattr(config, "USE_DIMRED", True)
CNV_PCA_N     = getattr(config, "CNV_PCA_N", 256)
METH_PCA_N    = getattr(config, "METH_PCA_N", 256)
RNA_PCA_N     = getattr(config, "RNA_PCA_N", 256)

SAMPLER       = getattr(config, "SAMPLER", "class_balance")
LOSS_NAME     = getattr(config, "LOSS", "focal")
FOCAL_GAMMA   = getattr(config, "FOCAL_GAMMA", 2.0)

# ===== 工具 =====
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
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class MLPClassifier(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_dim=0, dropout=0.5):
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

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma; self.weight = weight; self.reduction = reduction
    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1-pt) ** self.gamma) * ce
        if self.reduction == "mean": return loss.mean()
        if self.reduction == "sum":  return loss.sum()
        return loss

def build_criterion(name, class_weight, gamma):
    if name.lower() == "focal": return FocalLoss(gamma=gamma, weight=class_weight)
    return nn.CrossEntropyLoss(weight=class_weight)

def make_class_weight(y, n_class):
    counts = np.bincount(y, minlength=n_class).astype(np.float32)
    counts[counts == 0] = 1.0
    w = counts.sum() / counts
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)

def _cap_pca_n(preset, X):
    """
    根据当折训练矩阵 X 的形状自动裁剪 PCA 维度：
    n_components ≤ min(n_samples-1, n_features)
    """
    n_samples, n_features = X.shape
    max_allowed = min(max(1, n_samples - 1), n_features)
    n = int(min(preset, max_allowed))
    return max(0, n)  # 返回0表示不做PCA

# ===== 训练一折 =====
def train_one_fold(Xtr, ytr, Xva, yva, labels, fold_dir, device):
    n_class = len(labels); in_dim = Xtr.shape[1]
    Xtr_t = torch.from_numpy(Xtr.astype(np.float32))
    ytr_t = torch.from_numpy(ytr.astype(np.int64))
    Xva_t = torch.from_numpy(Xva.astype(np.float32))
    yva_t = torch.from_numpy(yva.astype(np.int64))

    pin = (device.type == "cuda")
    train_ds = TensorDataset(Xtr_t, ytr_t)
    if SAMPLER == "class_balance":
        counts = np.bincount(ytr, minlength=n_class)
        weights = 1.0 / np.clip(counts, 1, None)
        sample_w = torch.tensor([weights[c] for c in ytr], dtype=torch.float32)
        sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=pin)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(TensorDataset(Xva_t, yva_t), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin)

    model = MLPClassifier(in_dim, n_class, hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(device)
    class_w = make_class_weight(ytr, n_class).to(device)
    criterion = build_criterion(LOSS_NAME, class_w, FOCAL_GAMMA)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=MAX_EPOCHS) if SCHEDULER == "cos" else None

    best_f1, best_state, no_improve = -1.0, None, 0
    iterator = range(1, MAX_EPOCHS + 1)
    if USE_TQDM and tqdm is not None: iterator = tqdm(iterator, desc="train", leave=False)

    for epoch in iterator:
        model.train(); tot=0.0
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
        y_pred = np.concatenate(preds); y_true = np.concatenate(y_true)
        f1m = f1_score(y_true, y_pred, average="macro")

        if USE_TQDM and tqdm is not None: iterator.set_postfix({"loss": f"{tot/len(train_loader.dataset):.4f}", "mF1": f"{f1m:.4f}"})
        else: print(f"[epoch {epoch:03d}] loss={tot/len(train_loader.dataset):.4f} mF1={f1m:.4f}")

        if f1m > best_f1 + 1e-6:
            best_f1 = f1m; no_improve = 0
            best_state = {"model": model.state_dict(), "meta": {"in_dim": in_dim, "n_class": n_class, "hidden_dim": HIDDEN_DIM, "dropout": DROPOUT}}
        else:
            no_improve += 1
            if no_improve >= PATIENCE: break

    if best_state is not None: model.load_state_dict(best_state["model"])

    # 验证集预测输出
    model.eval(); preds=[]
    with torch.no_grad():
        for xb,_ in val_loader:
            xb = xb.to(device, non_blocking=True)
            preds.append(torch.argmax(model(xb), dim=1).cpu().numpy())
    y_pred = np.concatenate(preds)

    # 保存模型
    torch.save(best_state if best_state is not None else {"model": model.state_dict()}, os.path.join(fold_dir, "model.pt"))
    return y_pred

# ===== 主流程 =====
def train_eval_cv():
    set_seed(); device = get_device()
    print(f"[INFO] 使用设备: {device}")

    cnv_df, meth_df, rna_df, y_series, sample_order = align_modalities()
    print(f"[INFO] 对齐后样本={len(sample_order)} | CNV={None if cnv_df is None else cnv_df.shape} | METH={None if meth_df is None else meth_df.shape} | RNA={None if rna_df is None else rna_df.shape}")

    le = LabelEncoder()
    if y_series is None or len(y_series)==0:
        raise RuntimeError("未找到 GeneExp_Subtype 标签")
    y_enc = le.fit_transform(y_series.values); labels=list(le.classes_)
    print("[INFO] 类别顺序:", labels)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.arange(len(sample_order)), y_enc))
    oof_pred = np.zeros_like(y_enc)
    ensure_dir(config.MODEL_DIR); ensure_dir(config.OUTPUT_DIR)

    for fold,(tr_idx,va_idx) in enumerate(splits, 1):
        print(f"\n========== [Torch] Fold {fold}/{N_SPLITS} ==========")
        tr_ids = [sample_order[i] for i in tr_idx]
        va_ids = [sample_order[i] for i in va_idx]

        blocks_tr, blocks_va = [], []

        # --- CNV ---
        if USE_CNV and cnv_df is not None:
            keep_cnv = PP.filter_cnv_nonzero_frac(cnv_df, tr_ids, nz_frac=CNV_NZ_FRAC)
            if CNV_ENCODING == "onehot":
                Xtr_cnv, _ = PP.encode_cnv_onehot(cnv_df, tr_ids, keep_cnv)
                Xva_cnv, _ = PP.encode_cnv_onehot(cnv_df, va_ids, keep_cnv)
                Xtr_cnv_s, Xva_cnv_s = Xtr_cnv, Xva_cnv
                sc_cnv = None
            else:
                Xtr_cnv, _ = PP.encode_cnv_numeric(cnv_df, tr_ids, keep_cnv)
                Xva_cnv, _ = PP.encode_cnv_numeric(cnv_df, va_ids, keep_cnv)
                Xtr_cnv_s, sc_cnv = PP.standardize_fit(Xtr_cnv)
                Xva_cnv_s = PP.standardize_apply(Xva_cnv, sc_cnv)

            # 动态裁剪 PCA 维度
            n_cnv = _cap_pca_n(CNV_PCA_N, Xtr_cnv_s)
            if USE_DIMRED and n_cnv > 0:
                print(f"[F{fold}] CNV PCA: preset={CNV_PCA_N} -> use={n_cnv}")
                Xtr_cnv_s, pca_cnv = PP.pca_fit(Xtr_cnv_s, n_components=n_cnv)
                Xva_cnv_s = PP.pca_apply(Xva_cnv_s, pca_cnv)
            else:
                pca_cnv = None

            Xtr_cnv_w = PP.apply_block_weight(Xtr_cnv_s, BW_CNV)
            Xva_cnv_w = PP.apply_block_weight(Xva_cnv_s, BW_CNV)
            blocks_tr.append(Xtr_cnv_w); blocks_va.append(Xva_cnv_w)
        else:
            sc_cnv = pca_cnv = None

        # --- METH ---
        if USE_METH and meth_df is not None:
            keep_meth = PP.filter_meth_by_var(
                meth_df, tr_ids,
                topk_var=METH_TOPK_VAR,
                min_std=METH_MIN_STD if METH_TOPK_VAR <= 0 else None
            )
            Xtr_meth = meth_df.loc[keep_meth, tr_ids].T.values.astype(np.float32)
            Xva_meth = meth_df.loc[keep_meth, va_ids].T.values.astype(np.float32)
            Xtr_meth_s, sc_meth = PP.standardize_fit(Xtr_meth)
            Xva_meth_s = PP.standardize_apply(Xva_meth, sc_meth)

            n_meth = _cap_pca_n(METH_PCA_N, Xtr_meth_s)
            if USE_DIMRED and n_meth > 0:
                print(f"[F{fold}] METH PCA: preset={METH_PCA_N} -> use={n_meth}")
                Xtr_meth_s, pca_meth = PP.pca_fit(Xtr_meth_s, n_components=n_meth)
                Xva_meth_s = PP.pca_apply(Xva_meth_s, pca_meth)
            else:
                pca_meth = None

            Xtr_meth_w = PP.apply_block_weight(Xtr_meth_s, BW_METH)
            Xva_meth_w = PP.apply_block_weight(Xva_meth_s, BW_METH)
            blocks_tr.append(Xtr_meth_w); blocks_va.append(Xva_meth_w)
        else:
            sc_meth = pca_meth = None

        # --- RNA ---
        if USE_RNA and rna_df is not None:
            keep_rna = PP.filter_rna_lowvar(rna_df, tr_ids, min_std=RNA_MIN_STD)
            Xtr_rna = rna_df.loc[keep_rna, tr_ids].T.values.astype(np.float32)
            Xva_rna = rna_df.loc[keep_rna, va_ids].T.values.astype(np.float32)
            Xtr_rna_s, sc_rna = PP.standardize_fit(Xtr_rna)
            Xva_rna_s = PP.standardize_apply(Xva_rna, sc_rna)

            n_rna = _cap_pca_n(RNA_PCA_N, Xtr_rna_s)
            if USE_DIMRED and n_rna > 0:
                print(f"[F{fold}] RNA PCA: preset={RNA_PCA_N} -> use={n_rna}")
                Xtr_rna_s, pca_rna = PP.pca_fit(Xtr_rna_s, n_components=n_rna)
                Xva_rna_s = PP.pca_apply(Xva_rna_s, pca_rna)
            else:
                pca_rna = None

            Xtr_rna_w = PP.apply_block_weight(Xtr_rna_s, BW_RNA)
            Xva_rna_w = PP.apply_block_weight(Xva_rna_s, BW_RNA)
            blocks_tr.append(Xtr_rna_w); blocks_va.append(Xva_rna_w)
        else:
            sc_rna = pca_rna = None

        # --- 融合 ---
        Xtr = PP.concat_blocks(*blocks_tr)
        Xva = PP.concat_blocks(*blocks_va)
        ytr = y_enc[tr_idx]; yva = y_enc[va_idx]
        print(f"[F{fold}] 融合后维度={Xtr.shape[1]} | 训练/验证={Xtr.shape[0]}/{Xva.shape[0]}")

        # 融合后 TopK（可选）
        sel = None
        if TOPK_TOTAL and TOPK_TOTAL > 0:
            Xtr, Xva, sel = PP.select_kbest_global(Xtr, ytr, Xva, TOPK_TOTAL)
            print(f"[F{fold}] 全局TopK选择后维度={Xtr.shape[1]} (K={TOPK_TOTAL})")

        # 训练与评估
        fold_dir = os.path.join(config.MODEL_DIR, f"fold{fold}")
        ensure_dir(fold_dir)
        y_pred = train_one_fold(Xtr, ytr, Xva, yva, labels, fold_dir, device)
        print_and_save_report(yva, y_pred, labels, config.OUTPUT_DIR, f"fold{fold}")
        oof_pred[va_idx] = y_pred

        # 保存预处理器
        if USE_CNV and cnv_df is not None:
            if CNV_ENCODING == "numeric": joblib.dump(sc_cnv, os.path.join(fold_dir, "scaler_cnv.joblib"))
            if 'pca_cnv' in locals() and pca_cnv is not None: joblib.dump(pca_cnv, os.path.join(fold_dir, "pca_cnv.joblib"))
        if USE_METH and meth_df is not None:
            joblib.dump(sc_meth, os.path.join(fold_dir, "scaler_meth.joblib"))
            if 'pca_meth' in locals() and pca_meth is not None: joblib.dump(pca_meth, os.path.join(fold_dir, "pca_meth.joblib"))
        if USE_RNA and rna_df is not None:
            joblib.dump(sc_rna, os.path.join(fold_dir, "scaler_rna.joblib"))
            if 'pca_rna' in locals() and pca_rna is not None: joblib.dump(pca_rna, os.path.join(fold_dir, "pca_rna.joblib"))
        if sel is not None:
            joblib.dump(sel, os.path.join(fold_dir, "selector_topk.joblib"))

    # OOF 总体
    print_and_save_report(y_enc, oof_pred, labels, config.OUTPUT_DIR, "OOF")

if __name__ == "__main__":
    train_eval_cv()
