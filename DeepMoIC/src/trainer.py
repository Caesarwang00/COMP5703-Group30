# src/trainer.py
# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import pandas as pd
from typing import Dict, List

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .labels import load_labels_from_clinical
from .dataio import load_tsv_as_df, align_samples
from .features import select_scale_reduce
from .fusion import modality_dropout, fuse_concat
from .datasets import NpDataset
from .models import FusionMLP


# ----------------- 小工具 -----------------
def _standardize(Ztr: np.ndarray, Zva: np.ndarray):
    mu = Ztr.mean(axis=0, keepdims=True)
    std = Ztr.std(axis=0, keepdims=True) + 1e-6
    return (Ztr - mu) / std, (Zva - mu) / std

def _init_bias_with_prior(model: FusionMLP, ytr: np.ndarray, device: str, num_classes: int):
    prior = np.bincount(ytr, minlength=num_classes).astype(np.float32)
    prior = prior / (prior.sum() + 1e-8)
    last = None
    if hasattr(model, "fc") and isinstance(model.fc, torch.nn.Sequential) and isinstance(model.fc[-1], torch.nn.Linear):
        last = model.fc[-1]
    if last is not None and last.bias is not None:
        with torch.no_grad():
            last.bias.copy_(torch.log(torch.tensor(prior + 1e-8, device=device)))

def _class_balanced_weights(y: np.ndarray, beta: float) -> np.ndarray:
    from collections import Counter
    cnt = Counter(y.tolist())
    classes = list(range(y.max() + 1))
    w = []
    for c in classes:
        n = max(cnt.get(c, 0), 1)
        eff_num = (1.0 - beta**n) / (1.0 - beta)
        w.append(1.0 / eff_num)
    w = np.asarray(w, dtype=np.float32)
    w = w / w.mean()
    return w

def _present_counts(sample_ids: List[str], permod_present: Dict[str, set]) -> np.ndarray:
    ks = np.zeros(len(sample_ids), dtype=np.int32)
    for i, sid in enumerate(sample_ids):
        k = 0
        for m, s in permod_present.items():
            if sid in s:
                k += 1
        ks[i] = k
    return ks

def _make_stratify_labels(y: np.ndarray, k_all: np.ndarray, buckets=(1,)) -> np.ndarray:
    if k_all is None:
        return y
    bins = np.digitize(k_all, bins=np.array(buckets))
    return y.astype(np.int32) * 10 + bins.astype(np.int32)


# ----------------- 损失函数 -----------------
def _balanced_softmax_ce(logits: torch.Tensor, target: torch.Tensor, cls_count: torch.Tensor) -> torch.Tensor:
    prior = torch.clamp(cls_count.float(), min=1.0).to(logits.device)
    adj_logits = logits - torch.log(prior)[None, :]
    return torch.nn.functional.cross_entropy(adj_logits, target)

def _focal_ce(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0,
              weight: torch.Tensor = None) -> torch.Tensor:
    logpt = torch.nn.functional.log_softmax(logits, dim=1)
    pt = torch.exp(logpt)
    logpt_t = logpt.gather(1, target.view(-1, 1)).squeeze(1)
    pt_t = pt.gather(1, target.view(-1, 1)).squeeze(1)
    loss = - ((1 - pt_t) ** gamma) * logpt_t
    if weight is not None:
        w = weight[target]
        loss = loss * w
    return loss.mean()

def _ce(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor = None, label_smooth: float = 0.0) -> torch.Tensor:
    if label_smooth > 0:
        return torch.nn.functional.cross_entropy(logits, target, weight=weight, label_smoothing=label_smooth)
    else:
        return torch.nn.functional.cross_entropy(logits, target, weight=weight)

def _compute_loss(cfg: dict, logits: torch.Tensor, target: torch.Tensor,
                  class_weight: torch.Tensor, cls_count_tensor: torch.Tensor):
    if bool(cfg.get("USE_BALANCED_SOFTMAX", False)):
        return _balanced_softmax_ce(logits, target, cls_count_tensor)
    elif bool(cfg.get("USE_FOCAL", True)):
        gamma = float(cfg.get("FOCAL_GAMMA", 2.0))
        return _focal_ce(logits, target, gamma=gamma, weight=class_weight)
    else:
        label_smooth = float(cfg.get("LABEL_SMOOTH", 0.0))
        return _ce(logits, target, weight=class_weight, label_smooth=label_smooth)


# ----------------- 学习率调度器 -----------------
def _build_scheduler(cfg, opt, tr_loader, epochs):
    import math as _math
    sched_name = str(cfg.get("LR_SCHED", "plateau")).lower()
    min_lr = float(cfg.get("MIN_LR", 1e-5))
    mode = "max" if str(cfg.get("EARLYSTOP_METRIC", "macro_f1")).lower() == "macro_f1" else "min"

    if sched_name == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode=mode,
            factor=float(cfg.get("LR_FACTOR", 0.5)),
            patience=int(cfg.get("LR_PATIENCE", 30)),
            min_lr=min_lr
        )
        return sched_name, sched, False, mode

    if sched_name == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=min_lr)
        return sched_name, sched, False, None

    if sched_name == "cosine_wr":
        T0 = int(cfg.get("SGDR_T0", max(10, epochs // 3)))
        T_mult = int(cfg.get("SGDR_T_MULT", 2))
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=T0, T_mult=T_mult, eta_min=min_lr)
        return sched_name, sched, False, None

    if sched_name == "warmup_cosine":
        warmup_epochs = int(cfg.get("WARMUP_EPOCHS", 5))
        base_lr = opt.param_groups[0]['lr']
        eta_min_ratio = min_lr / max(base_lr, 1e-12)
        def lr_lambda(ep):
            if ep < warmup_epochs:
                return (ep + 1) / max(1, warmup_epochs)
            prog = (ep - warmup_epochs) / max(1, epochs - warmup_epochs)
            cosine = 0.5 * (1.0 + _math.cos(_math.pi * prog))
            return eta_min_ratio + (1.0 - eta_min_ratio) * cosine
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return sched_name, sched, False, None

    if sched_name == "onecycle":
        base_lr = opt.param_groups[0]['lr']
        max_lr = float(cfg.get("ONECYCLE_MAX_LR", base_lr * 10.0))
        pct_start = float(cfg.get("ONECYCLE_PCT_START", 0.3))
        div_factor = float(cfg.get("ONECYCLE_DIV_FACTOR", 25.0))
        final_div = float(cfg.get("ONECYCLE_FINAL_DIV", 1e4))
        steps_per_epoch = max(1, len(tr_loader))
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=max_lr, epochs=epochs, steps_per_epoch=steps_per_epoch,
            pct_start=pct_start, div_factor=div_factor, final_div_factor=final_div
        )
        return sched_name, sched, True, None

    if sched_name == "cyclic":
        base_lr = float(cfg.get("CYCLIC_BASE_LR", opt.param_groups[0]['lr'] / 25.0))
        max_lr = float(cfg.get("CYCLIC_MAX_LR", opt.param_groups[0]['lr']))
        step_up_epochs = int(cfg.get("CYCLIC_STEP_UP_EPOCHS", 2))
        step_size_up = max(1, len(tr_loader)) * step_up_epochs
        mode_c = str(cfg.get("CYCLIC_MODE", "triangular2"))
        sched = torch.optim.lr_scheduler.CyclicLR(
            opt, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size_up,
            mode=mode_c, cycle_momentum=False
        )
        return sched_name, sched, True, None

    return "none", None, False, None


# ----------------- 训练（无进度条） -----------------
def train_classifier(Ztr, ytr, Zva, yva, num_classes, cfg, device: str,
                     sample_weights: np.ndarray = None, desc: str = "Train"):
    in_dim = Ztr.shape[1]
    hidden = int(cfg.get("FUSE_DIM", 128))
    drop = float(cfg.get("CLS_DROPOUT", 0.2))
    lr = float(cfg.get("CLS_LR", 3e-3))
    wd = float(cfg.get("CLS_WEIGHT_DECAY", 5e-4))
    epochs = int(cfg.get("CLS_EPOCHS", 600))
    patience = int(cfg.get("CLS_PATIENCE", 120))
    eval_every = int(cfg.get("EVAL_INTERVAL", 1))

    use_mixup = bool(cfg.get("USE_MIXUP", True))
    mixup_alpha = float(cfg.get("MIXUP_ALPHA", 0.4))

    use_cw = bool(cfg.get("USE_CLASS_WEIGHT", True)) and (sample_weights is None)
    class_weight = None
    if use_cw:
        from collections import Counter
        cnt = Counter(ytr.tolist())
        w = np.zeros(num_classes, dtype=np.float32)
        for c in range(num_classes):
            w[c] = 1.0 / max(cnt.get(c, 1), 1)
        w = w / w.sum() * num_classes
        class_weight = torch.tensor(w, dtype=torch.float32, device=device)

    cls_count_tensor = torch.tensor(np.bincount(ytr, minlength=num_classes), dtype=torch.float32, device=device)

    model = FusionMLP(in_dim, num_classes, hidden=hidden, drop=drop,
                      label_smooth=float(cfg.get("LABEL_SMOOTH", 0.0))).to(device)
    _init_bias_with_prior(model, ytr, device, num_classes)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    if sample_weights is not None:
        sampler = WeightedRandomSampler(sample_weights.astype(np.float64),
                                        num_samples=len(sample_weights),
                                        replacement=True)
        tr_loader = DataLoader(NpDataset(Ztr, ytr), batch_size=int(cfg.get("BATCH_SIZE", 64)),
                               sampler=sampler, shuffle=False, num_workers=0, pin_memory=(device=="cuda"))
    else:
        tr_loader = DataLoader(NpDataset(Ztr, ytr), batch_size=int(cfg.get("BATCH_SIZE", 64)),
                               shuffle=True, num_workers=0, pin_memory=(device=="cuda"))
    va_loader = DataLoader(NpDataset(Zva, yva), batch_size=int(cfg.get("VAL_BATCH_SIZE", 128)),
                           shuffle=False, num_workers=0, pin_memory=(device=="cuda"))

    sched_name, sched, step_on_batch, plateau_mode = _build_scheduler(cfg, opt, tr_loader, epochs)

    es_metric = str(cfg.get("EARLYSTOP_METRIC", "macro_f1")).lower()
    best = -1e18 if es_metric == "macro_f1" else math.inf
    best_state = None
    bad = 0

    for ep in range(1, epochs+1):
        model.train()
        total_tr = 0.0
        n_tr = 0

        for xb, yb in tr_loader:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)

            if use_mixup:
                lam = np.random.beta(mixup_alpha, mixup_alpha) if mixup_alpha > 0 else 1.0
                perm = torch.randperm(xb.size(0), device=device)
                yb_perm = yb[perm]
                logits = model(xb)
                loss = lam * _compute_loss(cfg, logits, yb, class_weight, cls_count_tensor) + \
                       (1.0 - lam) * _compute_loss(cfg, logits, yb_perm, class_weight, cls_count_tensor)
            else:
                logits = model(xb)
                loss = _compute_loss(cfg, logits, yb, class_weight, cls_count_tensor)

            opt.zero_grad(); loss.backward(); opt.step()
            total_tr += loss.item() * xb.size(0); n_tr += xb.size(0)

            if step_on_batch and (sched is not None):
                sched.step()

        tr_loss = total_tr / max(n_tr, 1)

        do_eval = (ep % eval_every == 0) or (ep == epochs)
        if not do_eval:
            continue

        model.eval()
        va_loss = 0.0
        n = 0
        all_logits = []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                l = _compute_loss(cfg, logits, yb, class_weight, cls_count_tensor).item()
                va_loss += l * xb.size(0); n += xb.size(0)
                all_logits.append(logits.cpu())
        va_loss /= max(n, 1)
        logits_all = torch.cat(all_logits, dim=0)
        y_pred = logits_all.argmax(dim=1).numpy()
        macro_f1 = f1_score(yva, y_pred, average="macro")

        if (sched is not None) and (not step_on_batch):
            if sched_name == "plateau":
                metric = macro_f1 if plateau_mode == "max" else va_loss
                sched.step(metric)
            else:
                sched.step()

        # print(f"[{desc}] ep={ep} tr={tr_loss:.4f} va={va_loss:.4f} macro_f1={macro_f1:.4f} lr={opt.param_groups[0]['lr']:.2e} bad={bad}")

        improved = (macro_f1 > best + 1e-6) if es_metric == "macro_f1" else (va_loss < best - 1e-6)
        cur_metric = macro_f1 if es_metric == "macro_f1" else va_loss
        if improved:
            best = cur_metric
            bad = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"[{desc}] 早停触发 @ epoch {ep}，best_{es_metric}={best:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ----------------- K 折主流程（无进度条） -----------------
def run_cv(CFG: dict):
    rs = int(CFG.get("RANDOM_STATE", 2024))
    n_splits = int(CFG.get("N_SPLITS", 3))
    device = str(CFG.get("DEVICE", "cpu")).lower()
    modalities = list(CFG.get("MODALITIES", ["cnv","mut","rna"]))
    data_dir = CFG["DATA_DIR"]; files = CFG["FILES"]
    align_mode = str(CFG.get("ALIGN_MODE", "intersection")).lower()
    add_indicator = bool(CFG.get("ADD_MISSING_INDICATOR", True))

    use_cb_sampler = bool(CFG.get("USE_CB_SAMPLER", True))
    cb_beta = float(CFG.get("CB_BETA", 0.999))
    present_alpha = float(CFG.get("PRESENT_ALPHA", 0.6))

    ens_mode = str(CFG.get("ENSEMBLE_MODE", "multi+single")).lower()
    w_multi_base = float(CFG.get("ENS_W_MULTI", 0.6))
    w_permod_cfg = CFG.get("ENS_W_PERMOD", {})
    ens_w_multi_k1 = float(CFG.get("ENS_W_MULTI_K1", 0.3))
    ens_w_pow = float(CFG.get("ENS_W_MULTI_POWER", 1.0))

    stratify_by_k = bool(CFG.get("SPLIT_STRATIFY_BY_K", True))
    k_buckets = tuple(CFG.get("K_BUCKETS", [1]))

    print(f"[INFO] 使用设备: {device}")

    # 标签
    y_ser = load_labels_from_clinical(os.path.join(data_dir, files["clinical"]), label_key="GeneExp_Subtype")
    if len(y_ser) == 0:
        raise RuntimeError("[trainer] 从 clinical 未取得任何标签（GeneExp_Subtype）。")

    # 数据
    mats = {}
    for m in modalities:
        mats[m] = load_tsv_as_df(os.path.join(data_dir, files[m]), m)

    mats2, sample_list = align_samples(mats, modalities, y_ser.index, mode=align_mode)
    print(f"[INFO] 模态: {modalities} | 对齐模式: {align_mode} | 样本数: {len(sample_list)}")
    if len(sample_list) == 0:
        for m in modalities:
            print(f"[DEBUG] {m} 示例列名: {list(mats[m].columns)[:5]}")
        print(f"[DEBUG] clinical 标签示例: {y_ser.index[:5].tolist()}")
        raise RuntimeError("[trainer] 样本为 0，请检查列名与 clinical。")

    y_ser2 = y_ser.loc[sample_list]
    le = LabelEncoder(); y_all = le.fit_transform(y_ser2.values)
    classes = list(le.classes_); num_classes = len(classes)
    print(f"[INFO] 类别顺序: {classes}")

    # 统计全体样本的 k
    global_present = {}
    for m in modalities:
        global_present[m] = set([sid for sid in sample_list if sid in mats2[m].columns])
    k_all = _present_counts(sample_list, global_present)

    # 分层切分标签（类别 × k 桶）
    y_strat = _make_stratify_labels(y_all, k_all, buckets=k_buckets) if stratify_by_k else y_all

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rs)

    oof_pred = np.zeros((len(sample_list), num_classes), dtype=np.float32)
    oof_true = y_all.copy()
    idx_map = {s:i for i,s in enumerate(sample_list)}

    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(sample_list, y_strat), start=1):
        print(f"\n========== [DeepMoIC] Fold {fold_id} / {n_splits} ==========")
        tr_s = [sample_list[i] for i in tr_idx]
        va_s = [sample_list[i] for i in va_idx]
        tr_n = len(tr_s); va_n = len(va_s)

        # 特征 → 标准化 → PCA
        Ztr_list, Zva_list = [], []
        tr_present_dict, va_present_dict = {}, {}

        ytr_map = {sid: int(oof_true[idx_map[sid]]) for sid in tr_s}
        yva_map = {sid: int(oof_true[idx_map[sid]]) for sid in va_s}

        for m in modalities:
            df = mats2[m]
            Ztr, Zva, tr_present, va_present = select_scale_reduce(
                df=df, modality=m, tr_ids=tr_s, va_ids=va_s,
                ytr_map=ytr_map, yva_map=yva_map, cfg=CFG, device=device
            )
            d_after = Ztr.shape[1]
            print(f"[F{fold_id}] 模态={m} | 训练可用={len(tr_present)}/{tr_n} | 验证可用={len(va_present)}/{va_n} | d={d_after}")
            Ztr_list.append(Ztr); Zva_list.append(Zva)
            tr_present_dict[m] = set(tr_present); va_present_dict[m] = set(va_present)

        k_tr = _present_counts(tr_s, tr_present_dict)
        k_va = _present_counts(va_s, va_present_dict)
        uniq, cnts = np.unique(k_tr, return_counts=True)
        print(f"[F{fold_id}] 训练集样本拥有模态数分布: " + ", ".join(f"k={int(u)}:{int(c)}" for u, c in zip(uniq, cnts)))

        # 融合 + 标准化（训练期可选模态dropout）
        Ztr_list_for_fuse = modality_dropout([z.copy() for z in Ztr_list],
                                             p=float(CFG.get("MODALITY_DROPOUT_P", 0.2)),
                                             train_mode=True)
        Zva_list_for_fuse = [z.copy() for z in Zva_list]
        for i, Z in enumerate(Ztr_list_for_fuse):
            assert Z.shape[0] == tr_n, f"[Fold {fold_id}] 训练集行数不符: 模态={modalities[i]} got {Z.shape[0]} expect {tr_n}"
        for i, Z in enumerate(Zva_list_for_fuse):
            assert Z.shape[0] == va_n, f"[Fold {fold_id}] 验证集行数不符: 模态={modalities[i]} got {Z.shape[0]} expect {va_n}"

        Ztr_fuse = fuse_concat(Ztr_list_for_fuse)
        Zva_fuse = fuse_concat(Zva_list_for_fuse)
        print(f"[F{fold_id}] 融合后维度={Ztr_fuse.shape[1]} | 训练/验证={Ztr_fuse.shape[0]}/{Zva_fuse.shape[0]}")
        Ztr_fuse, Zva_fuse = _standardize(Ztr_fuse, Zva_fuse)

        # 采样器权重
        if use_cb_sampler:
            cb_w_cls = _class_balanced_weights(y_all[tr_idx], beta=float(cb_beta))
            w_cls = cb_w_cls[y_all[tr_idx]]
            w_k = 1.0 + float(present_alpha) * np.maximum(k_tr - 1, 0)
            w_samples = (w_cls * w_k).astype(np.float32)
            w_samples = np.clip(w_samples, np.percentile(w_samples, 1), np.percentile(w_samples, 99))
        else:
            w_samples = None

        ytr = y_all[tr_idx]; yva = y_all[va_idx]

        # 多模态头
        model_fuse = train_classifier(Ztr_fuse, ytr, Zva_fuse, yva, num_classes, CFG, device=device,
                                      sample_weights=w_samples, desc=f"F{fold_id}-FUSE")

        # 各单模态头
        permod_probs = {}
        for i, m in enumerate(modalities):
            Zi_tr_full, Zi_va_full = Ztr_list[i], Zva_list[i]
            tr_mask = np.array([sid in tr_present_dict[m] for sid in tr_s], dtype=bool)
            va_mask = np.array([sid in va_present_dict[m] for sid in va_s], dtype=bool)
            Ztr_i, ytr_i = Zi_tr_full[tr_mask], ytr[tr_mask]
            Zva_i, yva_i = Zi_va_full[va_mask], yva[va_mask]
            Ztr_i, Zva_i = _standardize(Ztr_i, Zva_i)

            if use_cb_sampler:
                cb_w_cls_i = _class_balanced_weights(ytr_i, beta=float(cb_beta))
                w_samples_i = cb_w_cls_i[ytr_i].astype(np.float32)
                w_samples_i = np.clip(w_samples_i, np.percentile(w_samples_i, 1), np.percentile(w_samples_i, 99))
            else:
                w_samples_i = None

            model_i = train_classifier(Ztr_i, ytr_i, Zva_i, yva_i, num_classes, CFG, device=device,
                                       sample_weights=w_samples_i, desc=f"F{fold_id}-{m.upper()}")

            with torch.no_grad():
                ds_i = DataLoader(NpDataset(Zva_i, yva_i), batch_size=int(CFG.get("VAL_BATCH_SIZE", 128)),
                                  shuffle=False, num_workers=0, pin_memory=(device=="cuda"))
                pp = []
                for xb, _ in ds_i:
                    xb = xb.to(device, non_blocking=True)
                    logits = model_i(xb)
                    p = torch.softmax(logits, dim=1).cpu().numpy()
                    pp.append(p)
                pp = np.concatenate(pp, axis=0) if len(pp) > 0 else np.zeros((0, num_classes), dtype=np.float32)
            per_prob_full = np.zeros((len(va_s), num_classes), dtype=np.float32)
            per_prob_full[va_mask] = pp
            permod_probs[m] = per_prob_full

        # 多模态概率
        with torch.no_grad():
            ds = DataLoader(NpDataset(Zva_fuse, yva), batch_size=int(CFG.get("VAL_BATCH_SIZE", 128)),
                            shuffle=False, num_workers=0, pin_memory=(device=="cuda"))
            probs_fuse = []
            for xb, _ in ds:
                xb = xb.to(device, non_blocking=True)
                logits = model_fuse(xb)
                p = torch.softmax(logits, dim=1).cpu().numpy()
                probs_fuse.append(p)
            probs_fuse = np.concatenate(probs_fuse, axis=0)

        # 动态集成
        final_probs = np.zeros_like(probs_fuse)
        M = len(modalities)
        for i in range(len(va_s)):
            sid = va_s[i]
            avail = [m for m in modalities if sid in va_present_dict[m]]
            if ens_mode == "multi_only" or len(avail) == 0:
                final_probs[i] = probs_fuse[i]; continue
            k = int(k_va[i])
            if k <= 1:
                w_multi_i = float(ens_w_multi_k1)
            else:
                w_multi_i = float(w_multi_base) * ((k / M) ** float(ens_w_pow))
                w_multi_i = max(min(w_multi_i, 0.95), 0.05)
            if isinstance(w_permod_cfg, dict) and len(w_permod_cfg) > 0:
                weights = np.array([float(w_permod_cfg.get(m, 1.0)) for m in avail], dtype=np.float32)
            else:
                weights = np.ones(len(avail), dtype=np.float32)
            weights = weights / weights.sum()
            per_sum = np.zeros(probs_fuse.shape[1], dtype=np.float32)
            for j, m in enumerate(avail):
                per_sum += weights[j] * permod_probs[m][i]
            final_probs[i] = w_multi_i * probs_fuse[i] + (1.0 - w_multi_i) * per_sum

        for i, sid in enumerate(va_s):
            oof_pred[idx_map[sid]] = final_probs[i]

        y_pred_fuse = probs_fuse.argmax(axis=1)
        y_pred_final = final_probs.argmax(axis=1)

        print("\n--- 多模态(仅fuse) 验证报告 ---")
        print(classification_report(yva, y_pred_fuse, digits=4))
        cm = confusion_matrix(yva, y_pred_fuse, labels=list(range(num_classes)))
        print(pd.DataFrame(cm, index=classes, columns=classes).to_string())

        print("\n--- 集成(多模态+单模态, 动态权重) 验证报告 ---")
        print(classification_report(yva, y_pred_final, digits=4))
        cm = confusion_matrix(yva, y_pred_final, labels=list(range(num_classes)))
        print(pd.DataFrame(cm, index=classes, columns=classes).to_string())

    y_pred = oof_pred.argmax(axis=1)
    print("\n=== OOF (集成) ===")
    print(classification_report(oof_true, y_pred, digits=4))
    cm = confusion_matrix(oof_true, y_pred, labels=list(range(num_classes)))
    print(pd.DataFrame(cm, index=classes, columns=classes).to_string())
