# -*- coding: utf-8 -*-
import os
from typing import Dict, Any, List, Tuple, Union

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

from deepmoic.data.preprocess import align_and_standardize
from deepmoic.models.ae import AE, train_autoencoder
from deepmoic.utils.seed import set_seed
from deepmoic.utils.graph import normalize_adjacency
from deepmoic.utils.snf import build_snf_from_embeddings
from deepmoic.utils.train_utils import train_gcn
from deepmoic.utils.io import read_yaml

Cfg = Union[Dict[str, Any], str, os.PathLike]


def _resolve_path(base_dir: str, p: str) -> str:
    """
    更鲁棒的相对路径解析：
    - 先尝试相对于 YAML 目录；
    - 再尝试 YAML 目录的上一层、上两层；
    - 若路径的祖先中存在 'deepmoic' 目录，则以其父目录作为“项目根”再尝试；
    - 返回第一个存在的路径；若都不存在，返回默认（相对 YAML 目录）的拼接结果。
    """
    if p is None:
        return p
    if os.path.isabs(p):
        return p

    candidates = []

    # 相对于 YAML 目录
    candidates.append(os.path.normpath(os.path.join(base_dir, p)))
    # 再试两层父目录
    parent1 = os.path.abspath(os.path.join(base_dir, os.pardir))
    parent2 = os.path.abspath(os.path.join(base_dir, os.pardir, os.pardir))
    candidates.append(os.path.normpath(os.path.join(parent1, p)))
    candidates.append(os.path.normpath(os.path.join(parent2, p)))

    # 若识别到 'deepmoic' 目录，则其父目录为项目根
    cur = base_dir
    repo_root = None
    for _ in range(5):
        if os.path.basename(cur).lower() == "deepmoic":
            repo_root = os.path.abspath(os.path.join(cur, os.pardir))
            break
        new_cur = os.path.abspath(os.path.join(cur, os.pardir))
        if new_cur == cur:
            break
        cur = new_cur
    if repo_root:
        candidates.append(os.path.normpath(os.path.join(repo_root, p)))

    for c in candidates:
        if os.path.exists(c):
            return c

    return candidates[0]


def _load_config(config: Cfg) -> Tuple[Dict[str, Any], str]:
    """
    支持两种调用：
      - main("/path/to/config.yaml")
      - main(config_dict)
    返回：(cfg_dict, yaml_dir)
    """
    if isinstance(config, (str, os.PathLike)):
        cfg = read_yaml(str(config))
        yaml_dir = os.path.dirname(os.path.abspath(str(config)))
    elif isinstance(config, dict):
        cfg = config
        yaml_dir = os.getcwd()
    else:
        raise TypeError("config 必须是 YAML 路径或已解析的 dict")
    return cfg, yaml_dir


def _prepare_data(cfg: Dict[str, Any], yaml_dir: str):
    """调用 align_and_standardize 读数与对齐"""
    data_cfg = cfg.get("DATA", {}) or {}

    # 1) 模态路径
    mod_dict = data_cfg.get("MODALITIES") or {}
    mod_paths = {k: _resolve_path(yaml_dir, v) for k, v in mod_dict.items()}

    # 2) 标签来源
    clinical_cfg = None
    if data_cfg.get("LABELS_FROM_CLINICAL"):
        clinical_cfg = dict(data_cfg["LABELS_FROM_CLINICAL"])
        if "PATH" in clinical_cfg and clinical_cfg["PATH"]:
            clinical_cfg["PATH"] = _resolve_path(yaml_dir, clinical_cfg["PATH"])

    labels_path = None
    if data_cfg.get("LABELS"):
        labels_path = _resolve_path(yaml_dir, data_cfg["LABELS"])

    # 3) 其他选项
    take_intersection = bool(data_cfg.get("TAKE_INTERSECTION", True))
    zscore = bool(data_cfg.get("ZSCORE", True))
    select_top_var = data_cfg.get("SELECT_TOP_VAR", None)

    # 调试输出
    print("[PATH] resolved modalities:")
    for k, v in mod_paths.items():
        print(f"  - {k}: {v}")
    if labels_path:
        print(f"[PATH] labels_path: {labels_path}")
    if clinical_cfg and clinical_cfg.get("PATH"):
        print(f"[PATH] clinical_matrix: {clinical_cfg['PATH']}")

    # 4) 对齐
    X_list, y, sample_ids, le = align_and_standardize(
        modality_paths=mod_paths,
        labels_path=labels_path,
        clinical_cfg=clinical_cfg,
        take_intersection=take_intersection,
        zscore=zscore,
        select_top_var=select_top_var,
    )
    return X_list, y, sample_ids, le


def _train_all_autoencoders(X_list, ae_cfg, device):
    latent = int(ae_cfg.get("LATENT_PER_MODAL", 64))
    hidden = int(ae_cfg.get("HIDDEN", 128))
    lr = float(ae_cfg.get("LR", 1e-3))
    epochs = int(ae_cfg.get("EPOCHS", 300))
    batch_size = int(ae_cfg.get("BATCH_SIZE", 256))
    weight_decay = float(ae_cfg.get("WEIGHT_DECAY", 0.0))
    dropout = float(ae_cfg.get("DROPOUT", 0.1))
    Z_list = []
    for i, Xi in enumerate(X_list):
        Zi = train_autoencoder(
            Xi,
            latent=latent,
            hidden=hidden,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            weight_decay=weight_decay,
            dropout=dropout,
            device=device,
        )
        # === 新增：对每个模态嵌入做规范化，提升 SNF 稳定性 ===
        Zi = Zi / (np.linalg.norm(Zi, axis=1, keepdims=True) + 1e-9)  # 行 L2
        Zi = (Zi - Zi.mean(axis=0, keepdims=True)) / (Zi.std(axis=0, keepdims=True) + 1e-8)  # 列 z-score

        Z_list.append(Zi)
        print(f"[AE] modal {i}: X={Xi.shape} -> Z={Zi.shape}")
    return Z_list


def main(config: Cfg) -> None:
    # ---------- 1) 配置 ----------
    cfg, yaml_dir = _load_config(config)

    # ---------- 2) 数据 ----------
    X_list, y, sample_ids, le = _prepare_data(cfg, yaml_dir)
    n_samples = len(y)
    names = list(le.classes_) if hasattr(le, "classes_") else sorted(set(y))
    n_classes = len(names)
    print(f"[DATA] n_samples={n_samples}, n_classes={n_classes}, labels={names}")

    # ---------- 3) 设备 ----------
    train_cfg = cfg.get("TRAIN", {})
    use_gpu = bool(train_cfg.get("USE_GPU", True))
    val_ratio = float(train_cfg.get("VAL_RATIO", 0.2))
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    # ---------- 4) AE ----------
    ae_cfg = cfg.get("AE", {}) or {}
    Z_list = _train_all_autoencoders(X_list, ae_cfg, device)

    # ---------- 5) SNF -> 归一化邻接 ----------
    snf_cfg = cfg.get("SNF", {}) or {}
    metric = str(snf_cfg.get("METRIC", "cosine"))           # 允许 euclidean/cosine
    K = int(snf_cfg.get("K", max(5, min(20, int(0.3 * n_samples)))))  # 自适应 K
    T = int(snf_cfg.get("T", 25))
    P = build_snf_from_embeddings(Z_list, k=K, t=T, metric=metric)
    A_hat = normalize_adjacency(P, add_self_loop=True, to_tensor=True)

    # ---------- 6) 节点特征 ----------
    X_feat = torch.from_numpy(np.concatenate(Z_list, axis=1)).float()

    # ---------- 7) 划分 ----------
    exp_cfg = cfg.get("EXPERIMENT", {}) or {}
    repeats = int(exp_cfg.get("REPEATS", 20))
    stratified = bool(exp_cfg.get("STRATIFIED", True))
    train_ratio = float(exp_cfg.get("TRAIN_RATIO", 0.75))
    seeds = exp_cfg.get("RANDOM_SEEDS", list(range(1, repeats + 1)))
    if not isinstance(seeds, list) or len(seeds) < repeats:
        seeds = list(range(1, repeats + 1))

    # ---------- 8) GCN 超参 ----------
    gcn_cfg = cfg.get("GCN", {}) or {}
    hidden = int(gcn_cfg.get("HIDDEN", 64))
    n_layers = int(gcn_cfg.get("LAYERS", 8))
    alpha = float(gcn_cfg.get("ALPHA", 0.05))
    dropout = float(gcn_cfg.get("DROPOUT", 0.65))
    lr = float(gcn_cfg.get("LR", 1e-3))
    epochs = int(gcn_cfg.get("EPOCHS", 300))
    weight_decay = float(gcn_cfg.get("WEIGHT_DECAY", 5e-4))
    label_smoothing = float(gcn_cfg.get("LABEL_SMOOTHING", 0.05))
    early_stop = bool(gcn_cfg.get("EARLY_STOP", True))
    patience = int(gcn_cfg.get("EARLY_STOP_PATIENCE", 100))
    use_class_weight = bool(gcn_cfg.get("CLASS_WEIGHTED_CE", True))

    # ---------- 9) 搬设备 ----------
    A_hat = A_hat.to(device)
    X_feat = X_feat.to(device)
    y_t = torch.from_numpy(np.asarray(y)).long().to(device)

    sss_outer = StratifiedShuffleSplit(
        n_splits=repeats, train_size=train_ratio, random_state=0
    )

    from deepmoic.utils.metrics import evaluate_cls

    all_metrics: List[Dict[str, float]] = []

    for rep_idx, (tr_idx, te_idx) in enumerate(sss_outer.split(np.zeros(n_samples), y), start=1):
        set_seed(seeds[rep_idx - 1] if rep_idx - 1 < len(seeds) else rep_idx)

        # train/val 再划分
        if 0.0 < val_ratio < 1.0 and len(tr_idx) > 5:
            inner = StratifiedShuffleSplit(n_splits=1, train_size=(1.0 - val_ratio), random_state=seeds[rep_idx - 1])
            tr_sub, val_sub = next(inner.split(np.zeros(len(tr_idx)), y[tr_idx]))
            tr_mask = np.zeros(n_samples, dtype=bool); tr_mask[tr_idx[tr_sub]] = True
            val_mask = np.zeros(n_samples, dtype=bool); val_mask[tr_idx[val_sub]] = True
        else:
            tr_mask = np.zeros(n_samples, dtype=bool); tr_mask[tr_idx] = True
            val_mask = np.zeros(n_samples, dtype=bool)

        te_mask = np.zeros(n_samples, dtype=bool); te_mask[te_idx] = True

        # 类别权重
        class_weight_t = None
        if use_class_weight:
            cls = np.unique(y)
            weights = compute_class_weight(class_weight="balanced", classes=cls, y=y[tr_mask])
            order = np.argsort(cls)
            class_weight_t = torch.tensor(weights[order], dtype=torch.float32, device=device)

        # 训练 GCN
        model, pred_all = train_gcn(
            A_hat=A_hat,
            X=X_feat,
            y=y_t,
            train_mask=tr_mask,
            val_mask=val_mask,
            num_classes=n_classes,
            hidden=hidden,
            layers=n_layers,
            alpha=alpha,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            label_smoothing=label_smoothing,
            early_stop=early_stop,
            patience=patience,
            class_weight=class_weight_t,
            device=device,
        )

        m = evaluate_cls(y_true=y[te_mask], y_pred=pred_all[te_mask])
        print(f"[REP {rep_idx}] acc={m.get('acc', 0):.4f}, macro_f1={m.get('f1',0):.4f}")
        all_metrics.append(m)

    if all_metrics:
        accs = [m.get("acc", 0.0) for m in all_metrics]
        macro_f1s = [m.get("f1", 0.0) for m in all_metrics]
        print("========== SUMMARY ==========")
        print(f"acc  : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"maF1 : {np.mean(macro_f1s):.4f} ± {np.std(macro_f1s):.4f}")
