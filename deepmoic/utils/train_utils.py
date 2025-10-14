# -*- coding: utf-8 -*-
import math
import numpy as np
from typing import Optional, Sequence, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.gcn import DeepGCN

@torch.no_grad()
def _to_device(*tensors, device):
    return [t.to(device) if torch.is_tensor(t) else t for t in tensors]

def _balanced_class_weight(y: np.ndarray) -> np.ndarray:
    """
    sklearn 的 class_weight='balanced'：w_c = n_samples / (n_classes * n_c)
    """
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    w = n_samples / (n_classes * counts.astype(np.float32))
    # 索引对齐到 [0..C-1]
    weight = np.zeros(n_classes, dtype=np.float32)
    for c, wc in zip(classes, w):
        weight[int(c)] = wc
    return weight

def train_gcn(
    A_hat: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    num_classes: int,
    hidden: int,
    layers: int,
    alpha: float,
    dropout: float,
    lr: float,
    epochs: int,
    weight_decay: float,
    device: torch.device,
    class_weight: Optional[Sequence[float]] = None,
    label_smoothing: float = 0.0,
    early_stop: bool = False,
    patience: int = 50,
) -> Tuple[nn.Module, np.ndarray]:
    """
    训练 DeepGCN 并返回 (模型, 全体样本预测)。
    - class_weight: 类别权重（None 则不加权）
    - label_smoothing: 交叉熵的 label smoothing 系数
    - early_stop: 是否用验证集早停（val_mask 即为验证集）
    """
    N, F = X.shape
    model = DeepGCN(in_dim=F, hidden=hidden, out_dim=num_classes,
                    layers=layers, alpha=alpha, dropout=dropout).to(device)

    A_hat_d, X_d, y_d = _to_device(A_hat, X, y, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if class_weight is not None:
        w = class_weight if isinstance(class_weight, torch.Tensor) else torch.tensor(class_weight, dtype=torch.float32, device=device)
    else:
        w = None

    # PyTorch>=1.10 支持 label_smoothing
    criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=label_smoothing if label_smoothing > 0 else 0.0)

    train_mask_t = torch.from_numpy(train_mask).to(device)
    val_mask_t   = torch.from_numpy(val_mask).to(device)

    best_state = None
    best_val = math.inf
    no_improve = 0

    pbar = tqdm(range(epochs), desc="GCN训练", leave=False)
    for _ in pbar:
        model.train()
        logits = model(X_d, A_hat_d)
        loss = criterion(logits[train_mask_t], y_d[train_mask_t])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # 监控验证集
        model.eval()
        with torch.no_grad():
            logits_eval = model(X_d, A_hat_d)
            if early_stop:
                val_loss = criterion(logits_eval[val_mask_t], y_d[val_mask_t]).item()
                pbar.set_postfix(loss=f"{float(loss.item()):.4f}", vloss=f"{val_loss:.4f}")
                if val_loss + 1e-6 < best_val:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        break
            else:
                pbar.set_postfix(loss=f"{float(loss.item()):.4f}")

    if early_stop and best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    with torch.no_grad():
        logits = model(X_d, A_hat_d)
        pred = logits.argmax(dim=1).detach().cpu().numpy()

    return model, pred
