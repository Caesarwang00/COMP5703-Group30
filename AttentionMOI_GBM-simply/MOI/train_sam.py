import os
import logging
from datetime import datetime

import numpy as np
from scipy import sparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

# =========================
# Config（可按需修改）
# =========================
config = {
    'data_dir': './preprocessed',
    'batch_size': 16,
    'hidden_dim': 220,
    'dropout_rate': 0.4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'lr': 2e-4,
    'epochs': 150,
    'patience': 50,                 # 小数据建议别太大
    'scheduler_step': 0,            # 0=不用StepLR
    'scheduler_gamma': 0.8,
    'model_save_path': './models/best_model.pth',
    'optimizer': 'asam',            # 'adam' | 'adamw' | 'sgd' | 'sam' | 'asam'
    'sam_rho': 0.05,                # SAM/ASAM半径，0.03~0.20 可搜索
    'weight_decay': 2e-3,           # 手工L2正则系数（与底层优化器的WD分离）
    'momentum': 0.9,                # 仅SGD用
    'activation': 'gelu',           # 'gelu' | 'relu' | 'tanh'
    'label_smoothing': 0.1,         # 交叉熵标签平滑（0~0.2）
     'class_names': ['Classical', 'Mesenchymal', 'Neural', 'Proneural'],
}

# =========================
# Logging
# =========================
def setup_logging():
    os.makedirs("./logs", exist_ok=True)
    log_path = f"./logs/train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__), log_path

# =========================
# Dataset
# =========================
class MultiOmicsDataset(Dataset):
    def __init__(self, dna: np.ndarray, rna: np.ndarray, labels: np.ndarray):
        assert dna.shape[0] == rna.shape[0] == labels.shape[0]
        self.dna = torch.tensor(dna, dtype=torch.float32)
        self.rna = torch.tensor(rna, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.dna[idx], self.rna[idx], self.labels[idx]

# =========================
# Model
# =========================
class AttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key   = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.scale = float(np.sqrt(hidden_dim))
        self.norm  = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: [B, D]
        if x.dim() == 2:
            x = x.unsqueeze(1)      # [B, 1, D]
        Q = self.query(x)           # [B, L, H]
        K = self.key(x)             # [B, L, H]
        V = self.value(x)           # [B, L, H]
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, L, L]
        attn = F.softmax(attn, dim=-1)
        out  = torch.matmul(attn, V)                # [B, L, H]
        out  = out.sum(dim=1)                       # [B, H]
        return self.norm(out)

class AttentionMOI(nn.Module):
    def __init__(self, dna_dim, rna_dim, num_classes, hidden_dim=128, dropout_rate=0.5, activation='gelu'):
        super().__init__()
        act = nn.GELU() if activation == 'gelu' else (nn.ReLU() if activation == 'relu' else nn.Tanh())

        # DNA branch
        self.dna_attn = AttentionLayer(dna_dim, hidden_dim)
        self.dna_mlp  = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            act,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            act,
            nn.Dropout(dropout_rate),
        )

        # RNA branch
        self.rna_attn = AttentionLayer(rna_dim, hidden_dim)
        self.rna_mlp  = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            act,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            act,
            nn.Dropout(dropout_rate),
        )

        # 简化的融合（汇聚两个模态）
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            act,
            nn.Dropout(min(0.9, dropout_rate + 0.1)),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            act,
            nn.Dropout(min(0.9, dropout_rate + 0.1)),
        )

        self.fuse_norm = nn.LayerNorm(hidden_dim // 4)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 4, num_classes)
        )

    def forward(self, dna_x, rna_x):
        dna_vec = self.dna_mlp(self.dna_attn(dna_x))
        rna_vec = self.rna_mlp(self.rna_attn(rna_x))
        fused   = torch.stack([dna_vec, rna_vec], dim=1).sum(dim=1)  # [B, C]
        fused   = self.fuse_norm(self.fuse(fused))
        logits  = self.classifier(fused)
        return logits

# =========================
# SAM / ASAM
# =========================
class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization optimizer wrapper."""
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho > 0.0, "rho must be positive"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.is_sam = True

    @torch.no_grad()
    def _grad_norm(self):
        device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            adaptive = group["adaptive"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                scale = p.abs() if adaptive else 1.0
                norms.append(((scale * p.grad).norm(p=2)))
        if not norms:
            return torch.tensor(0., device=device)
        return torch.norm(torch.stack(norms), p=2)

    @torch.no_grad()
    def first_step(self):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            adaptive = group["adaptive"]
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = (p.abs() if adaptive else 1.0) * p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()

    def zero_grad(self):
        self.base_optimizer.zero_grad()

# =========================
# Trainer
# =========================
class RegularizedAttentionMOITrainer:
    def __init__(self, model: nn.Module, device: str, logger: logging.Logger):
        self.model = model.to(device)
        self.device = device
        self.logger = logger

    @staticmethod
    def compute_l2_penalty(model: nn.Module):
        l2 = torch.tensor(0., device=next(model.parameters()).device)
        for p in model.parameters():
            if p.requires_grad:
                l2 = l2 + torch.sum(p ** 2)
        return l2

    def _per_class_accuracy(self, labels, preds, num_classes):
        labels = np.asarray(labels)
        preds  = np.asarray(preds)
        acc = {}
        for c in range(num_classes):
            mask = (labels == c)
            acc[c] = float('nan') if mask.sum() == 0 else float((preds[mask] == c).mean())
        return acc

    def _format_pc(self, pc: dict, names=None) -> str:
        parts = []
        for c in sorted(pc):
            name = (names[c] if names and c < len(names) else f"C{c}")
            v = pc[c]
            parts.append(f"{name}:{'NA' if np.isnan(v) else f'{v:.3f}'}")
        return " | ".join(parts)

    def train_epoch(self, dataloader, optimizer, criterion, weight_decay):
        self.model.train()
        total = 0.0
        use_sam = getattr(optimizer, "is_sam", False)

        for dna, rna, labels in dataloader:
            dna, rna, labels = dna.to(self.device), rna.to(self.device), labels.to(self.device)

            if use_sam:
                # step 1
                optimizer.zero_grad()
                out = self.model(dna, rna)
                ce  = criterion(out, labels)
                l2  = weight_decay * self.compute_l2_penalty(self.model)
                loss = ce + l2
                loss.backward()
                optimizer.first_step()

                # step 2
                optimizer.zero_grad()
                out2 = self.model(dna, rna)
                ce2  = criterion(out2, labels)
                l22  = weight_decay * self.compute_l2_penalty(self.model)
                loss2= ce2 + l22
                loss2.backward()
                optimizer.second_step()
                total += float(loss.item())
            else:
                optimizer.zero_grad()
                out = self.model(dna, rna)
                ce  = criterion(out, labels)
                l2  = weight_decay * self.compute_l2_penalty(self.model)
                loss = ce + l2
                loss.backward()
                optimizer.step()
                total += float(loss.item())
        return total / max(1, len(dataloader))

    @torch.no_grad()
    def evaluate(self, dataloader, criterion):
        self.model.eval()
        total_loss = 0.0
        preds, gts = [], []
        for dna, rna, labels in dataloader:
            dna, rna, labels = dna.to(self.device), rna.to(self.device), labels.to(self.device)
            out  = self.model(dna, rna)
            loss = criterion(out, labels)
            total_loss += float(loss.item())
            preds.extend(torch.argmax(out, dim=1).cpu().numpy().tolist())
            gts.extend(labels.cpu().numpy().tolist())
        overall = accuracy_score(gts, preds) if len(gts) else 0.0
        num_classes = int(max(max(gts), max(preds)) + 1) if gts else self.model.classifier[-1].out_features
        per_class = self._per_class_accuracy(gts, preds, num_classes)
        return total_loss / max(1, len(dataloader)), overall, per_class

    def train(self, train_loader, val_loader, cfg: dict):
        # 1) optimizer （底层weight_decay=0，避免与手工L2重复）
        def opt_adam():
            return torch.optim.Adam(self.model.parameters(), lr=cfg['lr'], weight_decay=0.0)
        def opt_adamw():
            return torch.optim.AdamW(self.model.parameters(), lr=cfg['lr'], weight_decay=0.0)
        def opt_sgd():
            return torch.optim.SGD(self.model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=0.0)
        def opt_sam(adaptive=False):
            return SAM(self.model.parameters(), base_optimizer=torch.optim.AdamW,
                       rho=cfg.get('sam_rho', 0.05), adaptive=adaptive, lr=cfg['lr'], weight_decay=0.0)

        choice = (cfg.get('optimizer') or 'adamw').lower()
        if choice == 'adam':
            optimizer = opt_adam()
        elif choice == 'sgd':
            optimizer = opt_sgd()
        elif choice == 'sam':
            optimizer = opt_sam(adaptive=False)
        elif choice == 'asam':
            optimizer = opt_sam(adaptive=True)
        else:
            optimizer = opt_adamw()

        # 2) scheduler
        scheduler = None
        if cfg.get('scheduler_step', 0) > 0:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer.base_optimizer if getattr(optimizer, "is_sam", False) else optimizer,
                step_size=cfg['scheduler_step'], gamma=cfg.get('scheduler_gamma', 0.5)
            )

        # 3) criterion
        crit = nn.CrossEntropyLoss(label_smoothing=float(cfg.get('label_smoothing', 0.0)))

        # 4) training loop
        best_acc = -1.0
        best_state = None
        patience = int(cfg.get('patience', 20))
        no_improve = 0

        self.logger.info("Regularized Attention-MOI Training Started")
        self.logger.info(f"Training started with enhanced regularization")
        self.logger.info(f"Dropout: {cfg['dropout_rate']}, Weight decay: {cfg['weight_decay']}")

        for epoch in range(cfg['epochs']):
            # train
            tr_loss = self.train_epoch(train_loader, optimizer, crit, cfg['weight_decay'])
            # eval train
            tr_loss_e, tr_acc, tr_pc = self.evaluate(train_loader, crit)
            # eval val
            va_loss, va_acc, va_pc = self.evaluate(val_loader, crit)

            # log
            names = cfg.get('class_names')
            log_line = (f"Epoch [{epoch+1}/{cfg['epochs']}] "
                        f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f} ({self._format_pc(tr_pc, names)}) || "
                        f"Val Loss: {va_loss:.4f} | Val Acc: {va_acc:.4f} ({self._format_pc(va_pc, names)}) || "
                        f"LR: {scheduler.get_last_lr()[0] if scheduler else cfg['lr']:.6f}")
            self.logger.info(log_line)

            # save best
            if va_acc > best_acc:
                best_acc = va_acc
                best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                save_dir = os.path.dirname(cfg['model_save_path']) or "."
                os.makedirs(save_dir, exist_ok=True)
                torch.save({'model_state_dict': best_state, 'val_acc': float(best_acc)}, cfg['model_save_path'])
                self.logger.info(f"Best model saved with val_acc: {best_acc:.4f}")
                no_improve = 0
            else:
                no_improve += 1

            # step scheduler
            if scheduler:
                scheduler.step()

            # early stop
            if no_improve >= patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # load best back to model
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return float(best_acc)

# =========================
# IO Helper
# =========================
def load_split(data_dir, split: str):
    X_dna = sparse.load_npz(os.path.join(data_dir, f'dna_{split}.npz')).toarray()
    X_rna = sparse.load_npz(os.path.join(data_dir, f'rna_{split}.npz')).toarray()
    y     = np.load(os.path.join(data_dir, f'y_{split}.npy'))
    return X_dna, X_rna, y

# =========================
# Main
# =========================
def main(cfg: dict):
    logger, log_path = setup_logging()

    # ensure model dir exists
    os.makedirs(os.path.dirname(cfg['model_save_path']) or ".", exist_ok=True)

    # load data_RNA
    X_dna_train, X_rna_train, y_train = load_split(cfg['data_dir'], 'train')
    X_dna_val,   X_rna_val,   y_val   = load_split(cfg['data_dir'], 'val')

    train_ds = MultiOmicsDataset(X_dna_train, X_rna_train, y_train)
    val_ds   = MultiOmicsDataset(X_dna_val, X_rna_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'], shuffle=False, drop_last=False)

    num_classes = int(len(np.unique(y_train)))
    model = AttentionMOI(
        dna_dim=X_dna_train.shape[1],
        rna_dim=X_rna_train.shape[1],
        num_classes=num_classes,
        hidden_dim=cfg['hidden_dim'],
        dropout_rate=cfg['dropout_rate'],
        activation=cfg['activation'],
    )
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}, Classes: {num_classes}")
    logger.info(f"Enhanced regularization: Dropout={cfg['dropout_rate']}, Weight_decay={cfg['weight_decay']}")

    trainer = RegularizedAttentionMOITrainer(model, cfg['device'], logger)
    best_acc = trainer.train(train_loader, val_loader, cfg)

    logger.info(f"Training completed. Best accuracy: {best_acc:.4f}")
    logger.info(f"Log: {log_path}, Model: {cfg['model_save_path']}")

    # ===== Final per-class summary on best weights =====
    # reload best checkpoint
    try:
        ckpt = torch.load(cfg['model_save_path'], map_location=cfg['device'])
        state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        trainer.model.load_state_dict(state)
    except Exception as e:
        logger.warning(f"Reload best checkpoint failed: {e}")

    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg.get('label_smoothing', 0.0)))
    names = cfg.get('class_names')

    tr_loss_f, tr_acc_f, tr_pc_f = trainer.evaluate(train_loader, criterion)
    va_loss_f, va_acc_f, va_pc_f = trainer.evaluate(val_loader, criterion)

    logger.info("FINAL PER-CLASS | "
                f"Train Acc: {tr_acc_f:.4f} ({trainer._format_pc(tr_pc_f, names)}) || "
                f"Val Acc: {va_acc_f:.4f} ({trainer._format_pc(va_pc_f, names)})")

    # optional test set if exists
    test_files = ['dna_test.npz','rna_test.npz','y_test.npy']
    if all(os.path.exists(os.path.join(cfg['data_dir'], f)) for f in test_files):
        X_dna_test, X_rna_test, y_test = load_split(cfg['data_dir'], 'test')
        test_loader = DataLoader(MultiOmicsDataset(X_dna_test, X_rna_test, y_test),
                                 batch_size=cfg['batch_size'], shuffle=False)
        te_loss_f, te_acc_f, te_pc_f = trainer.evaluate(test_loader, criterion)
        logger.info("FINAL PER-CLASS | "
                    f"Test  Acc: {te_acc_f:.4f} ({trainer._format_pc(te_pc_f, names)})")

if __name__ == "__main__":
    main(config)
