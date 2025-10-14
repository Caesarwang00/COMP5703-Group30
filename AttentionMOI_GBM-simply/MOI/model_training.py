import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import numpy as np
from scipy import sparse
import os
import joblib
import logging
from datetime import datetime

# Configuration
config = {
    'data_dir': './preprocessed',
    'batch_size': 32,
    'hidden_dim': 128,
    'dropout_rate': 0.2,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'lr': 1e-3,
    'epochs': 100,
    'patience': 50,
    'scheduler_step': 0,
    'scheduler_gamma': 0.5,
    'model_save_path': './models/best_model.pth',
    'optimizer': 'adamw',
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'activation': 'gelu'
}


# Setup logging
def setup_logging():
    os.makedirs("./logs", exist_ok=True)
    log_path = f"./logs/train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )
    return log_path


# Dataset
class MultiOmicsDataset(Dataset):
    def __init__(self, dna_data, rna_data, labels):
        self.dna_data = torch.FloatTensor(dna_data)
        self.rna_data = torch.FloatTensor(rna_data)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.dna_data[idx], self.rna_data[idx], self.labels[idx]


# Model Components
class AttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.scale = np.sqrt(hidden_dim)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        Q, K, V = self.query(x), self.key(x), self.value(x)
        attn_weights = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / self.scale, dim=-1)
        attended = torch.matmul(attn_weights, V)
        return attended.sum(dim=1)


class AttentionMOI(nn.Module):
    def __init__(self, dna_features, rna_features, num_classes, hidden_dim=128, dropout_rate=0.3, activation='relu'):
        super().__init__()
        act = nn.GELU() if activation == 'gelu' else nn.ReLU() if activation == 'relu' else nn.Tanh()

        # DNA pathway
        self.dna_attention = AttentionLayer(dna_features, hidden_dim)
        self.dna_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), act, nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        # RNA pathway
        self.rna_attention = AttentionLayer(rna_features, hidden_dim)
        self.rna_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), act, nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        # Fusion and classification
        self.cross_attention = AttentionLayer(hidden_dim // 2, hidden_dim // 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4), act, nn.Dropout(dropout_rate + 0.2),
            nn.Linear(hidden_dim // 4, num_classes)
        )

    def forward(self, dna_data, rna_data):
        dna_vec = self.dna_mlp(self.dna_attention(dna_data))
        rna_vec = self.rna_mlp(self.rna_attention(rna_data))
        fused = self.cross_attention(torch.stack([dna_vec, rna_vec], dim=1))
        return self.classifier(fused)


# Trainer
class AttentionMOITrainer:
    def __init__(self, model, device, logger):
        self.model = model.to(device)
        self.device = device
        self.logger = logger
        self.best_acc = 0
        self.best_state = None

    def train_epoch(self, dataloader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        for dna, rna, labels in dataloader:
            dna, rna, labels = dna.to(self.device), rna.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            loss = criterion(self.model(dna, rna), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def evaluate(self, dataloader, criterion):
        self.model.eval()
        total_loss, preds, labels_list = 0, [], []
        with torch.no_grad():
            for dna, rna, labels in dataloader:
                dna, rna, labels = dna.to(self.device), rna.to(self.device), labels.to(self.device)
                outputs = self.model(dna, rna)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                preds.extend(torch.max(outputs, 1)[1].cpu().numpy())
                labels_list.extend(labels.cpu().numpy())
        num_classes = self.model.classifier[-1].out_features
        per_class = self._per_class_accuracy(labels_list, preds, num_classes)
        overall = accuracy_score(labels_list, preds)
        return total_loss / len(dataloader), overall, per_class

    def _per_class_accuracy(self, labels, preds, num_classes):
        labels = np.asarray(labels)
        preds = np.asarray(preds)
        acc = {}
        for c in range(num_classes):
            mask = (labels == c)
            if mask.sum() == 0:
                acc[c] = float('nan')
            else:
                acc[c] = float((preds[mask] == c).mean())
        return acc

    def train(self, train_loader, val_loader, cfg):
        criterion = nn.CrossEntropyLoss()

        # Optimizer selection
        optimizers = {
            'adam': torch.optim.Adam(self.model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay']),
            'adamw': torch.optim.AdamW(self.model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay']),
            'sgd': torch.optim.SGD(self.model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'],
                                   momentum=cfg['momentum'])
        }
        optimizer = optimizers.get(cfg['optimizer'], optimizers['adamw'])

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg['scheduler_step'], cfg['scheduler_gamma']) if cfg[
                                                                                                                     'scheduler_step'] > 0 else None

        patience_counter = 0
        self.logger.info(f"Training started with {cfg['epochs']} epochs, lr={cfg['lr']}, optimizer={cfg['optimizer']}")

        for epoch in range(cfg['epochs']):
            train_loss = self.train_epoch(train_loader, optimizer, criterion)

            train_loss_eval, train_acc, train_pc = self.evaluate(train_loader, criterion)
            val_loss, val_acc, val_pc = self.evaluate(val_loader, criterion)

            pc_val_str = " | ".join(
                [f"C{c}:{(val_pc[c] if not np.isnan(val_pc[c]) else float('nan')):.3f}" for c in sorted(val_pc)])
            pc_tr_str = " | ".join(
                [f"C{c}:{(train_pc[c] if not np.isnan(train_pc[c]) else float('nan')):.3f}" for c in sorted(train_pc)])

            # Save best models
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                patience_counter = 0
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                torch.save({'model_state_dict': self.model.state_dict(), 'val_acc': val_acc}, cfg['model_save_path'])
                self.logger.info(f"Best model saved with val_acc: {val_acc:.4f}")
            else:
                patience_counter += 1

            # Early stopping
            if cfg['patience'] > 0 and patience_counter >= cfg['patience']:
                self.logger.info(f"Early stopping at epoch {epoch + 1}")
                break

            if scheduler: scheduler.step()
            current_lr = scheduler.get_last_lr()[0] if scheduler else cfg['lr']

            self.logger.info(
                f"Epoch [{epoch + 1}/{cfg['epochs']}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} ({pc_tr_str}) || "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} ({pc_val_str}) || "
                f"LR: {current_lr:.6f}"
            )

        if self.best_state: self.model.load_state_dict(self.best_state)
        return self.best_acc


# Main function
def main(cfg):
    log_path = setup_logging()
    logger = logging.getLogger()
    logger.info("Attention-MOI Training Started")

    # Load data_RNA
    if not os.path.exists(cfg['data_dir']):
        logger.error(f"Data directory {cfg['data_dir']} not found")
        return

    X_dna_train = sparse.load_npz(os.path.join(cfg['data_dir'], 'dna_train.npz')).toarray()
    X_dna_val = sparse.load_npz(os.path.join(cfg['data_dir'], 'dna_val.npz')).toarray()
    X_rna_train = sparse.load_npz(os.path.join(cfg['data_dir'], 'rna_train.npz')).toarray()
    X_rna_val = sparse.load_npz(os.path.join(cfg['data_dir'], 'rna_val.npz')).toarray()
    y_train = np.load(os.path.join(cfg['data_dir'], 'y_train.npy'))
    y_val = np.load(os.path.join(cfg['data_dir'], 'y_val.npy'))

    # Create data_RNA loaders
    train_loader = DataLoader(MultiOmicsDataset(X_dna_train, X_rna_train, y_train), batch_size=cfg['batch_size'],
                              shuffle=True)
    val_loader = DataLoader(MultiOmicsDataset(X_dna_val, X_rna_val, y_val), batch_size=cfg['batch_size'])

    # Initialize models and trainer
    num_classes = len(np.unique(y_train))
    model = AttentionMOI(X_dna_train.shape[1], X_rna_train.shape[1], num_classes, cfg['hidden_dim'],
                         cfg['dropout_rate'], cfg['activation'])
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}, Classes: {num_classes}")

    trainer = AttentionMOITrainer(model, cfg['device'], logger)
    best_acc = trainer.train(train_loader, val_loader, cfg)

    logger.info(f"Training completed. Best accuracy: {best_acc:.4f}")
    logger.info(f"Log: {log_path}, Model: {cfg['model_save_path']}")


if __name__ == "__main__":
    main(config)