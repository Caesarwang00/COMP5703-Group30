import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
import argparse
from collections import Counter

# ---------------- Dataset ----------------
class GeneDataset(Dataset):
    def __init__(self, X, y):
        X = np.array(X)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ---------------- MLP Model ----------------
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=[256,128], dropout=0.25):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# ---------------- Preprocessing ----------------
def preprocess_data(cnv_file, label_file, sample_column, label_column):
    cnv_df = pd.read_csv(cnv_file, sep="\t").set_index("Gene Symbol").T
    cnv_df.index.name = "sampleID"
    label_df = pd.read_csv(label_file, sep="\t").set_index(sample_column)
    merged = label_df.join(cnv_df, how="inner")

    y = merged[label_column]
    valid_idx = ~y.isna()
    y = y[valid_idx]
    merged = merged.loc[valid_idx]

    label_map = {c:i for i,c in enumerate(y.astype("category").cat.categories)}
    y_idx = y.map(label_map).values

    X = merged.drop(columns=[label_column])
    X = X.select_dtypes(include=[np.number])
    X = StandardScaler().fit_transform(np.nan_to_num(X, nan=np.nanmean(X))).astype(np.float32)

    return X, y_idx, label_map

# ---------------- Training ----------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 打开日志文件
    log_file = open("train_log.txt", "w")

    X, y_idx, label_map = preprocess_data(args.cnv_file, args.label_file,
                                         args.sample_column, args.label_column)

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y_idx, test_size=0.3, stratify=y_idx, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42
    )

    train_ds = GeneDataset(X_train, y_train)
    val_ds = GeneDataset(X_val, y_val)
    test_ds = GeneDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = MLP(input_dim=X.shape[1],
                num_classes=len(label_map),
                hidden_dims=args.hidden_dims,
                dropout=args.dropout).to(device)

    # 保存模型结构
    log_file.write("Model architecture:\n")
    log_file.write(str(model) + "\n\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_f1, best_state, patience = 0, None, 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                preds.extend(torch.argmax(logits, 1).cpu().numpy())
                trues.extend(yb.cpu().numpy())
        f1 = f1_score(trues, preds, average="macro")

        # 真实分布和预测分布
        true_distribution = dict(Counter(trues))
        pred_distribution = dict(Counter(preds))

        log_line = (
            f"Epoch {epoch}: Train Loss={epoch_loss/len(train_loader):.4f}, "
            f"Val Macro-F1={f1:.4f}\n"
            f"    Val True Distribution={true_distribution}\n"
            f"    Val Predictions Distribution={pred_distribution}\n"
        )
        print(log_line.strip())
        log_file.write(log_line)

        if f1 > best_f1:
            best_f1 = f1
            best_state = model.state_dict()
            patience = 0
        else:
            patience += 1
            if patience >= args.early_stop:
                log_file.write("Early stopping triggered.\n")
                break

    torch.save(best_state, "best_model.pt")
    with open("label_map.json", "w") as f:
        json.dump(label_map, f)
    print("Training finished. Best model saved.")
    log_file.write("Training finished. Best model saved.\n")

    # Test
    model.load_state_dict(best_state)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds.extend(torch.argmax(logits, 1).cpu().numpy())
            trues.extend(yb.cpu().numpy())
    acc = accuracy_score(trues, preds)
    f1_test = f1_score(trues, preds, average="macro")
    result_line = f"Test Accuracy: {acc:.4f}, Test Macro-F1: {f1_test:.4f}\n"
    print(result_line.strip())
    log_file.write(result_line)

    # 测试集真实和预测分布
    true_distribution = dict(Counter(trues))
    pred_distribution = dict(Counter(preds))
    log_file.write(f"Test True Distribution={true_distribution}\n")
    log_file.write(f"Test Predictions Distribution={pred_distribution}\n")

    log_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnv_file", default="data/Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes")
    parser.add_argument("--label_file", default="data/TCGA.GBM.sampleMap_GBM_clinicalMatrix")
    parser.add_argument("--sample_column", default="sampleID")
    parser.add_argument("--label_column", default="GeneExp_Subtype")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256,128])
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--early_stop", type=int, default=8)
    args = parser.parse_args()

    train(args)
