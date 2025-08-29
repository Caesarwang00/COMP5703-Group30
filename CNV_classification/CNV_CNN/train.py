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

# ---------------- Dataset ----------------
class GeneDataset(Dataset):
    def __init__(self, X, y):
        X = np.array(X)
        self.X = torch.tensor(X, dtype=torch.long if X.dtype==np.int64 else torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ---------------- CNN Model ----------------
class CNNCNV(nn.Module):
    def __init__(self, num_genes, num_classes, discrete=True, embedding_dim=8,
                 conv_channels=[32,64,128], kernel_sizes=[21,11,7], strides=[4,2,1], dropout=0.25):
        super().__init__()
        self.discrete = discrete
        if discrete:
            self.embedding = nn.Embedding(5, embedding_dim, padding_idx=None)
            in_channels = embedding_dim
        else:
            self.embedding = None
            in_channels = 1

        convs = []
        for out_ch, k, s in zip(conv_channels, kernel_sizes, strides):
            convs.append(nn.Conv1d(in_channels, out_ch, kernel_size=k, stride=s))
            convs.append(nn.ReLU())
            convs.append(nn.Dropout(dropout))
            in_channels = out_ch
        self.conv = nn.Sequential(*convs)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        if self.discrete:
            x = self.embedding(x)
            x = x.permute(0,2,1)
        else:
            x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.mean(dim=-1)
        return self.fc(x)

# ---------------- Preprocessing ----------------
def preprocess_data(cnv_file, label_file, sample_column, label_column, discrete=True):
    cnv_thres = pd.read_csv(cnv_file, sep="\t").set_index("Gene Symbol").T
    cnv_thres.index.name = "sampleID"
    tcga = pd.read_csv(label_file, sep="\t").set_index(sample_column)
    merged = tcga.join(cnv_thres, how="inner")

    y = merged[label_column]
    valid_idx = ~y.isna()
    y = y[valid_idx]
    merged = merged.loc[valid_idx]

    label_map = {c:i for i,c in enumerate(y.astype("category").cat.categories)}
    y_idx = y.map(label_map).values

    X = merged.drop(columns=[label_column])
    X = X.select_dtypes(include=[np.number])

    if discrete:
        X = X.fillna(0)
        X = X.clip(0,4).astype(np.int64)
    else:
        X = StandardScaler().fit_transform(np.nan_to_num(X, nan=np.nanmean(X))).astype(np.float32)

    return X.values, y_idx, label_map

# ---------------- Logging layer shapes ----------------
def log_layer_shapes(model, discrete, num_genes, embedding_dim, device, filename="train_log.txt"):
    with open(filename,"w") as f:
        x = torch.zeros((1,num_genes), dtype=torch.long if discrete else torch.float32).to(device)
        if discrete:
            x = model.embedding(x)
            f.write(f"Embedding output shape: {x.shape}\n")
            x = x.permute(0,2,1)
            f.write(f"After permute for Conv1d: {x.shape}\n")
        else:
            x = x.unsqueeze(1)
            f.write(f"Input for Conv1d shape: {x.shape}\n")
        for i,layer in enumerate(model.conv):
            x = layer(x)
            f.write(f"Layer {i} ({layer.__class__.__name__}) output shape: {x.shape}\n")
        f.write(f"FC layer output shape: {model.fc(x.mean(dim=-1)).shape}\n\n")

# ---------------- Training ----------------
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X, y_idx, label_map = preprocess_data(
        os.path.join("data","Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes"),
        os.path.join("data","TCGA.GBM.sampleMap_GBM_clinicalMatrix"),
        args.sample_column, args.label_column, args.discrete_input
    )

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

    model = CNNCNV(num_genes=X.shape[1], num_classes=len(label_map),
                   discrete=args.discrete_input, embedding_dim=args.embedding_dim).to(device)

    log_layer_shapes(model, args.discrete_input, X.shape[1], args.embedding_dim, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_f1, best_state, patience = 0, None, 0

    for epoch in range(args.epochs):
        model.train()
        for xb,yb in train_loader:
            xb,yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        # validation
        model.eval()
        preds, trues = [],[]
        with torch.no_grad():
            for xb,yb in val_loader:
                xb,yb = xb.to(device), yb.to(device)
                logits = model(xb)
                preds.extend(torch.argmax(logits,1).cpu().numpy())
                trues.extend(yb.cpu().numpy())
        f1 = f1_score(trues,preds,average="macro")
        with open("train_log.txt","a") as f:
            f.write(f"Epoch {epoch}: Val Macro-F1={f1:.4f}\n")
        if f1 > best_f1:
            best_f1 = f1
            best_state = model.state_dict()
            patience=0
        else:
            patience+=1
            if patience>=args.early_stop: break

    torch.save(best_state, "best_model.pt")
    with open("label_map.json","w") as f: json.dump(label_map,f)
    with open("train_log.txt","a") as f:
        f.write("Training finished. Best model saved.\n")

    # test
    model.load_state_dict(best_state)
    model.eval()
    preds, trues = [],[]
    with torch.no_grad():
        for xb,yb in test_loader:
            xb,yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds.extend(torch.argmax(logits,1).cpu().numpy())
            trues.extend(yb.cpu().numpy())
    acc = accuracy_score(trues,preds)
    f1_test = f1_score(trues,preds,average="macro")
    with open("train_log.txt","a") as f:
        f.write(f"Test Accuracy: {acc:.4f}, Test Macro-F1: {f1_test:.4f}\n")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_column",default="sampleID")
    parser.add_argument("--label_column",default="GeneExp_Subtype")
    parser.add_argument("--discrete_input",type=lambda x:x.lower()=="true",default=True)
    parser.add_argument("--embedding_dim",type=int,default=8)
    parser.add_argument("--epochs",type=int,default=50)
    parser.add_argument("--batch_size",type=int,default=16)
    parser.add_argument("--lr",type=float,default=5e-4)
    parser.add_argument("--weight_decay",type=float,default=1e-4)
    parser.add_argument("--early_stop",type=int,default=8)
    args=parser.parse_args()
    train(args)
