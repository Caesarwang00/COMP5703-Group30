import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from train import CNNCNV, GeneDataset, LOG_FILE

def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load label map
    with open("label_map.json","r") as f:
        label_map = json.load(f)
    num_classes = len(label_map)

    # load data
    cnv_thres = pd.read_csv(os.path.join("data","Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes"), sep="\t")
    tcga = pd.read_csv(os.path.join("data","TCGA.GBM.sampleMap_GBM_clinicalMatrix"), sep="\t")
    cnv_thres = cnv_thres.set_index("Gene Symbol").T
    cnv_thres.index.name = "sampleID"
    merged = tcga.set_index("sampleID").join(cnv_thres, how="inner")

    # remove NaN labels
    valid_idx = ~merged["GeneExp_Subtype"].isna()
    merged = merged.loc[valid_idx]

    y = merged["GeneExp_Subtype"].astype("category").map(lambda c: label_map[c]).values
    X = merged[cnv_thres.columns].fillna(0).astype(np.int64).values + 2

    # dataset
    test_ds = GeneDataset(X,y)
    test_loader = DataLoader(test_ds,batch_size=16)

    # model
    model = CNNCNV(num_genes=X.shape[1], num_classes=num_classes, discrete=True, embedding_dim=8).to(device)
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.eval()

    preds, trues = [],[]
    with torch.no_grad():
        for xb,yb in test_loader:
            xb,yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds.extend(torch.argmax(logits,1).cpu().numpy())
            trues.extend(yb.cpu().numpy())

    f1 = f1_score(trues,preds,average="macro")
    acc = accuracy_score(trues,preds)

    # write test result to log
    with open(LOG_FILE,"a") as f:
        f.write(f"\nTest Set: F1={f1:.3f}, Acc={acc:.3f}\n")

    print(f"Test Set: F1={f1:.3f}, Acc={acc:.3f}")

if __name__=="__main__":
    test()
