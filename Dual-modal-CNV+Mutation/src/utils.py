# -*- coding: utf-8 -*-
import os, json, numpy as np, pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def print_and_save_report(y_true, y_pred, labels, out_dir, tag):
    rep = classification_report(y_true, y_pred, target_names=labels, output_dict=True, digits=4)
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n=== {tag} ===")
    print(f"Accuracy = {acc:.4f} | Macro-F1 = {f1m:.4f}")
    print(pd.DataFrame(rep).T.round(4))
    print("Confusion Matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=labels, columns=labels))

    ensure_dir(out_dir)
    save_json({"accuracy":acc, "macro_f1":f1m, "report":rep, "labels":list(labels), "confusion_matrix": cm.tolist()},
              os.path.join(out_dir, f"{tag}_metrics.json"))
