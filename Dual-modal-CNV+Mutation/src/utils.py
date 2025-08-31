
# -*- coding: utf-8 -*-
import os, json
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_json(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def print_and_save_report(y_true, y_pred, labels, out_dir, name):
    rep = classification_report(y_true, y_pred, target_names=labels, digits=4, output_dict=True, zero_division=0)
    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    rep["overall"] = {"accuracy": acc, "macro_f1": f1_score(y_true, y_pred, average="macro")}
    ensure_dir(out_dir)
    # 文本
    lines = [f"=== {name} ===", f"accuracy: {rep['overall']['accuracy']:.4f}  macro-F1: {rep['overall']['macro_f1']:.4f}", ""]
    for i, lab in enumerate(labels):
        d = rep.get(lab, {})
        lines.append(f"{lab:12s} prec={d.get('precision',0):.4f} rec={d.get('recall',0):.4f} f1={d.get('f1-score',0):.4f}")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(out_dir, f"{name}_report.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    # JSON
    save_json(rep, os.path.join(out_dir, f"{name}_report.json"))
