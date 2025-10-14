
from typing import Dict
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate_cls(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1}
