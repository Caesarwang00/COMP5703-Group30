# src/datasets.py
# -*- coding: utf-8 -*-
import numpy as np
from torch.utils.data import Dataset

class NpDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        super().__init__()
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, i):
        import torch
        return torch.from_numpy(self.X[i]), torch.from_numpy(np.asarray(self.y[i]))
