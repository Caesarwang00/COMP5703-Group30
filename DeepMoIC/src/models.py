# src/models.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionMLP(nn.Module):
    """
    小样本稳健设置：BN + Dropout；最后一层在外部用类先验 bias 初始化。
    """
    def __init__(self, in_dim, num_classes, hidden=128, drop=0.2, label_smooth=0.0):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(hidden, num_classes),
        )
        self.label_smooth = label_smooth

    def forward(self, x):
        return self.fc(x)

    def loss(self, logits, y, class_weight=None):
        if self.label_smooth and self.label_smooth > 0:
            n = logits.size(1)
            with torch.no_grad():
                true_dist = torch.zeros_like(logits)
                true_dist.fill_(self.label_smooth / (n - 1))
                true_dist.scatter_(1, y.unsqueeze(1), 1.0 - self.label_smooth)
            log_prob = F.log_softmax(logits, dim=1)
            loss = -(true_dist * log_prob).sum(dim=1)
            if class_weight is not None:
                loss = loss * class_weight[y]
            return loss.mean()
        return F.cross_entropy(logits, y, weight=class_weight)
