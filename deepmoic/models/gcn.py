# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, A_hat):
        # x: [N, F], A_hat: [N, N]
        return A_hat @ self.lin(x)

class DeepGCN(nn.Module):
    """
    DeepGCN with Identity Mapping + Initial Residual (alpha).
    关键：用 in_proj 将 x0 投影到 hidden 维，确保与 h 维度一致再融合。
    """
    def __init__(self, in_dim: int, hidden: int, out_dim: int, layers: int, alpha: float, dropout: float):
        super().__init__()
        assert layers >= 1, "layers must be >= 1"
        self.layers = layers
        self.alpha = alpha
        self.dropout = dropout

        # 初始输入投影（若维度相等则为恒等映射）
        self.in_proj = nn.Identity() if in_dim == hidden else nn.Linear(in_dim, hidden, bias=False)

        self.gcs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # 第1层: in_dim -> hidden
        self.gcs.append(GraphConv(in_dim, hidden))
        self.bns.append(nn.BatchNorm1d(hidden))

        # 剩余层: hidden -> hidden
        for _ in range(layers - 1):
            self.gcs.append(GraphConv(hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))

        self.classifier = nn.Linear(hidden, out_dim)

    def forward(self, x, A_hat):
        # 预先计算投影后的 x0（节省多层重复计算）
        x0h = self.in_proj(x)   # [N, hidden]
        h = x

        for i in range(self.layers):
            h_in = h
            h = self.gcs[i](h, A_hat)
            h = self.bns[i](h)
            h = F.relu(h, inplace=True)
            h = F.dropout(h, p=self.dropout, training=self.training)

            # 身份映射（维度匹配才相加）
            if h_in.shape == h.shape:
                h = h + h_in

            # 初始残差融合
            h = (1 - self.alpha) * h + self.alpha * x0h

        logits = self.classifier(h)
        return logits
