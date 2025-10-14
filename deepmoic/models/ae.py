
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

class AE(nn.Module):
    def __init__(self, in_dim: int, hidden: int, latent: int, dropout: float = 0.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

def train_autoencoder(X: np.ndarray, latent: int, hidden: int, lr: float, epochs: int,
                      batch_size: int, weight_decay: float, dropout: float, device: torch.device) -> np.ndarray:
    """
    对单一模态的 X (n_samples × d) 训练 AE，返回潜表征 Z (n_samples × latent)
    """
    n, d = X.shape
    model = AE(d, hidden, latent, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.MSELoss()

    ds = TensorDataset(torch.from_numpy(X))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    pbar = tqdm(range(epochs), desc="AE训练", leave=False)
    for _ in pbar:
        total = 0.0
        for (xb,) in dl:
            xb = xb.to(device)
            x_hat, _ = model(xb)
            loss = crit(x_hat, xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        pbar.set_postfix({"loss": f"{total/n:.4f}"})
    # 导出潜表征
    model.eval()
    with torch.no_grad():
        Z = []
        dl2 = DataLoader(ds, batch_size=256, shuffle=False, drop_last=False)
        for (xb,) in dl2:
            xb = xb.to(device)
            _, z = model(xb)
            Z.append(z.cpu().numpy())
        Z = np.vstack(Z)
    return Z
