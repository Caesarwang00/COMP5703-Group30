import numpy as np
import torch

def normalize_adjacency(W: np.ndarray, add_self_loop: bool = True, to_tensor: bool = True):
    """
    对称归一化 Ĥ = D^{-1/2} (W + I) D^{-1/2}
    to_tensor=True 则返回 torch.FloatTensor，否则返回 numpy.ndarray
    """
    A = W.copy()
    if add_self_loop:
        A = A + np.eye(A.shape[0], dtype=A.dtype)
    D = np.diag(1.0 / np.sqrt(np.clip(A.sum(axis=1), 1e-9, None)))
    A_hat = D @ A @ D
    A_hat = A_hat.astype(np.float32)
    return torch.from_numpy(A_hat) if to_tensor else A_hat
