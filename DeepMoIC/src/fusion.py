# src/fusion.py
# -*- coding: utf-8 -*-
import numpy as np
from typing import List

def modality_dropout(Z_list: List[np.ndarray], p: float = 0.2, train_mode: bool = True):
    """
    训练期按概率随机把若干模态整段置零；验证/测试不置零。
    """
    if not train_mode or p <= 0:
        return Z_list
    out = []
    for Z in Z_list:
        if Z is None:
            out.append(None)
        else:
            if np.random.rand() < p:
                out.append(np.zeros_like(Z))
            else:
                out.append(Z)
    return out

def fuse_concat(Z_list: List[np.ndarray]) -> np.ndarray:
    """拼接融合（跳过 None 模态）"""
    feats = [Z for Z in Z_list if Z is not None]
    if not feats:
        raise RuntimeError("[fusion] 全部模态为空，无法融合")
    return np.concatenate(feats, axis=1)
