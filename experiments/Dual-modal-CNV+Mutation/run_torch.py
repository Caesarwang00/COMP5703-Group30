
# -*- coding: utf-8 -*-
from src import config
from src.train_torch import train_eval_cv

if __name__ == "__main__":
    print("[CONFIG]")
    print("  DATA_DIR       =", config.DATA_DIR)
    print("  DEVICE         =", getattr(config, "DEVICE", "auto"))
    print("  EPOCHS         =", getattr(config, "EPOCHS", 60))
    print("  BATCH_SIZE     =", getattr(config, "BATCH_SIZE", 64))
    print("  LR / WD        =", getattr(config, "LR", 1e-3), "/", getattr(config, "WEIGHT_DECAY", 1e-4))
    print("  HIDDEN_DIM     =", getattr(config, "HIDDEN_DIM", 0))
    print("  TOPK_TOTAL     =", getattr(config, "TOPK_TOTAL", 0))
    print("  N_SPLITS       =", getattr(config, "N_SPLITS", 5))
    print("\n开始训练 ...")
    train_eval_cv()
