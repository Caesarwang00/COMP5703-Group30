# -*- coding: utf-8 -*-
from src import config
from src.train_torch import train_eval_cv

if __name__ == "__main__":
    print("[CONFIG]")
    print("  DATA_DIR       =", config.DATA_DIR)
    print("  MODALITIES     =", f"CNV={config.USE_CNV}, METH={config.USE_METH}, RNA={config.USE_RNA}")
    print("  DEVICE         =", getattr(config, "DEVICE", "auto"))
    print("  EPOCHS         =", getattr(config, "EPOCHS", 80))
    print("  BATCH_SIZE     =", getattr(config, "BATCH_SIZE", 64))
    print("  LR / WD        =", getattr(config, "LR", 1e-3), "/", getattr(config, "WEIGHT_DECAY", 5e-4))
    print("  HIDDEN_DIM     =", getattr(config, "HIDDEN_DIM", 512))
    print("  DIMRED         =", getattr(config, "USE_DIMRED", True),
          f"(CNV_PCA_N={getattr(config, 'CNV_PCA_N', 256)}, METH_PCA_N={getattr(config, 'METH_PCA_N', 256)}, RNA_PCA_N={getattr(config, 'RNA_PCA_N', 256)})")
    print("  SAMPLER/LOSS   =", getattr(config, "SAMPLER", "class_balance"), "/", getattr(config, "LOSS", "focal"))
    print("  BARCODE_GROUPS =", getattr(config, "SHORT_BARCODE_GROUPS", 3), "| DEBUG_LOG =", getattr(config, "DEBUG_LOG", False))
    print("\n开始训练 ...")
    train_eval_cv()
