# -*- coding: utf-8 -*-
"""
运行方式：
  cd Dual-modal-CNV+Mutation
  python run_baseline.py
"""
import os
import config
from src.train_cv import train_eval_cv
from src.utils import ensure_dir

if __name__ == "__main__":
    ensure_dir(config.OUTPUT_DIR)
    ensure_dir(config.MODEL_DIR)
    print("[CONFIG]")
    print("  SUBTYPE_MODE         =", config.SUBTYPE_MODE)
    print("  CNV_ENCODING         =", config.CNV_ENCODING)
    print("  CNV_NONZERO_FRAC     =", config.CNV_NONZERO_FRAC)
    print("  MUT_MIN_FRAC / ABS   =", config.MUT_MIN_FRAC, "/", config.MUT_MIN_ABS)
    print("  BLOCK_WEIGHT (CNV/MUT) =", config.BLOCK_WEIGHT_CNV, "/", config.BLOCK_WEIGHT_MUT)
    print("  MODEL_TYPE           =", config.MODEL_TYPE)
    print("  C grid               =", config.LOGREG_C_GRID)
    if config.MODEL_TYPE == "logreg_en":
        print("  l1_ratio grid        =", config.LOGREG_L1R_GRID)
    print("  N_SPLITS             =", config.N_SPLITS)
    print("\n开始训练 ...")

    train_eval_cv()

    print("\n[完成]")
    print(" - 指标与混淆矩阵：", os.path.abspath(config.OUTPUT_DIR))
    print(" - 每折模型与特征：", os.path.abspath(config.MODEL_DIR))
