# -*- coding: utf-8 -*-
import os

# 基础路径（相对 Dual-modal-CNV+Mutation/）
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "data"))
OUTPUT_DIR = os.path.join(HERE, "outputs")
MODEL_DIR = os.path.join(HERE, "models")

# 文件名（与你 data 目录一致）
FILES = {
    "clinical":  "TCGA.GBM.sampleMap_GBM_clinicalMatrix",
    "cnv_thres": "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes",
    "mutation":  "GBM_mc3_gene_level.txt",
}

# 任务设置
SUBTYPE_MODE = "4class"   # "4class" | "3class_merge_neural" | "3class_drop_neural"

# 预处理超参
CNV_NONZERO_FRAC = 0.05   # 训练折内 非0比例阈值
CNV_ENCODING = "numeric"  # "numeric" | "onehot"  (onehot 维度更大，内存占用更高)
MUT_MIN_FRAC = 0.01       # 训练折内 低频过滤：比例阈值
MUT_MIN_ABS = 3           # 训练折内 至少出现次数
BLOCK_WEIGHT_CNV = 1.0    # 分块权重（可调 0.75/1.25 做消融）
BLOCK_WEIGHT_MUT = 1.0

# 模型与训练
SEED = 42
N_SPLITS = 5
MODEL_TYPE = "logreg_l2"   # "logreg_l2" | "logreg_en"
LOGREG_C_GRID = [0.5, 1.0, 2.0, 5.0]
LOGREG_L1R_GRID = [0.1, 0.3, 0.5]  # 仅 elasticnet 使用
MAX_ITER = 4000
USE_TQDM = True          # 是否显示进度条
SHOW_GRID_SCORES = True  # 是否打印每个超参在验证折上的分数

# 保存/打印
VERBOSE = True

# --- Torch 训练参数 ---
DEVICE = "cuda"      # "cuda" / "cpu" / "auto"
EPOCHS = 60
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 8
HIDDEN_DIM = 0       # 0 = 逻辑回归；如 512 = 一层MLP
DROPOUT = 0.2
LABEL_SMOOTH = 0.0
SCHEDULER = "cos"    # "cos" 或 None

# 可选：融合后做一次全局特征选择（需要在 preprocess.py 里有 select_kbest_global）
TOPK_TOTAL = 8000    # 0 表示不开；先 8k，稳定后再调
