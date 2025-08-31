# -*- coding: utf-8 -*-
"""
Tri-modal（CNV + Methyl450 + RNAseq）配置
- 路径解析更健壮，优先环境变量 GBM_DATA_DIR
- 关键训练/预处理超参集中管理
- 新增：SHORT_BARCODE_GROUPS、DEBUG_LOG（便于对齐排错）
"""
import os

# ===== 路径 =====
THIS_DIR     = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))

# 数据目录：优先环境变量，其次父级 data（…/COMP5703Group30/data），再本地 data
_DATA_CANDIDATES = [
    (os.environ.get("GBM_DATA_DIR") or "").strip() or None,
    os.path.abspath(os.path.join(PROJECT_ROOT, "..", "data")),
    os.path.abspath(os.path.join(PROJECT_ROOT, "data")),
]
DATA_DIR = next((p for p in _DATA_CANDIDATES if p and os.path.isdir(p)), _DATA_CANDIDATES[1])

MODEL_DIR  = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# ===== 模态开关 =====
USE_CNV   = True
USE_METH  = True
USE_RNA   = True

# ===== 训练与CV =====
SEED          = 42
N_SPLITS      = 5
EPOCHS        = 80
BATCH_SIZE    = 64
LR            = 1e-3
WEIGHT_DECAY  = 5e-4
PATIENCE      = 10
SCHEDULER     = "cos"      # "cos" 或 None
DEVICE        = "auto"
USE_TQDM      = True

# ===== 模型容量与正则 =====
HIDDEN_DIM   = 512         # 0=Logistic；>0=一层MLP
DROPOUT      = 0.5
LABEL_SMOOTH = 0.0         # 仅在交叉熵时有用（当前默认 focal）

# ===== 特征工程参数 =====
CNV_ENCODING      = "numeric"   # "numeric"/"onehot"
BLOCK_WEIGHT_CNV  = 1.0
BLOCK_WEIGHT_METH = 1.0
BLOCK_WEIGHT_RNA  = 1.0

CNV_NONZERO_FRAC  = 0.05        # CNV 非零占比阈值
RNA_MIN_STD       = 0.1         # RNA 低方差过滤阈值

# 甲基化（高维，需更强筛选）
METH_MIN_STD      = 0.02        # 低方差阈值（当 TopK 未启用时使用）
METH_TOPK_VAR     = 20000       # 按方差TopK（>0 时启用 TopK 优先于 min_std）

# 融合后全局TopK（ANOVA F-score），0=禁用
TOPK_TOTAL        = 0

# ===== 降维（强烈推荐开启） =====
USE_DIMRED  = True
CNV_PCA_N   = 256
METH_PCA_N  = 256
RNA_PCA_N   = 256

# ===== 不平衡处理与损失 =====
SAMPLER     = "class_balance"   # None / "class_balance"
LOSS        = "focal"           # "ce" / "focal"
FOCAL_GAMMA = 2.0

# ===== 条形码与调试 =====
# 先尝试 3 段（TCGA-XX-XXXX）；若交集很小，会自动与 4 段（TCGA-XX-XXXX-01）比较并择优
SHORT_BARCODE_GROUPS = 3
# 打开后会打印文件朝向识别、条形码匹配比例、两两交集规模等调试信息
DEBUG_LOG = True
