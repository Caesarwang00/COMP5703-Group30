# -*- coding: utf-8 -*-
"""
项目配置（Dual-modal-CNV+Mutation）
- 路径解析更健壮：优先环境变量，其次父级 data_RNA 目录，最后本地 data_RNA 目录
- 关键超参统一整理并标注（★）
"""

import os

# ======================
# 路径与目录
# ======================
THIS_DIR      = os.path.dirname(__file__)
PROJECT_ROOT  = os.path.abspath(os.path.join(THIS_DIR, ".."))

# 优先使用环境变量 GBM_DATA_DIR；否则尝试父级 data_RNA（…/COMP5703Group30/data_RNA），再退到本地 data_RNA
_DATA_CANDIDATES = [
    (os.environ.get("GBM_DATA_DIR") or "").strip() or None,
    os.path.abspath(os.path.join(PROJECT_ROOT, "..", "data_RNA")),   # 父级 data_RNA（推荐你的项目结构）
    os.path.abspath(os.path.join(PROJECT_ROOT, "data_RNA")),         # 本地 data_RNA（备用）
]
DATA_DIR = next((p for p in _DATA_CANDIDATES if p and os.path.isdir(p)), _DATA_CANDIDATES[1])

# 模型与输出目录（放在项目根目录）
MODEL_DIR  = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# ======================
# 训练与交叉验证
# ======================
SEED      = 42
N_SPLITS  = 10  # ★ 分层K折数；样本少时 5 折比较稳

EPOCHS        = 100     # ★ 训练轮次；有早停保护，略大一点更稳
BATCH_SIZE    = 64
LR            = 1e-3   # ★ 学习率；配合AdamW效果好，必要时降到3e-4
WEIGHT_DECAY  = 5e-4   # ★ L2正则；略大于1e-4，抑制过拟合
PATIENCE      = 10     # ★ 早停耐心；升一点避免F1抖动过早停

SCHEDULER     = "cos"  # 余弦退火；None=不用
DEVICE        = "auto" # 'auto' / 'cuda' / 'cpu'
USE_TQDM      = True

# ======================
# 模型容量与正则
# ======================
HIDDEN_DIM   = 512   # ★ 0=Logistic；>0=一层MLP。建议 256–1024（你的数据量推荐 512）
DROPOUT      = 0.5   # ★ MLP时的dropout；0.4–0.6 常用
LABEL_SMOOTH = 0.0   # 标签平滑（仅交叉熵时生效）

# ======================
# 多模态预处理与融合
# ======================
CNV_ENCODING      = "numeric"  # "numeric" 或 "onehot"；阈值化CNV通常 numeric 足够
BLOCK_WEIGHT_CNV  = 1.0        # 块权重：CNV
BLOCK_WEIGHT_MUT  = 1.0        # 块权重：Mutation

# 特征筛选阈值（只在训练折上统计，避免泄露）
CNV_NONZERO_FRAC  = 0.05  # ★ CNV 非零占比阈值（0.03–0.10）；太低会引入噪声
MUT_MIN_FRAC      = 0.01  # ★ 突变阳性比例阈值；与下方绝对数取 max
MUT_MIN_ABS       = 3     # ★ 突变阳性绝对样本数；小于3多半是偶然点

# 融合后再做一次全局TopK（ANOVA F-score），0=禁用
TOPK_TOTAL        = 0     # 建议先关；若仍过拟合再开 2000–5000

# ======================
# 降维（强烈推荐打开，显著提升稳健性）
# ======================
USE_DIMRED  = True        # ★ 开关：对每个模态先降维再融合
CNV_PCA_N   = 256         # ★ CNV降维维度：128–512 看拟合与速度折中
MUT_SVD_N   = 128         # ★ MUT降维维度：64–256；稀疏特征建议略小

# ======================
# 类不平衡处理与损失函数
# ======================
SAMPLER     = "class_balance"  # ★ 训练集均衡采样：None / "class_balance"
LOSS        = "focal"          # ★ "ce"=交叉熵 / "focal"=焦点损失（少数类友好）
FOCAL_GAMMA = 2.0              # ★ 焦点损失的γ；2.0–3.0 提升难样本权重
