# run_deepmoic.py
# -*- coding: utf-8 -*-
"""
重构说明
- BASE: 不常改/与项目结构强相关的固定项
- TUNE: 常用调参项（全部集中在这里），每个参数都写了“作用/调大调小的影响/建议范围”
- 最后用 CFG = {**BASE, **TUNE} 合并，TUNE 会覆盖同名 BASE 键
"""
import json
from src.trainer import run_cv


# =========================
# 1) 固定配置（基本不动）
# =========================
BASE = {
    # ---- 路径/数据 ----
    "DATA_DIR": "../data",
    "FILES": {
        "clinical": "TCGA.GBM.sampleMap_GBM_clinicalMatrix",
        "cnv": "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes",
        "mut": "GBM_mc3_gene_level.txt",
        "rna": "HiSeqV2",
        "meth": "HumanMethylation450",
    },
    # 使用哪些模态（一般不改）
    "MODALITIES": ["cnv", "mut", "rna"],

    # ---- 设备/折数/日志 ----
    "DEVICE": "cuda",
    "N_SPLITS": 3,
    "USE_TQDM": True,           # 训练时显示每 epoch 的简短日志（已去掉花哨进度条）
    "FSD_PROGRESS_MODE": "one-line",  # FSD 的进度也仅单行
    "SHOW_BATCH_PROGRESS": False,     # 设 True 会非常啰嗦
    "FSD_PROGRESS": True,

    # ---- 训练轮次/调度（大框架不动，耐心/学习率放 TUNE）----
    "CLS_EPOCHS": 600,
    "LR_SCHED": "plateau",      # ReduceLROnPlateau
    "EARLYSTOP_METRIC": "macro_f1",  # 早停指标，用宏F1
    "EVAL_INTERVAL": 1,         # 每个 epoch 评估；训练太慢可调大（例如 3/5）

    # ---- 数据对齐/缺失指示 ----
    "ADD_MISSING_INDICATOR": True,   # 为缺模态样本加指示变量
    "SPLIT_STRATIFY_BY_K": True,     # 分层按“拥有模态数”做桶
    "K_BUCKETS": [1],                # 桶的边界（k<=1 与 k>1）
}


# =========================
# 2) 可调参区（只改这里）
# =========================
TUNE = {
    # ---- 样本对齐策略 ----
    "ALIGN_MODE": "union",   # 'union' 用更多样本但模态缺失多；'intersection' 用更完整样本但数量少
    # ↑ 改为 'intersection'：常见现象是多模态模型更稳定，单折 macro_f1 上升但总体样本减少

    # ---- FSD + 预筛（特征选择）----
    "FSD_ENABLE": True,          # 是否启用 FSD（子样本一致性+主成分）
    "FSD_MODALITIES": ["cnv", "rna"],  # 在信息量大的模态上做 FSD
    "FSD_K": 0.05,               # FSD 子样本比例阈值（更小=更严格、特征更“稳”但可能漏掉稀疏信号）
    "FSD_J": 0.80,               # 一致性阈值（调低到 0.6~0.7 可放宽，少数类/稀疏特征更容易留）
    "FSD_M": 25,                 # 先验降到 M 维做稳定性评估（调大保留信息更多，但慢）
    "FSD_SUBSAMPLE_FRAC": 0.6,   # 子采样比例（越大越稳但耗时）
    "FSD_MIN_CLASS": 3,          # 至少多少类有贡献才保留
    "FSD_FEAT_CAP": 4000,        # 进入 FSD 的候选特征上限（调大→更全面更慢）

    # 预筛（按方差/频次）——调大 topk 能留更多特征（信息↑/噪声↑/时间↑）
    "CNV_PRE_MIN_NONZERO_FRAC": 0.05,
    "CNV_PRE_TOPK_VAR": 5000,
    "RNA_PRE_TOPK_VAR": 5000,
    "MUT_MIN_FRAC": 0.02,
    "MUT_MIN_ABS": 5,
    "MUT_PRE_TOPK_FREQ": 1000,

    # 二级方差上限（FSD 后再截断）——调大能缓解欠拟合，调小更快更稳但可能掉性能
    "CNV_TOPK_VAR": 1500,
    "MUT_TOPK_VAR": 0,
    "RNA_TOPK_VAR": 1500,
    "METH_TOPK_VAR": 1500,

    # ---- 降维（PCA/自编码器输出维）----
    "USE_PCA": True,             # 目前用 PCA 风格的低维表征
    "AE_LATENT_DIM": {           # 每模态最终输入分类头的维度
        "rna": 64,               # ↑ 调大(rna/cnv)：信息↑、过拟合风险↑、训练稍慢
        "cnv": 48,
        "mut": 32,
        "default": 32
    },

    # ---- 分类头 & 正则 ----
    "FUSE_DIM": 128,             # 融合后隐层维度（调大更强表达/更慢；小数据建议 96~256）
    "CLS_DROPOUT": 0.2,          # Dropout（↑更稳但可能欠拟合；↓更容易过拟合）

    # ---- 优化器 ----
    "CLS_LR": 3e-3,              # 学习率（↑收敛快但震荡；↓稳但慢，需配合耐心）
    "CLS_WEIGHT_DECAY": 5e-4,    # L2 正则（↑更稳更抗过拟合；过大可能抑制表示）
    "LR_FACTOR": 0.5,            # plateau 时降 LR 的倍率
    "LR_PATIENCE": 30,           # plateau 连续多少次 bad 才降 LR（↓更快降学习率）
    "MIN_LR": 1e-5,              # 最小 LR
    "CLS_PATIENCE": 120,         # 早停耐心（↓会早停更快，↑更可能追到最优）

    # ---- 类不均衡处理 ----
    "USE_CB_SAMPLER": True,      # 类平衡采样器（长尾数据建议开）
    "CB_BETA": 0.999,            # β 越小越偏向少数类（可试 0.99）；过小会过拟合少数类
    "USE_CLASS_WEIGHT": True,    # 加类权重（若已用采样器，训练函数里可能忽略）

    # ---- 损失函数选型（两者二选一）----
    "USE_FOCAL": True,           # Focal Loss（长尾/难样本更友好）
    "FOCAL_GAMMA": 2.0,          # γ↑更关注难样本（过大可能不稳定）
    "USE_BALANCED_SOFTMAX": False,  # Balanced Softmax（对类频不敏感，很多长尾任务 > Focal）

    # ---- 数据增广/训练扰动 ----
    "USE_MIXUP": True,           # mixup 对大类有利，少数类边界可能被抹平；长尾建议先关再对照
    "MIXUP_ALPHA": 0.4,          # α↑扰动更强（建议 0.2~0.8）
    "MODALITY_DROPOUT_P": 0.2,   # 训练时随机丢模态（k=1很多时建议降到 0 或 0.1）

    # ---- 批大小（稳定性/速度权衡）----
    "BATCH_SIZE": 128,           # ↓更稳（正则化效果更强）但慢；↑更快但可能震荡
    "VAL_BATCH_SIZE": 256,

    # ---- 集成策略（多模态 + 单模态）----
    "ENSEMBLE_MODE": "multi+single",  # "multi_only" 只用融合；"single_only" 只用各模态
    "ENS_W_MULTI": 0.6,          # 多模态分支的基础权重（k>=2）
    "ENS_W_MULTI_K1": 0.3,       # k=1（仅单模态样本）时多模态分支的权重（越小单模态越重要）
    "ENS_W_MULTI_POWER": 1.0,    # 权重随 k 的幂次（>1 更偏向高 k）
    "ENS_W_PERMOD": {            # 单模态权重缩放（rna 一般更强，可>1）
        "rna": 1.5,
        "cnv": 1.0,
        "mut": 0.8
    },

    # ---- 让“拥有更多模态”的样本在采样时更靠前 ----
    "PRESENT_ALPHA": 0.6,        # ↑更偏向 k 大的样本；过大可能忽视单模态样本
}


# =========================
# 3) 合并配置 & 运行
# =========================
CFG = {**BASE, **TUNE}

if __name__ == "__main__":
    print("[CONFIG]")
    keys = [k for k in CFG.keys()]
    print(json.dumps({k: CFG[k] for k in keys}, indent=2, ensure_ascii=False))
    print("\n开始训练 ...")
    run_cv(CFG)
