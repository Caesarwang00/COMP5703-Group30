# -*- coding: utf-8 -*-
"""
train_cv.py
- 5 折分层交叉验证（CNV + Mutation 早期融合）
- 兼容导入：相对导入失败时，将父目录加入 sys.path
- 进度展示：tqdm 进度条 + 网格分数/耗时
- 模型：多类逻辑回归（Softmax, saga），L2 或 ElasticNet（由 config.MODEL_TYPE 决定）
"""

import os, sys, time, json, warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.linear_model")
import numpy as np

# ===== 兼容导入 config（支持直接运行本文件）=====
try:
    import config
except ModuleNotFoundError:
    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.dirname(HERE)  # Dual-modal-CNV+Mutation/
    if ROOT not in sys.path:
        sys.path.append(ROOT)
    import config

# ===== 相对导入本包，其它情况走兜底 =====
try:
    from .load_data import align_modalities
    from . import preprocess as PP
    from .utils import ensure_dir, print_and_save_report, save_json
except Exception:
    from load_data import align_modalities
    import preprocess as PP
    from utils import ensure_dir, print_and_save_report, save_json

import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder            # ← 显式导入 LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

# 进度条（可选）
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

# 若 config 无这些开关，给默认值
USE_TQDM         = getattr(config, "USE_TQDM", True)
SHOW_GRID_SCORES = getattr(config, "SHOW_GRID_SCORES", True)

def build_model(C=1.0, l1_ratio=None):
    """构建逻辑回归模型（Softmax, saga）"""
    base = dict(
        solver="saga",
        C=C,
        max_iter=config.MAX_ITER,
        class_weight="balanced",
        n_jobs=-1,
        random_state=config.SEED,
    )
    if config.MODEL_TYPE == "logreg_l2":
        return LogisticRegression(penalty="l2", **base)
    else:
        assert l1_ratio is not None, "ElasticNet 需要 l1_ratio"
        return LogisticRegression(penalty="elasticnet", l1_ratio=l1_ratio, **base)

def _iter_progress(it, desc):
    if USE_TQDM and tqdm is not None:
        return tqdm(it, desc=desc, leave=True)
    return it

def train_eval_cv():
    # 1) 读取并对齐（原发+有亚型 ∩ CNV ∩ MUT）
    cnv_df, mut_df, y_series, sample_order = align_modalities()
    print(f"[INFO] 对齐后样本数: {len(sample_order)} | CNV形状={cnv_df.shape} | MUT形状={mut_df.shape}")

    # 2) 标签编码
    le = LabelEncoder()
    y_enc = le.fit_transform(y_series.values)
    labels = list(le.classes_)
    print("[INFO] 类别顺序:", labels)

    # 3) 分层 5 折
    skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.SEED)
    splits = list(skf.split(np.arange(len(sample_order)), y_enc))
    oof_pred = np.zeros_like(y_enc)

    ensure_dir(config.OUTPUT_DIR); ensure_dir(config.MODEL_DIR)

    for i in _iter_progress(range(len(splits)), "CV folds"):
        tr_idx, va_idx = splits[i]
        fold = i + 1
        print(f"\n========== Fold {fold} / {config.N_SPLITS} ==========")
        tr_ids = [sample_order[j] for j in tr_idx]
        va_ids = [sample_order[j] for j in va_idx]

        # 4) 训练折内特征过滤
        keep_mut = PP.filter_mutation_lowfreq(mut_df, tr_ids,
                                              min_frac=config.MUT_MIN_FRAC,
                                              min_abs=config.MUT_MIN_ABS)
        keep_cnv = PP.filter_cnv_nonzero_frac(cnv_df, tr_ids,
                                              nz_frac=config.CNV_NONZERO_FRAC)
        print(f"[F{fold}] 保留突变基因: {len(keep_mut)} | 保留CNV基因: {len(keep_cnv)}")

        # 5) 编码
        if config.CNV_ENCODING == "onehot":
            Xtr_cnv, _ = PP.encode_cnv_onehot(cnv_df, tr_ids, keep_cnv)
            Xva_cnv, _ = PP.encode_cnv_onehot(cnv_df, va_ids, keep_cnv)
        else:
            Xtr_cnv, _ = PP.encode_cnv_numeric(cnv_df, tr_ids, keep_cnv)
            Xva_cnv, _ = PP.encode_cnv_numeric(cnv_df, va_ids, keep_cnv)
        Xtr_mut = mut_df.loc[keep_mut, tr_ids].T.values.astype(np.float32)
        Xva_mut = mut_df.loc[keep_mut, va_ids].T.values.astype(np.float32)

        # 6) 分块标准化（训练折内拟合）
        Xtr_cnv_s, sc_cnv = PP.standardize_fit(Xtr_cnv)
        Xva_cnv_s = PP.standardize_apply(Xva_cnv, sc_cnv)
        Xtr_mut_s, sc_mut = PP.standardize_fit(Xtr_mut)
        Xva_mut_s = PP.standardize_apply(Xva_mut, sc_mut)

        # 7) 分块权重
        Xtr_cnv_w = PP.apply_block_weight(Xtr_cnv_s, config.BLOCK_WEIGHT_CNV)
        Xtr_mut_w = PP.apply_block_weight(Xtr_mut_s, config.BLOCK_WEIGHT_MUT)
        Xva_cnv_w = PP.apply_block_weight(Xva_cnv_s, config.BLOCK_WEIGHT_CNV)
        Xva_mut_w = PP.apply_block_weight(Xva_mut_s, config.BLOCK_WEIGHT_MUT)

        # 8) 早期融合
        Xtr = PP.concat_blocks(Xtr_cnv_w, Xtr_mut_w)
        Xva = PP.concat_blocks(Xva_cnv_w, Xva_mut_w)
        ytr = y_enc[tr_idx]; yva = y_enc[va_idx]
        print(f"[F{fold}] Xtr维度={Xtr.shape} | Xva维度={Xva.shape}")

        # 9) 超参小网格
        best = None
        def _iter_grid(lst, desc): return _iter_progress(lst, f"F{fold} {desc}")

        if config.MODEL_TYPE == "logreg_l2":
            for C in _iter_grid(config.LOGREG_C_GRID, "C-grid"):
                t0 = time.perf_counter()
                clf = build_model(C=C)
                clf.fit(Xtr, ytr)
                pred = clf.predict(Xva)
                acc = accuracy_score(yva, pred)
                mF1 = f1_score(yva, pred, average="macro")
                dur = time.perf_counter() - t0
                if (best is None) or (mF1 > best[0]):
                    best = (mF1, C, None, clf)
                if SHOW_GRID_SCORES:
                    print(f"[F{fold}] C={C:<4}  acc={acc:.4f}  macroF1={mF1:.4f}  time={dur:.1f}s")
        else:
            for C in _iter_grid(config.LOGREG_C_GRID, "C-grid"):
                for l1r in _iter_grid(config.LOGREG_L1R_GRID, "l1r-grid"):
                    t0 = time.perf_counter()
                    clf = build_model(C=C, l1_ratio=l1r)
                    clf.fit(Xtr, ytr)
                    pred = clf.predict(Xva)
                    acc = accuracy_score(yva, pred)
                    mF1 = f1_score(yva, pred, average="macro")
                    dur = time.perf_counter() - t0
                    if (best is None) or (mF1 > best[0]):
                        best = (mF1, C, l1r, clf)
                    if SHOW_GRID_SCORES:
                        print(f"[F{fold}] C={C:<4} l1r={l1r:.2f}  acc={acc:.4f}  macroF1={mF1:.4f}  time={dur:.1f}s")

        # 10) 记录最佳并预测
        _, bestC, bestL1r, best_clf = best
        pred_va = best_clf.predict(Xva)
        oof_pred[va_idx] = pred_va
        iters = getattr(best_clf, "n_iter_", None)
        iters = int(np.max(iters)) if iters is not None else None
        print(f"[F{fold}] 选中: C={bestC}" + ("" if bestL1r is None else f", l1_ratio={bestL1r}") +
              ("" if iters is None else f" | n_iter={iters}"))

        # 11) 保存每折模型与变换器/元数据
        fold_dir = os.path.join(config.MODEL_DIR, f"fold{fold}")
        ensure_dir(fold_dir)
        joblib.dump(best_clf, os.path.join(fold_dir, "model.joblib"))
        joblib.dump(sc_cnv,   os.path.join(fold_dir, "scaler_cnv.joblib"))
        joblib.dump(sc_mut,   os.path.join(fold_dir, "scaler_mut.joblib"))

        meta = {
            "train_ids": tr_ids, "valid_ids": va_ids,
            "keep_mut": keep_mut, "keep_cnv": keep_cnv,
            "cnv_encoding": config.CNV_ENCODING,
            "block_weight": {"cnv": config.BLOCK_WEIGHT_CNV, "mut": config.BLOCK_WEIGHT_MUT},
            "best_hyperparams": {"C": bestC, **({"l1_ratio":bestL1r} if bestL1r is not None else {})}
        }
        save_json(meta, os.path.join(fold_dir, "meta.json"))

        # 12) 本折报告
        print_and_save_report(yva, pred_va, labels, config.OUTPUT_DIR, f"fold{fold}")

    # 13) OOF 总体报告
    print_and_save_report(y_enc, oof_pred, labels, config.OUTPUT_DIR, "OOF")

if __name__ == "__main__":
    # 允许直接跑本文件；但推荐从 run_baseline.py 运行
    train_eval_cv()
