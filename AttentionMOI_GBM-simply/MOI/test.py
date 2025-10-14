# test.py —— 对齐版
import os
import numpy as np
from scipy import sparse
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report

# 关键：从训练脚本导入同一份定义与训练超参
from train_sam import AttentionMOI, MultiOmicsDataset, config as train_cfg  # ← 文件名需与实际一致

CONFIG = {
    'data_dir': './preprocessed',
    'model_path': './models/best_model.pth',
    'batch_size': 16,
    # 默认沿用训练脚本里的 class_names，避免类数不一致
    'class_names': train_cfg.get('class_names', ['Classical','Mesenchymal','Neural','Proneural']),
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def load_test_data(data_dir):
    X_dna = sparse.load_npz(os.path.join(data_dir, 'dna_test.npz')).toarray()
    X_rna = sparse.load_npz(os.path.join(data_dir, 'rna_test.npz')).toarray()
    y     = np.load(os.path.join(data_dir, 'y_test.npy'))
    return X_dna, X_rna, y

def build_model_from_cfg_or_ckpt(dna_dim, rna_dim, num_classes, ckpt_obj, device):
    """
    优先使用 ckpt['arch'] 里的结构；否则回落到训练脚本的配置（与 train 对齐）。
    """
    arch = None
    if isinstance(ckpt_obj, dict):
        arch = ckpt_obj.get('arch', None)

    if arch:
        hidden_dim     = arch.get('hidden_dim', train_cfg['hidden_dim'])
        dropout_rate   = arch.get('dropout_rate', train_cfg['dropout_rate'])
        activation     = arch.get('activation', train_cfg.get('activation','gelu'))
        num_classes    = arch.get('num_classes', num_classes)
        # dna_dim/rna_dim 以当前测试特征维度为准；通常应与训练一致
    else:
        hidden_dim   = train_cfg['hidden_dim']
        dropout_rate = train_cfg['dropout_rate']
        activation   = train_cfg.get('activation', 'gelu')

    model = AttentionMOI(
        dna_dim=dna_dim,           # 注意：与训练定义一致的参数名
        rna_dim=rna_dim,           # 注意：与训练定义一致的参数名
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
        activation=activation,
    ).to(device)

    return model

def evaluate(model, test_loader, device, class_names):
    model.eval()
    all_preds, all_gts = [], []
    with torch.no_grad():
        for dna_x, rna_x, y in test_loader:
            dna_x = dna_x.to(device); rna_x = rna_x.to(device)
            logits = model(dna_x, rna_x)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(pred); all_gts.extend(y.numpy())
    all_preds = np.array(all_preds); all_gts = np.array(all_gts)
    acc = accuracy_score(all_gts, all_preds)
    print(f"Overall Accuracy: {acc:.4f}  |  Total Samples: {len(all_gts)}\n")
    # 每类命中率
    for i, name in enumerate(class_names):
        mask = (all_gts == i)
        if mask.sum() > 0:
            cls_acc = (all_preds[mask] == i).mean()
            print(f"Class: {name:<12} | Samples: {mask.sum():<3d} | Acc: {cls_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_gts, all_preds, target_names=class_names, digits=4))

def main():
    print("Testing Attention-MOI (aligned with train)")
    print("=" * 50)

    device = CONFIG['device']
    X_dna, X_rna, y = load_test_data(CONFIG['data_dir'])

    # 类数务必与训练一致（按训练脚本里的 class_names）
    num_classes = len(CONFIG['class_names'])

    # 读 checkpoint
    ckpt = torch.load(CONFIG['model_path'], map_location=device)

    # 构建与训练一致的模型结构
    model = build_model_from_cfg_or_ckpt(
        dna_dim=X_dna.shape[1],
        rna_dim=X_rna.shape[1],
        num_classes=num_classes,
        ckpt_obj=ckpt,
        device=device
    )

    # 加载权重（兼容直接 state_dict 或带 model_state_dict 的存档）
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[Warn] load_state_dict with diffs → missing={len(missing)} unexpected={len(unexpected)}")
        if missing:   print("  missing:", missing[:5], "..." if len(missing)>5 else "")
        if unexpected:print("  unexpected:", unexpected[:5], "..." if len(unexpected)>5 else "")

    # DataLoader 与训练里一致的 Dataset
    test_ds = MultiOmicsDataset(X_dna, X_rna, y)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False, drop_last=False)

    # 评估
    evaluate(model, test_loader, device, CONFIG['class_names'])

if __name__ == "__main__":
    main()
