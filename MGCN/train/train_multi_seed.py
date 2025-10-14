# -*- coding: utf-8 -*-
"""
train_multi_seed.py — 多随机种子训练 GraphSAGE
基于 train_mgcn_rna.py 的最终超参 (hid=48, lr=5e-4, drop=0.6, wd=1e-4)

运行:
    python3 train_multi_seed.py

输出:
  - seed_results.csv  # 每个种子的 Train/Val/Test 准确率
  - summary.txt       # 均值 ± 标准差
"""

import os
import subprocess
import pandas as pd
import numpy as np

# 你想要跑的随机种子
seeds = [21, 42, 63, 84, 100]

# 调用 train_mgcn_rna.py 的命令
# 注意：train_mgcn_rna.py 需要支持通过环境变量传入 SEED
# 例如:  os.environ["SEED"] = "42"
# 如果你没加，我可以帮你改 train_mgcn_rna.py，让它读取环境变量
CMD = "python3 train_mgcn_rna.py"

results = []
for seed in seeds:
    print(f"\n=== 🔄 开始训练 (seed={seed}) ===")
    env = os.environ.copy()
    env["SEED"] = str(seed)   # 设置环境变量
    # 运行训练脚本
    proc = subprocess.run(CMD, shell=True, capture_output=True, text=True, env=env)
    # 打印输出方便调试
    print(proc.stdout)
    if proc.stderr:
        print("⚠️ stderr:", proc.stderr)

    # 读取 metrics_summary.txt
    if os.path.exists("../other/metrics_summary.txt"):
        import json
        with open("../other/metrics_summary.txt", "r", encoding="utf-8") as f:
            metrics = json.load(f)
        results.append({
            "seed": seed,
            "train_acc": metrics["train_acc"],
            "val_acc": metrics["val_acc"],
            "test_acc": metrics["test_acc"]
        })
    else:
        print(f"❌ seed={seed} 没有生成 metrics_summary.txt")

# 保存结果
df = pd.DataFrame(results)
df.to_csv("seed_results.csv", index=False)

# 汇总 Test 准确率
tests = df["test_acc"].values
mean = np.mean(tests)
std  = np.std(tests)

with open("summary.txt", "w", encoding="utf-8") as f:
    f.write(f"Seeds: {seeds}\n")
    f.write(f"Test Accuracy Mean ± Std: {mean:.3f} ± {std:.3f}\n")

print("\n🎉 多随机种子运行完成！")
print(df)
print(f"📊 Test Accuracy 平均: {mean:.3f} ± {std:.3f}")
