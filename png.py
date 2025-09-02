import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from train import preprocess_pipeline_with_subtype

res = preprocess_pipeline_with_subtype(
    expr_path=r"D:/毕设数据/HiSeqV2.xlsx",
    clinical_path=r"D:/毕设数据/TCGA.GBM.sampleMap_GBM_clinicalMatrix.xlsx",
    topk_by_variance=100,   # 只挑100个基因，图会更清晰
    test_size=0.2, val_size=0.1, random_state=42
)

# 合并所有样本（用 pd.concat 替换 .append）
X = pd.concat([res["X_train"], res["X_val"], res["X_test"]], axis=0)
y = pd.concat([res["y_train"], res["y_val"], res["y_test"]], axis=0)

# 转成 行=基因，列=样本
matrix = X.T

# 样本按 subtype 排序
sorted_samples = y.sort_values().index
matrix = matrix[sorted_samples]

# subtype 映射到颜色
lut = {"Classical":"#1f77b4", "Mesenchymal":"#ff7f0e",
       "Proneural":"#2ca02c", "Neural":"#d62728"}
col_colors = y[sorted_samples].map(lut)

# 聚类热图
sns.clustermap(matrix,
               cmap="vlag", center=0,
               col_colors=col_colors,
               xticklabels=False, yticklabels=True,
               figsize=(15, 10))
plt.savefig("heatmap.png", dpi=300, bbox_inches="tight")
plt.show()
