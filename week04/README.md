# 第四周：PCA 降维与 K-means 聚类

本周两项实践（都属于无监督学习，只用特征 X、不用答案 y）：

1. 实践一：PCA 降维前后的线性回归对照；
2. 实践二：K-means 聚类与分组解释。

两项实践共用同一份数据划分和标准化结果。

## 数据来源

- scikit-learn 内置 `load_diabetes(as_frame=True, scaled=False)`，10 个特征 + 1 个目标 `target`
- 官方说明：<https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html>

## 主要文件

| 文件 | 作用 |
|---|---|
| `main.py` | 主脚本：划分 → 标准化 → PCA → 4 模型回归对照 → K-means 聚类 |
| `requirements.txt` | 依赖库及版本 |
| `results/metrics.csv` | 各模型训练/验证/测试 MAE、RMSE，及选定配置标记 |
| `results/pca_summary.csv` | 2 维 / 5 维的累计解释方差比例与重构 MSE |
| `results/rmse_comparison.png` | 四组模型训练/验证 RMSE 分组柱状图 |
| `results/pca_variance.png` | 1~10 维累计解释方差曲线，标出 2 维、5 维 |
| `results/cluster_metrics.csv` | K=2、K=3 的轮廓系数与配置 |
| `results/cluster_summary.csv` | 每组样本数、原始 bmi / bp 均值 |
| `results/cluster_comparison.png` | K=2 与 K=3 双面板聚类图（PC1-PC2 展示） |

## 环境与运行

- Python 版本：**3.14.5**
- 安装依赖：

```bash
pip install -r requirements.txt
```

- 运行（在 `week04/` 目录下）：

```bash
python main.py
```

## 关键设置

- 数据划分 60% / 20% / 20%（265 / 88 / 89），`random_state=42`，与第二、三周一致。
- 按列标准化，只用训练集各列均值/标准差（`StandardScaler` 先 `fit` 再 `transform`），目标值保持不变。
- PCA：`n_components=2` 和 `n_components=5` 各一个，`svd_solver="full"`、`whiten=False`，只在标准化训练集上 `fit`。
- 四组回归模型：均值基线、未降维 LR（10 特征）、PCA 2 维 + LR、PCA 5 维 + LR。
- 依据**验证 RMSE** 在两组 PCA 中选定维数，最终选定 **5 维**。
- 重构 MSE 在**标准化特征空间**计算：`mean((X_scaled - pca.inverse_transform(Z))²)`，对全部样本、全部 10 列取平均。
- K-means：`KMeans(n_clusters=2/3, random_state=42, n_init=10)`，在**标准化后的 10 维训练特征**上拟合。
- 聚类图复用 2 维 PCA 投影：横轴 PC1、纵轴 PC2，组中心经同一个 PCA `transform` 投影到二维后展示。

## 结果位置与主要结论

结果位于 `week04/results/`。

### 实践一：PCA 回归对照

- 累计解释方差比例：保留 2 维 = **0.5488**，保留 5 维 = **0.8351**；重构 MSE 2 维约 0.45、5 维约 0.16，维度越少保留信息越少、丢得越多。
- 验证 RMSE：基线 76.16，未降维 LR 49.15，PCA 2 维 61.52，PCA 5 维 48.99 → 选定 **5 维**。
- 保留更多输入方差并不保证预测更好：5 维保留了 83.5% 方差，但测试 RMSE 58.21，仍略差于未降维 LR 的 56.97，说明 PCA 舍弃的那部分可能含有对预测有用的信息。

### 实践二：K-means 聚类

- 轮廓系数：K=2 为 **0.2334**，K=3 为 **0.1862**，K=2 分组更清楚。
- 两组在 bmi / bp 上有明显差别：K=2 时低 bmi/bp 组 130 人、高 bmi/bp 组 135 人；K=3 时在中间再分出第三组（60 人）。
- 图上分组存在重叠，是因为在 10 维空间聚类、再投影到 2 维展示，投影会损失信息，属正常现象。
