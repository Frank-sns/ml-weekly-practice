# 第二周：单特征线性回归与梯度下降

用 NumPy 手写梯度下降，拟合 diabetes 数据集的**单特征（bmi）线性回归**
`预测值 = w × bmi + b`，并与均值基线、库函数 `LinearRegression` 对照，
理解"训练"与"评价"的区别。

## 任务与数据来源

- 数据：scikit-learn 内置 `load_diabetes`（原始特征，未标准化），只使用 `bmi` 一个特征
- 官方说明：<https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html>
- 完整流程：数据划分 → 训练集标准化 → 均值基线 → NumPy 梯度下降（两个学习率）→
  `LinearRegression` 对照 → MAE/RMSE 指标 → 测试集预测散点图

## 文件说明

| 文件 | 作用 |
|---|---|
| `main.py` | 主脚本，包含全部 8 个步骤 |
| `requirements.txt` | 依赖库及版本 |
| `results/loss_curves.png` | 两个学习率（0.01 / 0.1）的损失曲线 |
| `results/pred_vs_true.png` | 测试集"真实值 vs 预测值"散点图（含 y=x 线） |
| `results/metrics.csv` | 各模型在训练/验证/测试集上的 MAE、RMSE 指标表 |

## 环境与运行

- Python 版本：**3.14.5**
- 安装依赖：

```bash
pip install -r requirements.txt
```

- 运行（在 `week02/` 目录下）：

```bash
python main.py
```

## 结果位置与主要结论

运行后在 `week02/results/` 生成两张图和一个 CSV；终端打印划分比例、标准化统计量、
各模型参数与指标。

主要结论：
- 数据划分 60% / 20% / 20%（265 / 88 / 89）。
- 梯度下降收敛后与 `LinearRegression` 闭式解几乎一致（预测值平均绝对差约 0.0000）。
- 两个学习率最终收敛到同一解（500 轮均充分收敛）。
- 测试集 RMSE：线性模型 **67.04**，优于均值基线的 **76.91**；训练集 RMSE（61.45）
  低于测试集（67.04），属正常的泛化差距。
