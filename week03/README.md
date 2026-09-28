# 第三周：多特征线性回归与 Ridge 正则化

从单特征过渡到多特征线性回归，比较普通线性回归与 Ridge（L2 正则化），
用验证集选择 `alpha`、用测试集做最终评价。

## 数据来源

- scikit-learn 内置 `load_diabetes`（原始特征，未标准化），10 个特征 + 1 个目标 `target`
- 官方说明：<https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html>

## 主要文件

| 文件 | 作用 |
|---|---|
| `main.py` | 主脚本：划分 → 标准化 → 5 模型训练 → 指标 → 权重 |
| `requirements.txt` | 依赖库及版本 |
| `results/metrics.csv` | 各模型在训练/验证/测试集上的 MAE、RMSE |
| `results/weights.csv` | 多特征 LR 与两组 Ridge 的权重表 |
| `results/rmse_comparison.png` | 各模型训练/验证 RMSE 分组柱状图 |
| `results/weights.png` | 三组模型的权重对比图 |

## 环境与运行

- Python 版本：**3.14.5**
- 安装依赖：

```bash
pip install -r requirements.txt
```

- 运行（在 `week03/` 目录下）：

```bash
python main.py
```

## 关键设置

- 数据划分 60% / 20% / 20%（265 / 88 / 89），`random_state=42`，与第二周一致。
- 按列标准化，只用训练集各列的均值/标准差，目标值保持不变。
- 比较的 5 个模型：均值基线、单特征 LR（bmi）、多特征 LR（10 特征）、
  `Ridge α=1`、`Ridge α=100`。
- 依据**验证 RMSE** 选定 `alpha`，最终选定 **alpha = 1**。

## 结果位置与主要结论

结果位于 `week03/results/`。

- 多特征明显优于单特征：验证 RMSE 从 60.83（单特征）降到 **49.15**（多特征）。
- `Ridge α=1` 与多特征 LR 几乎一致（验证 RMSE 49.20）；`α=100` 验证 RMSE 升高到 50.81，
  说明更强的正则化在这份数据上反而造成欠拟合，正则化不保证效果一定更好。
- 权重整体随 `alpha` 增大而缩小，但个别权重因特征间相关性不单调变化。
