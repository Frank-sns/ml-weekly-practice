# 第一周：机器学习入门与 diabetes 数据读取绘图

第一周实践：理解机器学习基本概念，用 scikit-learn 内置的 **diabetes（糖尿病）数据集**完成
「加载数据 → 查看结构与统计 → 绘图」流程，为后续线性回归做准备。

## 数据来源

- scikit-learn 内置 `load_diabetes`（未标准化原始特征）
- 官方说明：<https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html>
- 规模：**442 行 × 11 列**，10 个特征 + 1 个目标 `target`（连续数值，回归任务）

## 文件说明

| 文件 | 作用 |
|---|---|
| `main.py` | 主脚本：读数据、统计、绘图 |
| `requirements.txt` | 依赖库及版本 |
| `results/target_distribution.png` | 目标值分布直方图 |
| `results/scatter_age_target.png` | age 与 target 的散点图 |
| `学习笔记.md` | 知识点理解笔记 |

## 环境与运行

- Python 版本：**3.14.5**
- 安装依赖：

```bash
pip install -r requirements.txt
```

- 运行（在 `week01/` 目录下）：

```bash
python main.py
```

## 结果

运行后在 `results/` 生成两张图，并在终端打印数据规模、缺失值、基本统计。

主要发现：数据无缺失值；`bmi` 与 `target` 的相关系数（0.586）高于 `age`（0.188）。
