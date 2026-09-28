# -*- coding: utf-8 -*-
"""
第三周实践：多特征线性回归与 Ridge 正则化对照

目标：
  1. 从单特征过渡到多特征线性回归，理解多个输入如何共同形成一个预测值。
  2. 比较普通线性回归与 Ridge（L2 正则化），理解 alpha（正则化强度）的作用。
  3. 用验证集选择 alpha，用测试集做最终评价。

数据：scikit-learn 内置 diabetes，10 个特征 + 1 个目标 target。
本周统一调用库函数训练（LinearRegression / Ridge），不再手写梯度下降。

流程：
  1. 固定划分 60% 训练 / 20% 验证 / 20% 测试（随机种子 42，与第二周一致）
  2. 按列标准化（各列只用训练集均值和标准差）
  3. 训练 5 个模型：均值基线、单特征LR(bmi)、多特征LR、Ridge(α=1)、Ridge(α=100)
  4. 比较训练/验证 MAE、RMSE，画 RMSE 分组柱状图
  5. 依据验证 RMSE 选定 Ridge 的 alpha
  6. 对 4 个模型计算测试集指标
  7. 保存权重表并画权重对比图

运行：在 week03/ 目录下执行  python main.py
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # 无界面后端，直接保存图片
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 中文字体配置
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)


# ---------- 步骤 1：加载数据 + 固定划分 ----------
dataset = load_diabetes(as_frame=True, scaled=False)
df = dataset.frame
feature_cols = [c for c in df.columns if c != "target"]  # 10 个特征列名
X = df[feature_cols].to_numpy()   # 特征矩阵，形状 (442, 10)
y = df["target"].to_numpy()       # 目标，形状 (442,)

# 与第二周完全一致的划分：先留出 40%，再均分为验证和测试 → 60/20/20
X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=42)

print(f"数据划分：训练 {len(y_train)} / 验证 {len(y_val)} / 测试 {len(y_test)}")
print(f"特征列（{len(feature_cols)} 个）：{feature_cols}\n")


# ---------- 步骤 2：按列标准化（只用训练集统计量） ----------
# axis=0 表示"对每一列"求统计量，得到长度为 10 的数组（每个特征一个均值/标准差）
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train_s = (X_train - mean) / std
X_val_s = (X_val - mean) / std
X_test_s = (X_test - mean) / std

# 单特征：取标准化后的 bmi 列。[:, [i]] 表示"所有行、只取第 i 列"，保持二维 (n,1)
bmi_idx = feature_cols.index("bmi")
X_train_bmi = X_train_s[:, [bmi_idx]]
X_val_bmi = X_val_s[:, [bmi_idx]]
X_test_bmi = X_test_s[:, [bmi_idx]]

print("标准化完成（每列用各自的训练集均值/标准差，目标值保持不变）\n")


# ---------- 步骤 3：训练 5 个模型 ----------
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train_s, y_train)          # 均值基线：预测恒等于训练集 target 均值

single = LinearRegression()
single.fit(X_train_bmi, y_train)          # 单特征线性回归（只用 bmi）

multi = LinearRegression()
multi.fit(X_train_s, y_train)             # 多特征线性回归（10 个特征）

ridge1 = Ridge(alpha=1)
ridge1.fit(X_train_s, y_train)            # Ridge，较弱正则化

ridge100 = Ridge(alpha=100)
ridge100.fit(X_train_s, y_train)          # Ridge，较强正则化

# 每个模型记录四样东西：名字、模型对象、训练输入、验证输入
models = [
    ("均值基线", baseline, X_train_s, X_val_s),
    ("单特征LR", single, X_train_bmi, X_val_bmi),
    ("多特征LR", multi, X_train_s, X_val_s),
    ("Ridge α=1", ridge1, X_train_s, X_val_s),
    ("Ridge α=100", ridge100, X_train_s, X_val_s),
]


def mae_rmse(y_true, y_pred):
    """同时返回 MAE 和 RMSE（越小越好）。"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return mae, rmse


# ---------- 步骤 4：训练/验证指标 + RMSE 柱状图 ----------
rows = []            # 收集所有指标行，最后存 CSV
model_names = []
train_rmse_list = []
val_rmse_list = []

for name, model, Xtr, Xva in models:
    tr_pred = model.predict(Xtr)
    va_pred = model.predict(Xva)
    mae_tr, rmse_tr = mae_rmse(y_train, tr_pred)
    mae_va, rmse_va = mae_rmse(y_val, va_pred)
    rows.append({"数据集": "训练", "模型": name, "MAE": round(mae_tr, 2), "RMSE": round(rmse_tr, 2)})
    rows.append({"数据集": "验证", "模型": name, "MAE": round(mae_va, 2), "RMSE": round(rmse_va, 2)})
    model_names.append(name)
    train_rmse_list.append(rmse_tr)
    val_rmse_list.append(rmse_va)

# 画训练 vs 验证 RMSE 的分组柱状图
x_pos = np.arange(len(model_names))
width = 0.35
plt.figure(figsize=(10, 5))
plt.bar(x_pos - width / 2, train_rmse_list, width, label="训练 RMSE")
plt.bar(x_pos + width / 2, val_rmse_list, width, label="验证 RMSE")
plt.xticks(x_pos, model_names)
plt.ylabel("RMSE")
plt.title("各模型训练/验证 RMSE 对比")
plt.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS}/rmse_comparison.png", dpi=150)
plt.close()
print(f"已保存: {RESULTS}/rmse_comparison.png\n")


# ---------- 步骤 5：依据验证 RMSE 选定 alpha ----------
val_rmse_r1 = mae_rmse(y_val, ridge1.predict(X_val_s))[1]
val_rmse_r100 = mae_rmse(y_val, ridge100.predict(X_val_s))[1]
if val_rmse_r1 <= val_rmse_r100:
    best_ridge, best_alpha = ridge1, 1
else:
    best_ridge, best_alpha = ridge100, 100
print(f"两组 Ridge 验证 RMSE：α=1 → {val_rmse_r1:.2f}，α=100 → {val_rmse_r100:.2f}")
print(f"依据验证 RMSE 选定 alpha = {best_alpha}\n")


# ---------- 步骤 6：测试集最终评价（4 个模型） ----------
test_models = [
    ("均值基线", baseline, X_test_s),
    ("单特征LR", single, X_test_bmi),
    ("多特征LR", multi, X_test_s),
    (f"Ridge α={best_alpha}", best_ridge, X_test_s),
]
for name, model, Xte in test_models:
    te_pred = model.predict(Xte)
    mae_te, rmse_te = mae_rmse(y_test, te_pred)
    rows.append({"数据集": "测试", "模型": name, "MAE": round(mae_te, 2), "RMSE": round(rmse_te, 2)})

metrics_df = pd.DataFrame(rows)[["数据集", "模型", "MAE", "RMSE"]]
metrics_df.to_csv(f"{RESULTS}/metrics.csv", index=False)
print("指标比较表（MAE / RMSE，越小越好）：")
print(metrics_df.to_string(index=False))
print(f"\n已保存: {RESULTS}/metrics.csv\n")


# ---------- 步骤 7：权重表 + 权重对比图 ----------
weight_df = pd.DataFrame({
    "特征": feature_cols,
    "多特征LR": np.round(multi.coef_, 4),
    "Ridge α=1": np.round(ridge1.coef_, 4),
    "Ridge α=100": np.round(ridge100.coef_, 4),
})
weight_df.to_csv(f"{RESULTS}/weights.csv", index=False)
print("权重表（标准化输入下的权重）：")
print(weight_df.to_string(index=False))
print(f"\n已保存: {RESULTS}/weights.csv")

plt.figure(figsize=(10, 5))
plt.plot(feature_cols, multi.coef_, marker="o", label="多特征LR")
plt.plot(feature_cols, ridge1.coef_, marker="s", label="Ridge α=1")
plt.plot(feature_cols, ridge100.coef_, marker="^", label="Ridge α=100")
plt.xlabel("特征（标准化输入）")
plt.ylabel("权重")
plt.title("三组模型的权重对比")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS}/weights.png", dpi=150)
plt.close()
print(f"已保存: {RESULTS}/weights.png")
