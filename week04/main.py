# -*- coding: utf-8 -*-
"""
第四周：PCA 降维与 K-means 聚类

本周做两件新事（都属于"无监督学习"，只用输入 X、不需要答案 y）：
  1. 实践一：PCA 降维前后，线性回归预测效果对照
     —— 把 10 个特征"压缩"成 2 个 / 5 个主成分，看预测准度怎么变。
  2. 实践二：K-means 聚类
     —— 让程序自己把样本分成 2 组 / 3 组，用轮廓系数和特征均值解释分组。

数据：还是 scikit-learn 自带的 diabetes（10 个特征 + 1 个目标 target）。

整个脚本按顺序执行，前两步（加载、划分、标准化）两个实践共用，之后各自跑。
运行：在 week04/ 目录下执行  python main.py
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")   # 修 Windows 控制台中文乱码

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")   # 无窗口后端，直接把图存成文件（必须写在 import pyplot 之前）
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler      # 标准化
from sklearn.decomposition import PCA                 # 主成分分析
from sklearn.dummy import DummyRegressor              # 均值基线
from sklearn.linear_model import LinearRegression     # 线性回归
from sklearn.cluster import KMeans                    # K-means 聚类
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    silhouette_score,          # 轮廓系数：衡量聚类分组清不清楚
)

# 中文字体配置（让图里的中文正常显示）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)   # exist_ok=True：目录已存在也不报错


# ============================================================
# 第一部分：加载数据 + 划分 + 标准化（两个实践共用）
# ============================================================
# 读数据。as_frame=True 得到表格，scaled=False 要"原始"数据（标准化我们自己做）
dataset = load_diabetes(as_frame=True, scaled=False)
df = dataset.frame

# 特征列 = 除 target 以外的所有列（共 10 个）
feature_cols = [c for c in df.columns if c != "target"]
X = df[feature_cols].to_numpy()   # 输入特征，形状 (442, 10)
y = df["target"].to_numpy()       # 预测目标，形状 (442,)

# 固定划分 60% / 20% / 20%。random_state=42 固定随机种子，保证每次划分结果一样
X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=42)

print(f"数据划分：训练 {len(y_train)} / 验证 {len(y_val)} / 测试 {len(y_test)}\n")

# 标准化：只在"训练集"上算均值和标准差，再套用到验证/测试集（防止信息泄漏）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit 先学统计量，再 transform 套用
X_val_scaled = scaler.transform(X_val)           # 验证/测试只用训练集学到的统计量
X_test_scaled = scaler.transform(X_test)

print("标准化完成（每列均值 0、标准差 1，只用训练集统计量）\n")


# ============================================================
# 实践一：PCA 前后的线性回归对照
# ============================================================
# PCA 会把 10 个特征"揉"成几个新的"主成分"。
# 主成分不是从原特征里挑两列，而是原特征按不同权重相加出来的新坐标。

# ---------- 步骤 1：拟合两个 PCA（保留 2 维 和 5 维）----------
# svd_solver="full"、whiten=False 是文档要求的标准设置
pca2 = PCA(n_components=2, svd_solver="full", whiten=False)
pca2.fit(X_train_scaled)                          # 只在训练集上"学投影方向"
Z_train2 = pca2.transform(X_train_scaled)         # 把三个集合都投影成 2 个主成分坐标
Z_val2 = pca2.transform(X_val_scaled)
Z_test2 = pca2.transform(X_test_scaled)

pca5 = PCA(n_components=5, svd_solver="full", whiten=False)
pca5.fit(X_train_scaled)
Z_train5 = pca5.transform(X_train_scaled)
Z_val5 = pca5.transform(X_val_scaled)
Z_test5 = pca5.transform(X_test_scaled)

print(f"PCA 完成：2 维投影形状 {Z_train2.shape}，5 维投影形状 {Z_train5.shape}\n")


# ---------- 步骤 2：训练 4 个模型 ----------
# 每个模型"输入不同"，但"训练样本相同"，标签都是 y_train
baseline = DummyRegressor(strategy="mean")        # 均值基线：不管输入，永远预测训练集均值
baseline.fit(X_train_scaled, y_train)

lr_full = LinearRegression()                      # 不降维：直接用 10 个标准化特征
lr_full.fit(X_train_scaled, y_train)

lr_pca2 = LinearRegression()                      # 降维到 2 维：用 2 个主成分
lr_pca2.fit(Z_train2, y_train)

lr_pca5 = LinearRegression()                      # 降维到 5 维：用 5 个主成分
lr_pca5.fit(Z_train5, y_train)


# ---------- 小工具：同时算 MAE 和 RMSE（都是越小越好）----------
def mae_rmse(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return mae, rmse


# ---------- 步骤 3：算训练/验证指标 + 画 RMSE 柱状图 ----------
# 把 4 个模型打包：名字、模型、训练输入、验证输入、PCA 维数（用来写进表格）
models = [
    ("均值基线", baseline, X_train_scaled, X_val_scaled, ""),
    ("未降维LR", lr_full, X_train_scaled, X_val_scaled, "无"),
    ("PCA 2维 + LR", lr_pca2, Z_train2, Z_val2, "2"),
    ("PCA 5维 + LR", lr_pca5, Z_train5, Z_val5, "5"),
]

rows = []              # 收集每一行指标，最后存 CSV
model_names = []       # 模型名（画图横轴用）
train_rmse_list = []   # 训练 RMSE
val_rmse_list = []     # 验证 RMSE

for name, model, Xtr, Xva, dim in models:
    tr_pred = model.predict(Xtr)
    va_pred = model.predict(Xva)
    mae_tr, rmse_tr = mae_rmse(y_train, tr_pred)
    mae_va, rmse_va = mae_rmse(y_val, va_pred)
    rows.append({"数据集": "训练", "模型": name, "PCA维数": dim,
                 "MAE": round(mae_tr, 2), "RMSE": round(rmse_tr, 2)})
    rows.append({"数据集": "验证", "模型": name, "PCA维数": dim,
                 "MAE": round(mae_va, 2), "RMSE": round(rmse_va, 2)})
    model_names.append(name)
    train_rmse_list.append(rmse_tr)
    val_rmse_list.append(rmse_va)

# 画"训练 vs 验证 RMSE"分组柱状图
x_pos = np.arange(len(model_names))
width = 0.35
plt.figure(figsize=(10, 5))
plt.bar(x_pos - width / 2, train_rmse_list, width, label="训练 RMSE")
plt.bar(x_pos + width / 2, val_rmse_list, width, label="验证 RMSE")
plt.xticks(x_pos, model_names)
plt.ylabel("RMSE（越小越好）")
plt.title("PCA 降维前后各模型训练/验证 RMSE 对比")
plt.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS}/rmse_comparison.png", dpi=150)
plt.close()
print(f"已保存: {RESULTS}/rmse_comparison.png\n")


# ---------- 步骤 4：依据验证 RMSE 选定 PCA 维数 ----------
val_rmse_pca2 = mae_rmse(y_val, lr_pca2.predict(Z_val2))[1]   # 末尾 [1] 取 RMSE
val_rmse_pca5 = mae_rmse(y_val, lr_pca5.predict(Z_val5))[1]

if val_rmse_pca2 <= val_rmse_pca5:
    best_pca, best_n = lr_pca2, 2
    best_Z_test = Z_test2
else:
    best_pca, best_n = lr_pca5, 5
    best_Z_test = Z_test5

print(f"两组 PCA 验证 RMSE：2 维 → {val_rmse_pca2:.2f}，5 维 → {val_rmse_pca5:.2f}")
print(f"依据验证 RMSE 选定 PCA 维数 = {best_n}\n")


# ---------- 步骤 5：测试集最终评价（3 个模型，另一组 PCA 不测）----------
test_models = [
    ("均值基线", baseline, X_test_scaled, ""),
    ("未降维LR", lr_full, X_test_scaled, "无"),
    (f"PCA {best_n}维 + LR（选定）", best_pca, best_Z_test, str(best_n)),
]
for name, model, Xte, dim in test_models:
    te_pred = model.predict(Xte)
    mae_te, rmse_te = mae_rmse(y_test, te_pred)
    rows.append({"数据集": "测试", "模型": name, "PCA维数": dim,
                 "MAE": round(mae_te, 2), "RMSE": round(rmse_te, 2)})

metrics_df = pd.DataFrame(rows)[["数据集", "模型", "PCA维数", "MAE", "RMSE"]]
metrics_df.to_csv(f"{RESULTS}/metrics.csv", index=False)
print("指标比较表（MAE / RMSE，越小越好）：")
print(metrics_df.to_string(index=False))
print(f"\n已保存: {RESULTS}/metrics.csv\n")


# ---------- 步骤 6：解释方差 + 累计解释方差曲线 ----------
# explained_variance_ratio_ 表示"每个主成分各自保留了多少输入信息"
var2 = float(pca2.explained_variance_ratio_.sum())   # 2 维累计解释方差比例
var5 = float(pca5.explained_variance_ratio_.sum())   # 5 维累计解释方差比例
print(f"累计解释方差比例：保留 2 维 = {var2:.4f}，保留 5 维 = {var5:.4f}")

# 再拟合一个保留全部 10 维的 PCA，画 1~10 维的累计方差曲线（看完整趋势）
pca_full = PCA(n_components=10, svd_solver="full", whiten=False)
pca_full.fit(X_train_scaled)
cum_ratio = np.cumsum(pca_full.explained_variance_ratio_)   # cumsum：逐个累加

plt.figure(figsize=(9, 5))
plt.plot(range(1, 11), cum_ratio, marker="o", label="累计解释方差比例")
plt.axvline(2, color="gray", linestyle="--", linewidth=0.8)  # 竖虚线标出 2 维
plt.axvline(5, color="gray", linestyle="--", linewidth=0.8)  # 竖虚线标出 5 维
plt.annotate("2 维", xy=(2, cum_ratio[1]), xytext=(2.3, cum_ratio[1] - 0.05))
plt.annotate("5 维", xy=(5, cum_ratio[4]), xytext=(5.3, cum_ratio[4] - 0.05))
plt.xlabel("保留的主成分维数")
plt.ylabel("累计解释方差比例（0~1，越接近 1 保留信息越多）")
plt.title("保留不同维数时，输入信息被保留的比例")
plt.xticks(range(1, 11))
plt.ylim(0, 1.05)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS}/pca_variance.png", dpi=150)
plt.close()
print(f"已保存: {RESULTS}/pca_variance.png\n")


# ---------- 步骤 7：重构误差 + PCA 信息表 ----------
# inverse_transform：把降维后的主成分坐标"还原"回 10 维，看丢了多少信息。
# 丢得越多，还原出来的和原来的差得越多 → 重构误差越大。
def recon_mse(pca, Z, X_scaled):
    X_recon = pca.inverse_transform(Z)                       # 用主成分近似还原 10 维
    return float(np.mean((X_scaled - X_recon) ** 2))         # 对全部样本、全部列取平方差的平均

recon_train2 = recon_mse(pca2, Z_train2, X_train_scaled)
recon_val2 = recon_mse(pca2, Z_val2, X_val_scaled)
recon_train5 = recon_mse(pca5, Z_train5, X_train_scaled)
recon_val5 = recon_mse(pca5, Z_val5, X_val_scaled)

pca_summary_df = pd.DataFrame([
    {"维数": 2, "累计解释方差比例": round(var2, 4),
     "训练重构MSE": round(recon_train2, 4), "验证重构MSE": round(recon_val2, 4)},
    {"维数": 5, "累计解释方差比例": round(var5, 4),
     "训练重构MSE": round(recon_train5, 4), "验证重构MSE": round(recon_val5, 4)},
])
pca_summary_df.to_csv(f"{RESULTS}/pca_summary.csv", index=False)
print("PCA 信息表（重构 MSE 越小，说明降维丢的信息越少）：")
print(pca_summary_df.to_string(index=False))
print(f"\n已保存: {RESULTS}/pca_summary.csv\n")


# ============================================================
# 实践二：K-means 聚类
# ============================================================
# K-means 是无监督学习：只用特征 X，不用答案 y。
# 它按"距离"把样本分到最近的组，再反复用组内均值更新组中心。

# ---------- 步骤 1：训练 K=2 和 K=3 两组 ----------
# n_clusters 是要分的组数；n_init=10 表示尝试 10 次初始化、保留最好的一次
km2 = KMeans(n_clusters=2, random_state=42, n_init=10)
km2.fit(X_train_scaled)

km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
km3.fit(X_train_scaled)

# labels_ 是每个训练样本被分到的组号；cluster_centers_ 是每组的中心（10 维）
labels2 = km2.labels_
labels3 = km3.labels_


# ---------- 步骤 2：轮廓系数 ----------
# 轮廓系数衡量"组内是否紧密、组间是否分开"，越高说明分组越清楚（范围约 -1 到 1）
sil2 = silhouette_score(X_train_scaled, labels2, metric="euclidean")
sil3 = silhouette_score(X_train_scaled, labels3, metric="euclidean")
print(f"轮廓系数：K=2 → {sil2:.4f}，K=3 → {sil3:.4f}（越高分组越清楚）\n")


# ---------- 步骤 3：统计每组样本数 + 原始 bmi / bp 均值 ----------
# 组号是按"标准化特征"分的，但统计均值时用"原始(未标准化)数据"更方便理解
bmi_idx = feature_cols.index("bmi")
bp_idx = feature_cols.index("bp")

cluster_rows = []
for k_val, km in [(2, km2), (3, km3)]:
    labels = km.labels_
    for g in range(k_val):
        mask = labels == g                       # 挑出属于第 g 组的样本（布尔掩码）
        n = int(mask.sum())                      # 该组样本数
        bmi_mean = X_train[mask, bmi_idx].mean() # 该组原始 bmi 均值
        bp_mean = X_train[mask, bp_idx].mean()   # 该组原始 bp 均值
        cluster_rows.append({"K": k_val, "组号": g, "样本数": n,
                             "bmi均值": round(float(bmi_mean), 2),
                             "bp均值": round(float(bp_mean), 2)})

cluster_summary_df = pd.DataFrame(cluster_rows)
cluster_summary_df.to_csv(f"{RESULTS}/cluster_summary.csv", index=False)
print("各组样本数 + 原始 bmi/bp 均值：")
print(cluster_summary_df.to_string(index=False))

# 核对：各组样本数加起来应该等于训练样本总数
for k_val, km in [(2, km2), (3, km3)]:
    total = int((km.labels_ >= 0).sum())
    print(f"  K={k_val} 样本数之和 = {total}（应等于训练样本数 {len(y_train)}）")
print(f"\n已保存: {RESULTS}/cluster_summary.csv\n")


# ---------- 步骤 4：轮廓系数等信息表 ----------
cluster_metrics_df = pd.DataFrame([
    {"K": 2, "轮廓系数": round(sil2, 4), "随机种子": 42, "n_init": 10, "特征维数": 10},
    {"K": 3, "轮廓系数": round(sil3, 4), "随机种子": 42, "n_init": 10, "特征维数": 10},
])
cluster_metrics_df.to_csv(f"{RESULTS}/cluster_metrics.csv", index=False)
print("聚类指标表：")
print(cluster_metrics_df.to_string(index=False))
print(f"\n已保存: {RESULTS}/cluster_metrics.csv\n")


# ---------- 步骤 5：双面板聚类图（PC1 vs PC2）----------
# 聚类是在 10 维空间做的，但人只能看二维，所以用实践一的 2 维 PCA 投影到平面展示。
# 注意：投影后图上可能有重叠，这是"10 维聚成 2 维"的正常现象。
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, k_val, km in zip(axes, [2, 3], [km2, km3]):
    labels = km.labels_
    # 训练样本的 2 维主成分坐标（前两列就是 PC1、PC2）
    Z2 = Z_train2
    ax.scatter(Z2[:, 0], Z2[:, 1], c=labels, cmap="viridis", s=15, alpha=0.7)
    # 组中心原本在 10 维标准化空间，也要投影到 2 维
    centers_2d = pca2.transform(km.cluster_centers_)
    ax.scatter(centers_2d[:, 0], centers_2d[:, 1], c="red", marker="X",
               s=200, edgecolors="black", label="组中心")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"K = {k_val}（10 维聚类，2 维展示）")
    ax.legend()

plt.tight_layout()
plt.savefig(f"{RESULTS}/cluster_comparison.png", dpi=150)
plt.close()
print(f"已保存: {RESULTS}/cluster_comparison.png")

print("\n全部完成！结果都在 week04/results/ 目录下。")
