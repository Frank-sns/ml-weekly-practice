# -*- coding: utf-8 -*-
"""
第一周实践二：diabetes 数据读取与绘图

功能：
  1. 查看数据规模、列名、前几行，区分特征列与目标列
  2. 统计各列缺失值，查看数值列基本统计
  3. 绘制目标值分布图、一个特征与目标的散点图，并保存结果

数据来源：scikit-learn 内置的 load_diabetes 回归数据集（原始特征，未标准化）。
运行方式：在本项目根目录执行  python diabetes_analysis.py
"""
import os
import sys

# 让 print 的中文输出在 Git Bash 下正常显示（终端默认编码与 UTF-8 不一致会导致乱码）
sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
# 使用无界面后端，脚本直接保存图片，不弹窗
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes

# 中文字体配置：优先微软雅黑，解决中文标签显示为方块的问题
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示

# ---------- 1. 加载数据 ----------
dataset = load_diabetes(as_frame=True, scaled=False)
df = dataset.frame

print("=== 1. 数据规模与结构 ===")
print(f"形状(行, 列): {df.shape}")
print(f"\n列名: {df.columns.tolist()}")

# 特征列 = 除 target 外的所有列；目标列 = target
feature_cols = [c for c in df.columns if c != "target"]
target_col = "target"
print(f"\n前 5 行:\n{df.head()}")
print(f"\n特征列({len(feature_cols)} 个): {feature_cols}")
print(f"目标列: {target_col}")

# ---------- 2. 缺失值与基本统计 ----------
print("\n=== 2. 缺失值统计 ===")
print(df.isna().sum())

print("\n=== 数值列基本统计 ===")
print(df.describe())

# ---------- 3. 绘图 ----------
os.makedirs("figures", exist_ok=True)

# 3.1 目标值分布直方图
plt.figure(figsize=(8, 5))
plt.hist(df[target_col], bins=30, edgecolor="white", color="#4C72B0")
plt.xlabel("目标值 target（一年后疾病进展量化指标）")
plt.ylabel("样本数")
plt.title("diabetes 目标值分布")
plt.tight_layout()
plt.savefig("figures/target_distribution.png", dpi=150)
plt.close()
print("\n已保存: figures/target_distribution.png")

# 3.2 特征与目标散点图（本次改为 age，作为一次有目的的修改：观察年龄与目标的关系）
feat = "age"
plt.figure(figsize=(8, 5))
plt.scatter(df[feat], df[target_col], alpha=0.6, color="#DD8452")
plt.xlabel(f"特征 {feat}（年龄）")
plt.ylabel("目标 target（疾病进展量化指标）")
plt.title(f"{feat} 与 target 的散点图")
plt.tight_layout()
plt.savefig("figures/scatter_age_target.png", dpi=150)
plt.close()
print("已保存: figures/scatter_age_target.png")
