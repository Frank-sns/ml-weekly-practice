# -*- coding: utf-8 -*-
"""
第二周实践：单特征线性回归（bmi）—— NumPy 梯度下降与模型评价

目标：理解线性回归、MSE 损失、梯度下降，以及"训练"与"评价"的区别。
只用 bmi 一个特征拟合  预测值 = w × bmi + b，重点观察参数 w、b 如何随训练变化。

流程（对应本周 8 个步骤）：
  1. 划分 60% 训练 / 20% 验证 / 20% 测试（固定随机种子 42）
  2. 只用训练集统计量标准化特征
  3. 均值基线（DummyRegressor）
  4. NumPy 手写梯度下降（w=b=0 起步，每轮用全部训练样本）
  5. 比较两个学习率 0.01 / 0.1 的损失曲线
  6. 与 sklearn LinearRegression 对照
  7. 计算 MAE / RMSE 评价泛化
  8. 测试集真实值 vs 预测值散点图（含 y=x 参考线）

运行：在本目录（week02/）下执行  python main.py
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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 中文字体配置
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)


# ---------- 步骤 1：加载数据 + 划分 60/20/20 ----------
dataset = load_diabetes(as_frame=True, scaled=False)
# bmi 取成二维 (n, 1)，sklearn 模型要求二维输入；target 取一维 (n,)
X = dataset.frame[["bmi"]].to_numpy()
y = dataset.frame["target"].to_numpy()

# 先留出 40%（验证+测试），再把这 40% 均分为验证和测试，得到 60/20/20
X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=42)

print(f"数据划分：训练 {len(y_train)} / 验证 {len(y_val)} / 测试 {len(y_test)}")
print(f"  占比：训练 {len(y_train)/len(y):.1%}，验证 {len(y_val)/len(y):.1%}，测试 {len(y_test)/len(y):.1%}\n")


# ---------- 步骤 2：只用训练集统计量标准化特征 ----------
# 关键：均值/标准差只用训练集算，再用这同一组数字变换验证集和测试集。
# 不能分别对三部分各自算（等于"偷看"验证/测试信息），也不能先对全部数据算再划分（同样泄露未来信息）。
mean_bmi = X_train.mean()
std_bmi = X_train.std()
X_train_s = (X_train - mean_bmi) / std_bmi
X_val_s = (X_val - mean_bmi) / std_bmi
X_test_s = (X_test - mean_bmi) / std_bmi

print(f"标准化统计量（仅训练集）：bmi 均值 = {mean_bmi:.3f}，标准差 = {std_bmi:.3f}\n")


# ---------- 步骤 3：均值基线 ----------
# 不看任何特征，对每条输入都预测训练集 target 的均值。看似"没用"，
# 却是重要的对照：任何正经模型至少得比它好，否则还不如直接猜平均数。
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train_s, y_train)
print(f"均值基线：预测值恒为训练集 target 均值 = {y_train.mean():.2f}\n")


# ---------- 步骤 4：NumPy 手写梯度下降 ----------
def gradient_descent(X, y, lr, epochs):
    """批量梯度下降，拟合 y = w*x + b。

    X、y 传入后先 ravel() 成一维，避免二维预测与一维目标相减时
    广播成 (n, n) 矩阵（这是新手最容易踩的形状坑）。
    """
    X = X.ravel()
    y = y.ravel()
    w = 0.0  # 权重（斜率）
    b = 0.0  # 截距
    losses = []
    for _ in range(epochs):
        y_pred = w * X + b                # ① 预测：用当前参数算输出
        error = y_pred - y                # ② 误差：预测值减真实值
        mse = float(np.mean(error ** 2))  # 损失 MSE = mean(error^2)
        losses.append(mse)
        dw = 2 * np.mean(error * X)       # ③ 梯度：损失对 w 的偏导
        db = 2 * np.mean(error)           # ③ 梯度：损失对 b 的偏导
        w -= lr * dw                      # ④ 更新参数：沿负梯度方向走一步
        b -= lr * db
    return w, b, losses


def predict_linear(w, b, X):
    """用线性模型参数 w、b 对特征 X 做预测。"""
    return w * X.ravel() + b


# ---------- 步骤 5：比较两个学习率 ----------
LR1, LR2 = 0.01, 0.1
EPOCHS = 500

w1, b1, losses1 = gradient_descent(X_train_s, y_train, LR1, EPOCHS)
w2, b2, losses2 = gradient_descent(X_train_s, y_train, LR2, EPOCHS)

print(f"梯度下降训练完成（各 {EPOCHS} 轮）")
print(f"  学习率 {LR1}：w = {w1:.4f}, b = {b1:.4f}, 最终训练 MSE = {losses1[-1]:.2f}")
print(f"  学习率 {LR2}：w = {w2:.4f}, b = {b2:.4f}, 最终训练 MSE = {losses2[-1]:.2f}\n")

plt.figure(figsize=(8, 5))
plt.plot(losses1, label=f"lr={LR1}")
plt.plot(losses2, label=f"lr={LR2}")
plt.xlabel("训练轮数 (epoch)")
plt.ylabel("训练损失 MSE")
plt.title("不同学习率下梯度下降的损失曲线")
plt.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS}/loss_curves.png", dpi=150)
plt.close()
print(f"已保存: {RESULTS}/loss_curves.png\n")


# ---------- 步骤 6：与库函数 LinearRegression 对照 ----------
# LinearRegression 直接解最小二乘（闭式解），不是梯度下降；这里当作"正确答案"做对照。
lr_model = LinearRegression()
lr_model.fit(X_train_s, y_train)
w_lr = lr_model.coef_[0]
b_lr = lr_model.intercept_
y_train_pred_lr = lr_model.predict(X_train_s)
train_mse_lr = mean_squared_error(y_train, y_train_pred_lr)

print("与库函数对照：")
print(f"  LinearRegression    ：w = {w_lr:.4f}, b = {b_lr:.4f}, 训练 MSE = {train_mse_lr:.2f}")
print(f"  梯度下降 lr={LR2}    ：w = {w2:.4f}, b = {b2:.4f}, 训练 MSE = {losses2[-1]:.2f}")


# 比较两者预测值本身的平均绝对差（越小说明越接近）
gd_pred_lr2 = predict_linear(w2, b2, X_train_s)
pred_diff = float(np.mean(np.abs(gd_pred_lr2 - y_train_pred_lr)))
print(f"  两者预测值的平均绝对差 = {pred_diff:.4f}（应很小）")
print("  （两者应接近；若差很多，检查梯度、数据形状、迭代次数或学习率）\n")


# ---------- 步骤 7：MAE / RMSE 评价泛化 ----------
def make_row(name, y_true, y_pred):
    return {
        "模型": name,
        "MAE": round(mean_absolute_error(y_true, y_pred), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
    }


rows = []

# 训练集、验证集：所有模型都算（用来观察拟合与选模型）
for name, pred_fn in [
    ("均值基线", lambda X: baseline.predict(X)),
    (f"GD lr={LR1}", lambda X: predict_linear(w1, b1, X)),
    (f"GD lr={LR2}", lambda X: predict_linear(w2, b2, X)),
    ("LinearRegression", lambda X: lr_model.predict(X)),
]:
    rows.append({"数据集": "训练", **make_row(name, y_train, pred_fn(X_train_s))})
    rows.append({"数据集": "验证", **make_row(name, y_val, pred_fn(X_val_s))})

# 依据验证集 RMSE 选定一个梯度下降配置（用验证集选，不用测试集选）
val_rmse_gd1 = np.sqrt(mean_squared_error(y_val, predict_linear(w1, b1, X_val_s)))
val_rmse_gd2 = np.sqrt(mean_squared_error(y_val, predict_linear(w2, b2, X_val_s)))
best_lr = LR1 if val_rmse_gd1 <= val_rmse_gd2 else LR2
best_w = w1 if best_lr == LR1 else w2
best_b = b1 if best_lr == LR1 else b2
print(f"依据验证集 RMSE 选定梯度下降配置：lr={best_lr}\n")

# 测试集：只对 均值基线、选定 GD、库函数 计算（测试集只用一次，不反复调方案）
for name, pred_fn in [
    ("均值基线", lambda X: baseline.predict(X)),
    (f"GD lr={best_lr}", lambda X: predict_linear(best_w, best_b, X)),
    ("LinearRegression", lambda X: lr_model.predict(X)),
]:
    rows.append({"数据集": "测试", **make_row(name, y_test, pred_fn(X_test_s))})

metrics_df = pd.DataFrame(rows)[["数据集", "模型", "MAE", "RMSE"]]
metrics_df.to_csv(f"{RESULTS}/metrics.csv", index=False)
print("指标比较表（MAE / RMSE，越小越好）：")
print(metrics_df.to_string(index=False))
print(f"\n已保存: {RESULTS}/metrics.csv\n")


# ---------- 步骤 8：检查预测 ----------
y_test_pred = predict_linear(best_w, best_b, X_test_s)

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_test_pred, alpha=0.6, color="#4C72B0")
lo = min(y_test.min(), y_test_pred.min())
hi = max(y_test.max(), y_test_pred.max())
plt.plot([lo, hi], [lo, hi], "r--", label="y = x")
plt.xlabel("真实值 target")
plt.ylabel("预测值")
plt.title(f"测试集：真实值 vs 预测值（GD lr={best_lr}）")
plt.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS}/pred_vs_true.png", dpi=150)
plt.close()
print(f"已保存: {RESULTS}/pred_vs_true.png")

# 核对几条样本
print("\n核对前 5 条测试样本（真实值 / 预测值 / 误差）：")
for i in range(5):
    err = y_test[i] - y_test_pred[i]
    print(f"  样本{i}: 真实 = {y_test[i]:6.1f}，预测 = {y_test_pred[i]:6.1f}，误差 = {err:6.1f}")

print("\n说明：训练损失不断下降，只代表模型在训练集上拟合得越来越好；")
print("对新数据（测试集）预测是否准确，要看测试集指标和散点图离 y=x 线的远近。")
