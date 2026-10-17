# -*- coding: utf-8 -*-
"""
training.py —— 训练与更新逻辑（本周的核心文件）

职责：
1. 实践一：一次梯度更新演示（run_step_demo）。
2. 实践二：一批更新、训练一轮、完整训练循环（含最佳轮次记录）。
3. 实践二：拟合均值基线和线性回归基线。

本文件只做"训练/更新"；评价指标的计算交给 evaluation.py，
保存表格和画图交给 reporting.py。
"""
import copy
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
import evaluation


# ============================================================
# 实践一：把"一次梯度更新"和代码逐行对应起来
# ============================================================
def _grad_item(param):
    """安全地取出参数的梯度数值；还没反向传播时梯度是 None，返回空字符串。"""
    if param.grad is not None:
        return round(float(param.grad.item()), 6)
    return ""


def run_step_demo(model, x, y, loss_fn, optimizer):
    """
    实践一核心：对两个样本完成"一次"梯度更新，并把关键节点记录下来。

    先手动推导一遍（方便和代码输出对照）：
      模型 y = w*x + b，初始 w=0、b=0。
      两个样本 x=[1,2]、y=[2,4]，所以预测都是 0，损失 MSE = (2^2 + 4^2)/2 = 10。

      反向传播算梯度（MSE = mean((pred - y)^2)，n=2）：
        dw = sum_i (pred_i - y_i) * x_i = (0-2)*1 + (0-4)*2 = -10
        db = sum_i (pred_i - y_i)       = (0-2)   + (0-4)   = -6

      更新（SGD，lr=0.1）：
        w_new = 0 - 0.1*(-10) = 1.0
        b_new = 0 - 0.1*(-6)  = 0.6

      用新参数重新前向：pred = 1*x + 0.6 = [1.6, 2.6]，
      新损失 MSE = ((1.6-2)^2 + (2.6-4)^2)/2 = 1.06。

    记录三个阶段（对应三行 step_trace.csv）：
      1. 清梯度后：参数还是初始值，梯度还没算出来。
      2. 反传后  ：loss.backward() 算出了梯度，但参数还没动。
      3. 更新后  ：optimizer.step() 已按梯度更新了参数，并重新前向算了损失。

    返回一个 list，每个元素是一个 dict，字段与 CSV 列一致。
    """
    trace = []

    # ---- 阶段 1：清梯度后（准备本次计算）----
    # zero_grad 把上一轮可能残留的梯度清零，避免累加到本次。
    optimizer.zero_grad()
    trace.append({
        "stage": "清梯度后",
        "weight": round(float(model.weight.item()), 6),
        "bias": round(float(model.bias.item()), 6),
        "weight_grad": _grad_item(model.weight),
        "bias_grad": _grad_item(model.bias),
        "loss": "",
    })

    # ---- 前向 + 损失 ----
    pred = model(x)              # 用当前 w、b 计算预测值
    assert pred.shape == y.shape  # 确认样本和标签一一对应
    loss = loss_fn(pred, y)      # 计算 MSE

    # ---- 阶段 2：反传后（算出梯度）----
    # backward 沿计算图自动求导，把 loss 对 w、b 的梯度存进各自的 .grad。
    # 这就是第二周"手写 dw、db"的替代（组会 B2 会问）。
    loss.backward()
    trace.append({
        "stage": "反传后",
        "weight": round(float(model.weight.item()), 6),
        "bias": round(float(model.bias.item()), 6),
        "weight_grad": _grad_item(model.weight),
        "bias_grad": _grad_item(model.bias),
        "loss": round(float(loss.item()), 6),   # 更新前的损失
    })

    # ---- 阶段 3：更新后（参数变了）----
    # step 按 SGD 规则更新：新参数 = 旧参数 - lr * 梯度（这一步才是真正改参数）。
    optimizer.step()
    # 参数已更新，但之前保存的 pred 不会自动重算，所以要重新前向、重新算损失。
    with torch.no_grad():
        new_pred = model(x)
        new_loss = loss_fn(new_pred, y)
    trace.append({
        "stage": "更新后",
        "weight": round(float(model.weight.item()), 6),
        "bias": round(float(model.bias.item()), 6),
        "weight_grad": _grad_item(model.weight),   # step 不会清梯度，所以还是刚才的值
        "bias_grad": _grad_item(model.bias),
        "loss": round(float(new_loss.item()), 6),   # 更新后的损失
    })

    return trace


# ============================================================
# 实践二：小批量训练循环
# ============================================================
def train_batch(model, xb, yb, loss_fn, optimizer):
    """
    "一批更新"：清梯度 -> 前向 -> 损失 -> 反向 -> 更新。返回这一批的损失。

    xb、yb 是 DataLoader 取出的"当前批次"（组会 A5 会问）：
    - 本次梯度只用这一批里的样本计算，不是整个训练集；
    - 换一批样本，梯度自然不同。
    """
    optimizer.zero_grad()          # 清梯度，为本次计算做准备
    pred = model(xb)               # 前向：用当前参数算这批的预测
    loss = loss_fn(pred, yb)       # 损失：把这批预测和对应标签比较
    loss.backward()                # 反向：自动求导，算出各参数梯度
    optimizer.step()               # 更新：按优化器规则用梯度更新参数
    return float(loss.item())


def train_epoch(model, loader, loss_fn, optimizer):
    """训练一轮：按顺序遍历 DataLoader 里的所有批，逐批更新参数。"""
    for xb, yb in loader:
        train_batch(model, xb, yb, loss_fn, optimizer)


def train_model(model, optimizer, loss_fn, loader,
                X_train_t, y_train_t, X_val_t, y_val_t, epochs):
    """
    完整训练：外层 epoch、内层 batch 两层循环（组会 B2 会问整体流程）。

    每轮流程：
      1. model.train()：切回训练模式（上一轮评价时被切到过 eval）。
      2. 逐批更新参数（见 train_epoch）。
      3. 这一轮结束后，用"完整训练集 / 完整验证集"重新评价 MSE，
         得到这一轮真正的训练、验证 MSE（组会 B4 会问为什么不能
         直接拿最后一批的 loss 当本轮训练 MSE）。
      4. 若验证 MSE 创历史新低，就记下这一轮的轮次和参数（深拷贝）。
         只有"严格更小"才更新，所以并列时保留较早那一轮（组会 B5）。

    训练结束后把参数恢复到"最佳那一轮"，并返回 (history, best_epoch, best_val_mse)。
    """
    model.train()
    history = []                  # 每轮一条：{epoch, train_mse, val_mse}
    best_val_mse = float("inf")
    best_epoch = None
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()             # 关键：恢复训练模式（评价后要切回来）
        train_epoch(model, loader, loss_fn, optimizer)

        # 用完整数据重新评价这一轮的表现
        train_mse = evaluation.mse_scaled(model, X_train_t, y_train_t)
        val_mse = evaluation.mse_scaled(model, X_val_t, y_val_t)
        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse})

        # 记录历史最佳验证轮次与参数
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    # 恢复到验证表现最好那一轮的参数（而不是最后一轮）
    model.load_state_dict(best_state)
    return history, best_epoch, best_val_mse


# ============================================================
# 实践二：基线拟合
# ============================================================
def fit_mean_baseline(y_train_raw):
    """
    均值基线：不管输入是什么，永远预测训练目标的均值（原始单位）。
    这是一个"不利用特征"的参考水平，用来衡量模型是不是真的学到了东西。
    """
    return float(np.mean(y_train_raw))


def fit_linear_baseline(X_train_np, y_train_raw):
    """
    线性回归基线：用"标准化输入 + 原始目标"拟合 LinearRegression。
    因为目标用的是原始值，所以它的预测直接在原始单位，无需再恢复尺度。
    """
    lr = LinearRegression()
    lr.fit(X_train_np, y_train_raw)
    return lr
