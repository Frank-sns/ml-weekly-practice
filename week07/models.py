# -*- coding: utf-8 -*-
"""
models.py —— LSTM 模型、形状检查，以及两种上周基线

本周模型很小：
    输入 (B, 24, 1) → LSTM(1 → H) → 取最后一个时间步的隐藏状态 (B, H)
                   → Linear(H → 1) → 预测 (B, 1)

为什么只取最后一个时间步：
    LSTM 逐步读取 24 个小时，每读一步就更新一次 (h, c)。读完最后一个小时后，
    h 是这个窗口"组织后的表示"，里面压缩了模型认为与预测有关的历史信息。
    输出层再把这个表示映射成一个数。

为什么输出层不加激活函数：
    目标是标准化后的油温，可以取任意实数（正负都行），
    加 ReLU/sigmoid 反而会把预测限死在一个范围里。
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class LSTMRegressor(nn.Module):
    """
    单层、单向 LSTM + 一个线性输出层。

    参数（与任务书要求一致）：
      input_size=1     每个时间步只有 OT 一个特征
      batch_first=True 输入/输出的第 0 维是 batch
      dropout=0        单层 LSTM 的 dropout 不起作用，显式设 0
      proj_size=0      不做投影
      num_layers=1     本周只训练一层
      bidirectional=False  单向，只从过去读到当前
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            proj_size=0,
            bidirectional=False,
        )
        # 输出层：H 维隐藏表示 → 1 个预测值，不加激活
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """正常前向：只返回预测值 (B, 1)，训练时用这个。"""
        output, _ = self.lstm(x)
        last_step = output[:, -1, :]        # 取最后一个时间步： (B, H)
        return self.head(last_step)

    def forward_with_states(self, x):
        """
        观察用前向：额外把序列输出、最终隐藏状态、最终细胞状态返回出来。

        注意这些 h_n / c_n 只是"读数"，不是训练时手动传回去的状态。
        PyTorch 内部每一步都会把状态传给下一步；这里取出来是为了看形状。
        """
        output, (h_n, c_n) = self.lstm(x)
        last_step = output[:, -1, :]
        pred = self.head(last_step)
        return pred, output, h_n, c_n


def count_parameters(model):
    """
    统计可训练参数总数（只统计 requires_grad=True 的）。

    LSTM 的参数来自四组变换（输入门 i、遗忘门 f、候选 g、输出门 o），
    每组都是 (输入维度 + 隐藏维度) × 隐藏维度 的权重 + 隐藏维度 的偏置：
        4 × H × (1 + H + 1)
    输出层：H × 1 + 1
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def describe_shapes(model, x_batch):
    """
    喂一小批真实样本，打印并返回各张量的形状。

    返回字典，键：input / output / h_n / c_n / pred / head_input
    """
    model.eval()
    with torch.no_grad():
        pred, output, h_n, c_n = model.forward_with_states(x_batch)

    shapes = {
        "input": tuple(x_batch.shape),
        "output": tuple(output.shape),
        "h_n": tuple(h_n.shape),
        "c_n": tuple(c_n.shape),
        "pred": tuple(pred.shape),
        "head_input": tuple(output[:, -1, :].shape),
        "param_count": count_parameters(model),
    }
    return shapes


def check_last_step_equivalence(model, x_batch):
    """
    核对：单层、单向时，序列输出的最后一步 == 最终隐藏状态。
    这是组会 B3 的常见追问，用实际张量比对而不是口头保证。
    """
    model.eval()
    with torch.no_grad():
        _pred, output, h_n, _c_n = model.forward_with_states(x_batch)
    last_from_output = output[:, -1, :]
    last_from_h_n = h_n[-1]          # h_n 第 0 维是"层数 × 方向数"
    return bool(torch.allclose(last_from_output, last_from_h_n, atol=1e-6))


# ---------------- 上周基线（本文件内重新实现，确保比较口径一致） ----------------

def predict_last_value(X):
    """
    last_value 基线：直接用最近一次观测作为预测。

    X 可以是 (N, 24, 1) 或 (N, 24)；都取第 24 个时间步（最近一小时）。
    这是一个不学参数的规则。
    """
    X = np.asarray(X)
    if X.ndim == 3:
        return X[:, -1, 0]
    return X[:, -1]


def fit_lag_linear(X, y):
    """
    lag_linear 基线：把 24 个历史值当成 24 个特征，做 标准化 + 线性回归。

    注意：这里用的是【原始油温】和它自己的标准化管道，
    不套用 LSTM 的 μ/σ —— 任务书明确要求两种预处理分开。
    """
    X2 = np.asarray(X)
    if X2.ndim == 3:
        X2 = X2[:, :, 0]
    return make_pipeline(StandardScaler(), LinearRegression()).fit(X2, y)
