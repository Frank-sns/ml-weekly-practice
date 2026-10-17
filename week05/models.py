# -*- coding: utf-8 -*-
"""
models.py —— 模型定义

职责：
1. 实践一：单层线性模型（零初始化，方便手算核对梯度）。
2. 实践二：一个小型 MLP（一个隐藏层，宽度可调）。

MLP 的单条样本前向公式（组会 A1/A2/A3 要会讲）：
    隐藏层：h = ReLU(W1 * x + b1)     # 10 个输入 -> hidden_dim 个隐藏特征
    输出层：pred = W2 * h + b2        # hidden_dim 个特征 -> 1 个预测值

即：先从原始输入学出一组"中间特征" h，再用 h 算最终的预测值。
"""
import torch
import torch.nn as nn


def make_zero_linear():
    """
    实践一用：单层线性模型 y = w * x + b，输入/输出都是 1 维。

    特意把权重 w 和偏置 b 都初始化为 0：
    - 预测一开始是 0，损失可以手算核对；
    - 梯度也容易手算（见 training.py 里的说明）。
    注意：这只是实践一这个小例子方便核对才用零初始化，
    实践二的 MLP 用 PyTorch 默认的随机初始化。
    """
    model = nn.Linear(1, 1)
    with torch.no_grad():   # 用 no_grad 包裹，避免这次赋值被当成需要求导的操作
        model.weight.zero_()
        model.bias.zero_()
    return model


def make_mlp(input_dim=10, hidden_dim=16, output_dim=1):
    """
    实践二用：一个隐藏层的 MLP。

    结构：Linear(10, hidden_dim) -> ReLU -> Linear(hidden_dim, 1)

    ⭐ hidden_dim 同时控制两个位置（组会 B6 会问）：
       - 第一层 Linear(input_dim, hidden_dim) 的【输出维数】= hidden_dim
       - 第二层 Linear(hidden_dim, output_dim) 的【输入维数】= hidden_dim
       所以改宽度时这两处必须一起改，否则第一层输出 8 维、第二层却按 16 维接收，
       两层"接不上"，会报形状错误。

    为什么中间要夹一个 ReLU（组会 A2）：
       - 如果只有 Linear 叠 Linear，等价于一个更大的 Linear，仍然只能表示线性关系。
       - ReLU 是"非线性"激活函数，把它夹在两层之间，网络才具备拟合非线性关系的能力。

    为什么最后一层不加激活函数（组会 A3）：
       - 我们的目标做过标准化，取值是任意实数（有正有负）。
       - ReLU 会把负数压成 0，若最后一层加 ReLU，就无法输出负的预测值。
       - 所以回归输出层保持"裸的" Linear，让输出范围不受限制。
    """
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def forward_shape_chain(model, X):
    """
    前向形状检查：把 nn.Sequential 逐层跑一遍，打印每一层之后的形状。

    用来验证宽度修改（如 hidden_dim=8）后，形状链是否正确：
      (B, 10) -> (B, 8) -> (B, 8) -> (B, 1)
    其中 ReLU 不改变形状（只是把负数变 0），所以 (B,8) 出现两次。
    """
    model.eval()
    with torch.no_grad():
        cur = X
        print(f"输入形状：{tuple(cur.shape)}")
        for layer in model:
            cur = layer(cur)
            print(f"  经过 {type(layer).__name__:<10} -> {tuple(cur.shape)}")
    return cur
