# -*- coding: utf-8 -*-
"""
optimizers.py —— 根据配置创建优化器

职责：接收模型参数、优化器名称和学习率，返回对应的优化器。

两种优化器的思路（组会 B3 会问）：
- SGD：每一步只按"当前梯度"更新，规则是  w = w - lr * grad。
- Adam：除了当前梯度，还会记录"历史梯度的一阶矩（类似动量）和二阶矩（类似方差）"，
  给每个参数自动调节一个合适的更新幅度；但整体上仍需要一个用户给定的 lr
  作为"基准步长"，所以 Adam 依然需要学习率。

注意：优化器本身只负责"拿到梯度后怎么更新参数"；
"梯度从哪来"是 training.py 里 loss.backward() 自动算的。
"""
import torch.optim as optim


def make_optimizer(model, name, lr):
    """
    按名称创建优化器。

    参数：
      model : 要训练的模型，内部会取 model.parameters() 作为待更新参数
      name  : "sgd" 或 "adam"（不区分大小写）
      lr    : 学习率

    本周两组具体配置（写在这里，方便组会查找）：
      - SGD  : lr=0.01, momentum=0, weight_decay=0
      - Adam : lr=0.001, weight_decay=0（其余参数用 PyTorch 默认值）
    """
    name = name.lower()
    if name == "sgd":
        # momentum=0、weight_decay=0：纯 SGD，不带动量、不带权重衰减正则
        return optim.SGD(model.parameters(), lr=lr, momentum=0, weight_decay=0)
    elif name == "adam":
        # weight_decay=0；其余（betas、eps 等）用默认值
        return optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    else:
        raise ValueError(f"未知优化器名称：{name}（只支持 sgd / adam）")
