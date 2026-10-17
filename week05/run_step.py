# -*- coding: utf-8 -*-
"""
run_step.py —— 实践一入口：把"一次梯度更新"和代码对应起来

用两个样本 x=[[1],[2]]、y=[[2],[4]]，完整走一遍：
    清梯度 -> 前向 -> 损失 -> 反向 -> 更新
并把每个阶段记录下来，用于和第二周手写梯度下降的知识对照。

运行（在 week05/ 目录下）：python run_step.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")   # 修 Windows 控制台中文乱码

import torch
import torch.nn as nn

import data
import models
import optimizers
import training
import reporting


def main():
    # 1. 准备数据：两个样本，形状 (2, 1)
    x, y = data.make_step_samples()

    # 2. 建模型：y = w*x + b，w、b 都初始化为 0
    model = models.make_zero_linear()

    # 3. 建优化器：SGD，学习率 0.1（不带动量、不带权重衰减）
    optimizer = optimizers.make_optimizer(model, "sgd", lr=0.1)

    # 4. 损失函数：均方误差 MSE
    loss_fn = nn.MSELoss()

    print("初始：w=0、b=0，预测都是 0，初始 MSE 应为 10\n")

    # 5. 完成一次更新，记录三个阶段
    trace = training.run_step_demo(model, x, y, loss_fn, optimizer)

    # 6. 保存记录到 results/step_trace.csv
    reporting.save_step_trace(trace)

    # 7. 用反传得到的梯度，人工核对更新公式：新参数 = 旧参数 - lr * 梯度
    print("\n核对（SGD 更新公式 w_new = w_old - lr * grad）：")
    print("  反传后梯度：weight_grad = -10，bias_grad = -6")
    print("  更新后参数：weight = 0 - 0.1*(-10) = 1.0，bias = 0 - 0.1*(-6) = 0.6")
    print("  （可对照上面 CSV 里'反传后'的梯度、'更新后'的参数验算）")


if __name__ == "__main__":
    main()
