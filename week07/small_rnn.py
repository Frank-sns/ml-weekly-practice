# -*- coding: utf-8 -*-
"""
small_rnn.py —— 实践一的核心：把一个小 RNN 沿时间展开，观察前向与反向

这个文件只用常数和 5 个标量，故意做得很小，目的是能一眼看清：

    前向：z[1..5] 依次进入同一个计算单元，每一步都复用 wx、wh、b 三个参数
    反向：只在最后一步算一次损失，梯度沿着展开的链条一路传回第 1 步

记号（与任务书一致）：
    h[k] = tanh(wx * z[k] + wh * h[k-1] + b),  k = 1..5,  h[0] = 0
    预测值 = 0.8 * h[5] + 0.1
    损失   = (预测值 - z[6])^2

注意两点（组会常问）：
1. 五步用的是【同一组】参数，程序里只有 3 个 torch 张量，不是 15 个。
2. 记录用的 float(h) 只是"抄下来的数值"，绝不能拿它再接回计算，
   否则计算图会在那一刀断开，梯度就传不回前面几步了。
"""
import numpy as np
import torch

# ---- 演示用的固定设置 ----
N_DEMO = 6                      # 取最前面连续 6 小时的油温 u[1] ~ u[6]
N_INPUT = 5                     # 前 5 个作为输入，第 6 个作为目标
WX_INIT = 0.6                   # 输入项权重
WH_INIT = 0.5                   # 历史项（循环）权重
B_INIT = 0.0                    # 偏置
OUT_W = 0.8                     # 输出变换：0.8 * h[5] + 0.1（固定常数，不参与求导）
OUT_B = 0.1
ORIGINAL_ORDER = [1, 2, 3, 4, 5]        # 原顺序：z1 → z2 → ... → z5
PERMUTED_ORDER = [1, 4, 3, 2, 5]        # 顺序对照：z1 → z4 → z3 → z2 → z5


def build_demo_data(df):
    """
    取最前面 6 小时，构造演示数据。

    u[k] 是原始油温；统一变换 z[k] = (u[k] - u[1]) / 10，
    减去的基准【始终是原始第一条油温】，不是逐差、也不是滚动基准。

    返回：
      u     : (6,) 原始油温
      z     : (6,) 变换后的值（前 5 个是输入，第 6 个是目标）
      times : (6,) 对应时间
      index : (6,) 原始数据中的下标（从 0 开始）
    """
    u = df["OT"].iloc[:N_DEMO].to_numpy(dtype=float)
    times = df["date"].iloc[:N_DEMO].to_numpy()
    index = np.arange(N_DEMO)
    z = (u - u[0]) / 10.0
    return u, z, times, index


def forward_case(z, order, target_z, requires_grad):
    """
    按给定顺序把 5 个输入喂进同一个 RNN 单元，做一次前向计算。

    参数
    ----
    z            : (6,) 变换后的 6 个值
    order        : 输入顺序（1 基的步号），如 [1,2,3,4,5] 或 [1,4,3,2,5]
    target_z     : 预测目标 z[6]
    requires_grad: 是否让输入张量可求导（原顺序案例需要，对照案例不需要）

    返回
    ----
    pred, loss, h_final, rows, inputs_tensor, (wx, wh, b)

    rows 里每一步记录：
      step, source_index, source_time, z, h_prev, input_term, memory_term, h
    这些数值只用于保存到 CSV，不参与后续计算。
    """
    # 把 5 个输入按 order 重新排好，做成一个【可求导】的一维张量。
    # 之所以要 requires_grad=True：任务要求观察损失对每个输入分量的梯度。
    inputs = torch.tensor(
        [z[s - 1] for s in order],
        dtype=torch.float64,
        requires_grad=requires_grad,
    )

    # 三个标量参数，五步共用同一组；都启用梯度计算
    wx = torch.tensor(WX_INIT, dtype=torch.float64, requires_grad=True)
    wh = torch.tensor(WH_INIT, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(B_INIT, dtype=torch.float64, requires_grad=True)

    h = torch.zeros((), dtype=torch.float64)   # h[0] = 0
    rows = []

    for k, source_step in enumerate(order, start=1):
        z_k = inputs[k - 1]                     # 当前时刻的输入（张量，带计算图）
        h_prev = h                             # 上一状态（张量，带计算图）

        input_term = wx * z_k                   # 输入项：当前观测经 wx 进入
        memory_term = wh * h_prev               # 历史项：上一步状态经 wh 进入
        h = torch.tanh(input_term + memory_term + b)   # 新状态

        # ⚠️ 这里 float(...) 只是把数值"抄一份"记录下来。
        # 下一轮循环用的是上面的张量 h，不是这个 float，否则梯度会断。
        rows.append({
            "step": k,
            "source_step": source_step,
            "z": float(z_k.detach()),
            "h_prev": float(h_prev.detach()),
            "input_term": float(input_term.detach()),
            "memory_term": float(memory_term.detach()),
            "h": float(h.detach()),
        })

    pred = OUT_W * h + OUT_B                    # 只在最后一步生成预测
    loss = (pred - target_z) ** 2               # 也只在最后一步算一次损失
    return pred, loss, h, rows, inputs, (wx, wh, b)


def run_original(z, target_z, times, index):
    """
    原顺序 [1,2,3,4,5]：做前向 + 反向，观察参数梯度和每个输入的梯度。

    返回 (rows, info)，info 里含末端预测/目标/损失/三个参数梯度/输入梯度。
    """
    pred, loss, h_final, rows, inputs, (wx, wh, b) = forward_case(
        z, ORIGINAL_ORDER, target_z, requires_grad=True)

    # 对同一次损失调用反向传播。只算梯度，不调用优化器、不更新参数。
    loss.backward()

    # 把输入梯度抄进每一步的记录（顺序与 order 一致）
    for row in rows:
        row["input_grad"] = float(inputs.grad[row["step"] - 1])

    # 补上原始下标与时间：order 里的第 s 步就是原始第 s-1 个下标
    for row in rows:
        src = row["source_step"] - 1
        row["source_index"] = int(index[src])
        row["source_time"] = str(times[src])

    info = {
        "pred": float(pred.detach()),
        "target": float(target_z),
        "loss": float(loss.detach()),
        "h_final": float(h_final.detach()),
        "wx_grad": float(wx.grad),
        "wh_grad": float(wh.grad),
        "b_grad": float(b.grad),
        "input_grads": [float(g) for g in inputs.grad],
    }
    return rows, info


def run_permuted(z, target_z, times, index):
    """
    顺序对照 [1,4,3,2,5]：只做前向，不反向传播。

    数值集合与原顺序完全相同，第 1 步和第 5 步的输入也相同，
    但第 2 步开始输入不同 → 状态不同 → 末端预测不同。
    这一组用来证明"最后的状态不仅取决于最后一个输入"。
    """
    pred, loss, h_final, rows, _inputs, _params = forward_case(
        z, PERMUTED_ORDER, target_z, requires_grad=False)

    for row in rows:
        src = row["source_step"] - 1
        row["source_index"] = int(index[src])
        row["source_time"] = str(times[src])
        row["input_grad"] = None        # 这一组不做反向，梯度列留空

    info = {
        "pred": float(pred.detach()),
        "target": float(target_z),
        "loss": float(loss.detach()),
        "h_final": float(h_final.detach()),
    }
    return rows, info


def manual_check(rows):
    """
    用公式独立手算第 2 步，与程序输出核对。

    h[1] = tanh(wx*z[1] + wh*h[0] + b)
    h[2] = tanh(wx*z[2] + wh*h[1] + b)

    返回 (h2_manual, h2_program, 差值)。
    """
    h1 = np.tanh(WX_INIT * rows[0]["z"] + WH_INIT * 0.0 + B_INIT)
    h2_manual = np.tanh(WX_INIT * rows[1]["z"] + WH_INIT * h1 + B_INIT)
    h2_program = rows[1]["h"]
    return float(h2_manual), float(h2_program), abs(h2_manual - h2_program)


def linear_chain_coefficients(order=(1, 2)):
    """
    "去掉 tanh 之后"的展开，用来说明循环连接怎样带来历史依赖。

    去掉 tanh 且 h[0]=0、b=0 时：
        h[1] = wx * z[1]
        h[2] = wx * z[2] + wh * h[1] = wx * z[2] + wh*wx * z[1]
    所以 z[1] 的系数是 wh*wx，z[2] 的系数是 wx。

    判断的是【参数固定后，预测与输入的关系】：这里只是输入的一次函数，
    不会出现 z[1]*z[2] 这样的输入乘积。参数之间相乘（wh*wx）是常数，
    并不等于对输入具有非线性——这也是 MLP 的层与层之间需要激活函数的原因。
    """
    return {"z1_coef": WH_INIT * WX_INIT, "z2_coef": WX_INIT}


def gradient_decay_example():
    """
    "梯度沿路径连乘"的小算例（帮助理解长序列为什么更难学）。

    每经过一步乘 0.5：2 步后 0.25，5 步后 0.03125；
    反复乘 1.5：5 步后约 7.59，可能越来越大。
    这只是示意，真实 RNN 的局部导数还取决于权重和 tanh 的导数。
    """
    return {
        "0.5^2": 0.5 ** 2,
        "0.5^5": 0.5 ** 5,
        "0.5^10": 0.5 ** 10,
        "1.5^5": 1.5 ** 5,
    }
