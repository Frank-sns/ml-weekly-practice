# 第五周：MLP 与 PyTorch 训练基础

本周把前面几周的数学与算法知识对应到 PyTorch 代码：搞清前向、损失、反向传播、参数更新分别在哪完成，
再用一个小型 MLP 比较 SGD 和 Adam 两套配置。项目按组件拆成多个 `.py` 文件，两个入口只负责连接流程。

## 数据来源

- scikit-learn 内置 `load_diabetes(as_frame=True, scaled=False)`，10 个特征 + 1 个目标 `target`
- 实践一另用两个手工样本（`x=[[1],[2]]`、`y=[[2],[4]]`），不划分数据集

## 主要文件

| 文件 | 职责 |
|---|---|
| `data.py` | 数据读取、60/20/20 划分、X 与 y 标准化、张量与 DataLoader |
| `models.py` | 实践一的零初始化线性模型；实践二的 MLP（隐藏层宽度可调） |
| `optimizers.py` | 根据名称创建 SGD 或 Adam |
| `training.py` | 一次更新、一批更新、训练循环、最佳轮次记录、基线拟合 |
| `evaluation.py` | 预测与 MSE/MAE/RMSE，把标准化预测恢复到原始单位，不更新参数 |
| `reporting.py` | 保存 CSV、绘制并保存图形 |
| `run_step.py` | 实践一入口：一次梯度更新演示 |
| `run_mlp.py` | 实践二入口：基线与两组 MLP 实验 |

## 环境与运行

- Python 版本：**3.14.5**
- PyTorch：**2.14.1（CPU 版）**

安装依赖（先装 CPU 版 torch，体积小；不要用默认 pip 拉取 CUDA 大包）：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

两个运行命令（都在 `week05/` 目录下执行）：

```bash
python run_step.py     # 实践一：一次梯度更新
python run_mlp.py      # 实践二：训练 MLP + 基线对照
```

## 主要实验设置

- 数据划分 60% / 20% / 20%（265 / 88 / 89），`random_state=42`。
- 特征 X 只用训练集 `StandardScaler`；目标 y 用训练集的 `y_mean`、`y_std`（`np.std(ddof=0)`）标准化，评价时恢复到原始单位。
- MLP：`Linear(10, 16) → ReLU → Linear(16, 1)`，`nn.MSELoss()`，200 轮，`batch_size=32`，`drop_last=False`。
- 两组优化器：SGD `lr=0.01`（momentum=0、weight_decay=0）；Adam `lr=0.001`（weight_decay=0）。
- `torch.manual_seed(42)` 建初始网络，`copy.deepcopy` 得到两组相同初始参数；两组各自新建 DataLoader，用 `torch.Generator().manual_seed(42)`、`num_workers=0`。
- 每轮结束后用完整训练/验证集重新评价 MSE，记录验证 MSE 最小的轮次与参数（相同取较早一轮），训练结束后恢复最佳参数。
- 按验证 RMSE 在两组 MLP 中选一组（相同选 SGD）；只对均值基线、线性回归和选定 MLP 算测试指标。

## 结果位置与主要结论

结果位于 `week05/results/`：

| 文件 | 内容 |
|---|---|
| `step_trace.csv` | 实践一三个阶段（清梯度后/反传后/更新后）的权重、偏置、梯度、损失 |
| `history.csv` | 两组各 200 行的逐轮训练/验证 MSE（标准化目标） |
| `metrics.csv` | 四组模型的 MAE/RMSE（原始单位）与选定标记 |
| `loss_curves.png` | SGD、Adam 双面板训练/验证 MSE 曲线，标出最佳轮次 |
| `rmse_comparison.png` | 四组模型训练/验证 RMSE 柱状图 |

本次运行的主要结论：

- **实践一**：反向传播得到梯度 w=-10、b=-6，更新后 w=1.0、b=0.6，符合 `新参数 = 旧参数 - 0.1 × 梯度`；损失从 10 降到 1.06。
- **实践二**：Adam 最佳验证 MSE 0.4158（第 88 轮）略优于 SGD 的 0.4259（第 167 轮），依据验证 RMSE（Adam 49.86 < SGD 50.47）选定 **Adam**。
  两组 MLP 都明显好于均值基线（验证 RMSE 76.16），并接近线性回归（49.15）。
  （结论按实际结果讨论即可，不要求 MLP 超过线性回归，也不要求 Adam 超过 SGD。）

## 隐藏层修改检查

`run_mlp.py` 末尾有一段 `hidden_dim=8` 的前向形状检查（只检查、不重新训练）。
修改位置在 `models.py` 的 `make_mlp` 里，`hidden_dim` 同时控制两层：
`Linear(10, hidden_dim)` 的**输出维数** 和 `Linear(hidden_dim, 1)` 的**输入维数**，两处必须一起改。

检查输出形状：

```text
(32, 10) -> (32, 8) -> (32, 8) -> (32, 1)
```

其中 ReLU 不改变形状（只是把负数变成 0），所以 (32, 8) 出现两次。
