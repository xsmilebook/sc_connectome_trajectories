# 方法（Methods）

本文档总结本项目的模型与训练/评估口径，强调**实现一致性**与**可复现性**（不包含结果性结论）。本项目围绕纵向结构连接组（SC）序列建模，并可选引入形态学协同建模。

## 输入与对齐假设

- 基本输入为 SC 时间序列（每个被试至少 2 个时间点），文件组织与排序见 `docs/workflow.md`。
- 若使用 CLG-ODE，需要额外的形态学输入与对齐表 `subject_info_sc.csv`。
- 对齐优先以显式键（如 `scanid/subid/sesid`）完成，不依赖“行序一致”。

## 模型

### 1) VectorLSTM baseline

代码：`src/models/vector_lstm.py`，训练入口：`scripts/train.py`（推荐 `python -m scripts.train`）。

- 将每个时间点的 SC 矩阵向量化为上三角特征（见 `src/data/dataset.py` 与 `src/data/utils.py`）。
- 线性层 + ReLU 编码后输入 LSTM，最后解码回同维度特征，优化目标为预测序列的下一时刻特征（MSE）。

### 2) CLG-ODE（Coupled Latent Graph Neural ODE）

代码：`src/models/clg_ode.py`，训练入口：`scripts/train_clg_ode.py`（推荐 `python -m scripts.train_clg_ode`），训练器：`src/engine/clg_trainer.py`。

高层结构：

- 以图编码器分别得到形态学与连接的潜变量分布（`mu/logvar`），通过重参数化采样初始潜变量。
- 使用相对时间 `Δt = age - age0` 作为 ODE 的积分变量（每次预测从 `t0=0` 开始），并将起点绝对年龄 `age0` 作为协变量输入动力学函数以避免重复建模。
- 将协变量（`age0/sex/siteid` + SC 全局强度标量 `s`（原始权重域计算）+ 可选 `s_mean` + 拓扑摘要向量 `μ(A)`）编码后，与时间 `t` 一同作为 ODE 右端项的输入，得到耦合潜空间动力学。
- 解码得到形态学预测 `x_hat` 与连接预测 `a_hat`；连接使用连续权重回归（`Softplus(z z^T)`），不做额外 hard sparsify（禁用 Top‑K/阈值截断）。

损失项（见 `src/engine/clg_trainer.py`）：

- 重建误差：
  - 形态学：按 (ROI, metric) 列做训练集统计量的 Z-score 后，对 `x_hat` 与真实 `x` 计算 masked MSE/Huber。
  - 连接：为避免零边主导，使用两项式损失：存在性加权 BCE（pos:neg=5:1）+ 正边权重的 log 域 Huber；`L = L_edge + λ_w * L_weight`（默认 `λ_w=1.0`；上三角计算避免冗余）。
  - 连接输出共享同一个内积分数 `s_ij=z_i^T z_j`，存在性与强度分别通过可学习标定得到：`p_ij=σ(α(s_ij-δ))`，`ŵ_ij=softplus(γ s_ij+β)`（默认初始化 `α=10, δ=0, γ=1, β=0`）。
- KL 正则：对潜变量分布的 KL divergence（权重 `lambda_kl`）。
- 平滑正则：对潜变量二阶差分的平滑惩罚（权重 `lambda_smooth`）。
- 拓扑：本版本不作为训练损失（占位接口 `src/engine/losses.py` 保留；默认 `lambda_topo=0`），仅将拓扑摘要 `μ(A)` 作为条件输入与评估解释项（阈值分位数的 Euler characteristic curve，见 `docs/reports/implementation_specification.md`）。

## 训练与评估口径

### 数据划分

训练器以“被试”为分组单位：

- 外层：GroupShuffleSplit 划分 trainval/test（默认 test=0.2）。
- 内层：在 trainval 内部进行 5-fold GroupKFold 交叉验证并早停，选择最佳 fold 权重。

### 早停与输出

- 以验证集损失为准进行早停（`patience`）。
- 输出建议写入 `outputs/results/<model_name>/`，包含 `.pt` 权重与 JSON 摘要。
