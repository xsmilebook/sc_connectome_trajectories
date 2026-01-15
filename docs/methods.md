# 方法（Methods）

本文档总结本项目的模型与训练/评估口径，强调**实现一致性**与**可复现性**（不包含结果性结论）。本项目围绕纵向结构连接组（SC）序列建模，并可选引入形态学协同建模。

## 输入与对齐假设

- 基本输入为 SC 时间序列（每个被试至少 **1** 个时间点），文件组织与排序见 `docs/workflow.md`。
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
- SC 预处理：对称化并清零对角线，输入与权重回归在 `log(1 + A)` 域进行，不做额外 hard sparsify。
- 将协变量（`age0/sex/siteid` + SC 全局强度标量 `s`（原始权重域计算）+ 可选正边平均强度 `s_mean` + 拓扑摘要向量 `μ(A)`）编码后，与时间 `t` 一同作为 ODE 右端项的输入，得到耦合潜空间动力学。
- 解码得到形态学预测 `x_hat` 与连接预测 `a_hat`；连接使用连续权重回归（`Softplus(z z^T)`），不做额外 hard sparsify（禁用 Top‑K/阈值截断）。
- 训练采样采用多起点预测：70% 选择相邻访视对，30% 从所有 `i<j` 组合中均匀采样。
- 默认启用轻量去噪增强（train-only）：形态学加入高斯噪声（`morph_noise_sigma=0.05`），SC 对正边做少量 dropout（`sc_pos_edge_drop_prob=0.02`），目标仍为重建“干净”输入。

损失项（见 `src/engine/clg_trainer.py`）：

- 重建误差：
  - 形态学：按 (ROI, metric) 列做训练集统计量的 Z-score 后，对 `x_hat` 与真实 `x` 计算 MSE。
  - 连接：为避免零边主导，使用两项式损失：存在性加权 BCE（pos:neg=5:1）+ 正边权重的 log 域 Huber；`L = L_edge + λ_w * L_weight`（默认 `λ_w=1.0`；上三角计算避免冗余）。
  - 连接输出共享同一个内积分数 `s_ij=z_i^T z_j`，存在性与强度分别通过可学习标定得到：`p_ij=σ(α(s_ij-δ))`，`ŵ_ij=softplus(γ s_ij+β)`（默认初始化 `α=10, δ=0, γ=1, β=0`）。
- KL 正则：对潜变量分布的 KL divergence（权重 `lambda_kl`）。
- 拓扑：仅作为条件输入与评估解释项，不参与训练 loss（阈值分位数的 Euler characteristic curve，见 `docs/reports/implementation_specification.md`）。
- Tiered 训练目标（按每个被试可用时间点数自动触发）：
  - `L_manifold`：Tier 3/2/1 均可用（重建 + 多起点预测重建 + 去噪重建）。
  - `L_vel`：Tier 2/1 使用（潜空间速度场一致性/监督）。
  - `L_acc`：Tier 1 使用（潜空间加速度/非线性项）。
  - 默认权重与 warmup：`λ_manifold=1.0`，`λ_vel=0.2`，`λ_acc=0.1`；前 10 个 epoch 仅优化 `L_manifold`，第 11–20 个 epoch 启用 `L_vel`，第 21 个 epoch 起启用 `L_acc`。

## 训练与评估口径

### 数据划分

训练器以“被试”为分组单位：

- 外层：GroupShuffleSplit 划分 trainval/test（默认 test=0.2）。
- 内层：在 trainval 内部进行 5-fold GroupKFold 交叉验证并早停，选择最佳 fold 权重。

### 早停与输出

- 以验证集损失为准进行早停（`patience`）。
- 输出建议写入 `outputs/results/<model_name>/`，包含 `.pt` 权重与 JSON 摘要。
- CLG-ODE 默认在 `--results_dir/runs/<timestamp>_job<jobid>/` 创建独立运行目录，保存 `args.json`、`run_meta.json`、`metrics.csv`，便于追溯与横向对比。
- 测试集评估（当前仅评估 SC）在 `test` 上计算并保存：
  - `sc_log_mse` / `sc_log_mae` / `sc_log_pearson`：预测与真实的 `log(1 + A)` 上三角向量指标。
  - `ecc_l2` / `ecc_pearson`：预测与真实的 Euler characteristic curve 相似性指标（基于 `log(1 + A)` 域，预测矩阵先按真实正边数量做 top-k 稀疏化）。
  - 当被试只有 1 个时间点时，测试指标基于同一时点的重建输出；当被试有 ≥2 个时间点时，使用固定的 `t0→t1` 预测输出（保证可重复）。
