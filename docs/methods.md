# 方法（Methods）

本文档总结本项目的模型与训练/评估口径，强调**实现一致性**与**可复现性**（不包含结果性结论）。本项目围绕纵向结构连接组（SC）序列建模，并可选引入形态学协同建模。

## 输入与对齐假设

- 基本输入为 SC 时间序列（每个被试至少 2 个时间点），文件组织与排序见 `docs/workflow.md`。
- 若使用 CLG-ODE，需要额外的形态学输入与对齐表 `subject_info_sc.csv`。
- 对齐优先以显式键（如 `scanid/subid/sesid`）完成，不依赖“行序一致”。

## 模型

### 1) VectorLSTM baseline

代码：`src/models/vector_lstm.py`，训练入口：`src/train.py`。

- 将每个时间点的 SC 矩阵向量化为上三角特征（见 `src/data/dataset.py` 与 `src/data/utils.py`）。
- 线性层 + ReLU 编码后输入 LSTM，最后解码回同维度特征，优化目标为预测序列的下一时刻特征（MSE）。

### 2) CLG-ODE（Coupled Latent Graph Neural ODE）

代码：`src/models/clg_ode.py`，训练入口：`src/train_clg_ode.py`，训练器：`src/engine/clg_trainer.py`。

高层结构：

- 以图编码器分别得到形态学与连接的潜变量分布（`mu/logvar`），通过重参数化采样初始潜变量。
- 将协变量（`sex/siteid`）嵌入后，与时间 `t` 一同作为 ODE 右端项的输入，得到潜空间动力学。
- 解码得到形态学预测 `x_hat` 与连接预测 `a_hat`（可选 top-k 稀疏化）。

损失项（见 `src/engine/clg_trainer.py`）：

- 重建误差：形态学 `x_hat` 与真实 `x` 的 masked MSE；连接 `a_hat` 与真实 `a` 的 masked MSE（使用上三角索引避免冗余）。
- KL 正则：对潜变量分布的 KL divergence（权重 `lambda_kl`）。
- 平滑正则：对潜变量二阶差分的平滑惩罚（权重 `lambda_smooth`）。
- 拓扑损失：占位接口 `src/engine/losses.py`（权重 `lambda_topo`；默认 0 不启用）。

## 训练与评估口径

### 数据划分

训练器以“被试”为分组单位：

- 外层：GroupShuffleSplit 划分 trainval/test（默认 test=0.2）。
- 内层：在 trainval 内部进行 5-fold GroupKFold 交叉验证并早停，选择最佳 fold 权重。

### 早停与输出

- 以验证集损失为准进行早停（`patience`）。
- 输出建议写入 `outputs/results/<model_name>/`，包含 `.pt` 权重与 JSON 摘要。

