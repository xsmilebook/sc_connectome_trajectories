# fold0 报告：mask 根因定位（pos_weight × λ_density + focal）（2026-01-20）

## 目的

在不再“盲扫 `lambda_density`”的前提下，定位两类关键问题：

1) **密度问题**：`sum(p_hat)/k` 是否长期 `>> 1`（过稠密）或 `<< 1`（过稀疏），以及是否主要由 `L_edge`（BCE `pos_weight`）驱动。
2) **位置问题**：即使密度能对齐，`p_hat` 的 top-k 位置是否正确（`precision@k/recall@k/AUPRC`），从而判断下一步是否需要更强的结构性约束（如 `L_deg` / 解耦 head）。

本轮只跑 `fold0`，用于快速判断方向。

## 实验设置

### 诊断指标（写入 `metrics.csv`）

每个 epoch 记录（train/val）：

- `mask_ratio = sum(p_hat)/k`（上三角、去对角；`k = #(A_true>0)` 同口径）
- `mask_p10/p50/p90`（`p_hat` 分位数）与 `mask_mean_p`
- `mask_precision_at_k` / `mask_recall_at_k`（按 `p_hat` 取 top-k 预测正边）
- `mask_auprc`（AUPRC，可选；本批实验均开启）

### 实验列表

- 1×诊断基线：`clg_maskdiag_fold0`（BCE、`pos_weight=5`、`lambda_density=0`）
- 9×网格：BCE `pos_weight ∈ {1,2,5}` × `lambda_density ∈ {0, 0.01, 0.05}`
- 1×focal 对照：`clg_focal_ld01`（`edge_loss=focal, gamma=2, alpha=0.25, lambda_density=0.01`）

所有 run 的输出目录均为：

- `outputs/results/clg_ode/runs/<run>/fold0/`

其中包含 `metrics.csv` 与 `test_sc_metrics.json`。

## 结果汇总

下表展示 **验证集 best-epoch 的 mask 指标**（与早停监控一致）与 **测试集 SC 指标**（`test_sc_metrics.json`）。

### 1) BCE 网格（pos_weight × λ_density）

| run | pos_weight | λ_density | epochs | best_val_mask_ratio | best_val_precision@k | best_val_AUPRC | test_sc_log_pearson_sparse | test_sc_log_pearson_topk | test_sc_log_mse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clg_pw1_ld0 | 1 | 0 | 61 | 0.669 | 0.0845 | 0.0761 | 0.6162 | 0.6165 | 0.1470 |
| clg_pw1_ld0.01 | 1 | 0.01 | 76 | 0.653 | 0.0871 | 0.0786 | 0.6177 | 0.6173 | 0.1496 |
| clg_pw1_ld0.05 | 1 | 0.05 | 101 | 0.556 | 0.0870 | 0.0786 | **0.6217** | **0.6198** | 0.1572 |
| clg_pw2_ld0 | 2 | 0 | 66 | 0.792 | 0.0854 | 0.0770 | 0.6058 | 0.6088 | 0.1350 |
| clg_pw2_ld0.01 | 2 | 0.01 | 79 | 0.760 | 0.0881 | 0.0796 | 0.6058 | 0.6084 | 0.1360 |
| clg_pw2_ld0.05 | 2 | 0.05 | 98 | 0.739 | 0.0863 | 0.0779 | 0.6108 | 0.6118 | 0.1420 |
| clg_pw5_ld0 | 5 | 0 | 66 | 1.208 | 0.0788 | 0.0712 | 0.5878 | 0.5960 | 0.1239 |
| clg_pw5_ld0.01 | 5 | 0.01 | 78 | 1.075 | 0.0843 | 0.0760 | 0.5916 | 0.5983 | 0.1250 |
| clg_pw5_ld0.05 | 5 | 0.05 | 139 | 0.870 | 0.0877 | 0.0790 | 0.5951 | 0.5997 | 0.1291 |

### 2) 诊断基线与 focal 对照

| run | edge_loss | pos_weight | λ_density | epochs | best_val_mask_ratio | best_val_precision@k | best_val_AUPRC | test_sc_log_pearson_sparse | test_sc_log_pearson_topk | test_sc_log_mse |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clg_maskdiag_fold0 | bce | 5 | 0 | 35 | 1.033 | 0.0805 | 0.0724 | 0.5906 | 0.5982 | 0.1236 |
| clg_focal_ld01 | focal | - | 0.01 | 76 | 0.841 | 0.0803 | 0.0726 | 0.6040 | 0.6089 | 0.1321 |

## 结论（根因定位）

1) **`pos_weight` 确实在驱动密度**：从 `best_val_mask_ratio` 可以看到，`pos_weight=5` 更容易出现 `ratio ≥ 1`（偏稠密），而 `pos_weight=1/2` 基本都落在 `ratio < 1`（偏稀疏）。因此“`sum(p_hat)` 压不下来”的现象，**很可能与 `L_edge` 的正样本权重设置有关**（或至少存在显著对抗）。

2) **`lambda_density` 在 `pos_weight=5` 时能把 ratio 从 1.21 拉到 0.87（趋近 1）**，说明 density 不是完全无效；但在 `pos_weight=1` 时反而进一步把 ratio 拉低（更稀疏），提示当前密度项在不同 BCE 驱动力下会产生不同平衡点。

3) **mask 位置质量非常差且几乎不随配置变化**：
   - `precision@k ≈ 0.08–0.088`，`AUPRC ≈ 0.07–0.079`，整体偏低且在 9 组网格间变化很小。
   - 这意味着：即使密度（边数）能被调到接近，`p_hat` 的 top-k 边位置仍然大量偏离真值；**仅靠全局边数约束（density）很难学到“非 0 区域”**。

4) **focal 作为替代 BCE 权重的手段有一定价值**：在本配置下 focal 将 `ratio` 调整到 0.84，并将 `test_sc_log_pearson_sparse/topk` 提升到约 0.604/0.609（优于 `pos_weight=5` 系列），但对 `precision@k/AUPRC` 并没有明显改善（仍 ~0.08/~0.073）。

5) **任务口径间存在权衡**：`pos_weight=1` 系列在 `test_sc_log_pearson_sparse/topk` 最好，但 `sc_log_mse` 明显变差；`pos_weight=5` 系列 `mse` 最好但 sparse/topk 较差。这符合“稀疏 mask 与权重拟合存在冲突/拉扯”的预期。

## 下一步建议（按信息增益排序）

1) **在确认“位置不对”后，优先上结构性约束而不是继续扫 λ_density**：
   - 加 `L_deg`（期望度匹配）比全局 density 更能约束“概率质量落在哪些节点周围”。

2) 若仍要让 density 更贴近目标密度：
   - 将 `pos_weight` 与 `lambda_density` 联合调参是必要的（本轮已经证明强耦合）。
   - focal 可作为更稳的替代，但需要配合结构性约束才能解决“位置不对”。

3) 若后续诊断显示 `p_hat` 与 `w_hat` 的共享打分导致冲突明显：
   - 考虑将 `p_head` 与 `w_head` 解耦（共享 embedding，但不共享最后一层打分）。
   - 或在两阶段训练 Phase-2 中 freeze 住与权重相关的部分，仅训练 `p_head`（再配 `L_deg`/density/zero），避免破坏 Phase-1 已学到的权重/动力学。

