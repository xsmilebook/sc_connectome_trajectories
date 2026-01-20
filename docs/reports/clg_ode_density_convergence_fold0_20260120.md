# fold0 报告：density 约束收敛性网格 + 两阶段训练（2026-01-20）

## 目的

针对此前 density 约束“刚起效就被 early stop 掐断”的问题，本轮实验按如下策略重新设计并运行：

- **更长训练**：`MAX_EPOCHS=100`，`PATIENCE=25`
- **更温和的 λ_density 网格**：`{0.002, 0.005, 0.01, 0.02}`
- **短 warmup + 慢 ramp**：`DENSITY_WARMUP_EPOCHS ∈ {2,5}`，`DENSITY_RAMP_EPOCHS=20`
- **早停监控口径对齐稀疏评估**：`EARLY_STOP_METRIC=val_sc_log_pearson_sparse`（并设置 `VAL_SC_EVAL_EVERY=1`）
- 额外对比：**两阶段训练（Phase-1 → Phase-2）**，Phase-2 用更小学习率 + 开启 density/zero 约束，且对比两种监控口径（sparse pearson vs composite）。

范围：仅 `fold0`（用于快速判断趋势与可收敛性）。

## 实验列表（runs）

8 组网格（fold0）：

- `clg_density_w2_ld002` / `clg_density_w2_ld005` / `clg_density_w2_ld01` / `clg_density_w2_ld02`
- `clg_density_w5_ld002` / `clg_density_w5_ld005` / `clg_density_w5_ld01` / `clg_density_w5_ld02`

两阶段训练（fold0）：

- `clg_density_twostage_a_p1` → `clg_density_twostage_a_p2`（Phase-2 monitor=`val_sc_log_pearson_sparse`）
- `clg_density_twostage_b_p1` → `clg_density_twostage_b_p2`（Phase-2 monitor=`monitor_mse_plus_density`）

结果文件位置：

- `outputs/results/clg_ode/runs/<run>/fold0/metrics.csv`
- `outputs/results/clg_ode/runs/<run>/fold0/test_sc_metrics.json`

## 结果汇总（测试集）

下表从每个 run 的 `test_sc_metrics.json` 读取测试集指标；`val_density_last` 来自该 run `metrics.csv` 的最后一行（训练阶段的 density loss，**不是测试指标**）。

说明：

- `val_density_last` 定义为 `((sum(p_hat)-k)/k)^2` 的均值（上三角），因此 `sqrt(val_density_last)` 约等于 “期望边数与真值边数的相对误差”。
- 由于当前日志未记录 `sum(p_hat)/k` 的符号与分位数分布，`val_density_last` 只能用于判断“偏差是否接近 0”，不能直接判断“过稠密还是过稀疏”。

| run | epochs | monitor_metric | best_epoch | sc_log_pearson_sparse | sc_log_pearson_topk | sc_log_mse | ecc_l2 | val_density_last | sqrt(val_density_last) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| clg_density_w2_ld02 | 49 | val_sc_log_pearson_sparse | 24 | 0.592764 | 0.598691 | 0.126110 | 13385.4 | 4.452256 | 2.110 |
| clg_density_w5_ld005 | 29 | val_sc_log_pearson_sparse | 4 | 0.592325 | 0.599206 | 0.125185 | 12229.2 | 4.704724 | 2.169 |
| clg_density_w5_ld02 | 54 | val_sc_log_pearson_sparse | 29 | 0.591085 | 0.597494 | 0.125475 | 13106.7 | 4.678601 | 2.163 |
| clg_density_w2_ld005 | 28 | val_sc_log_pearson_sparse | 3 | 0.590879 | 0.598138 | 0.124816 | 12268.5 | 5.111752 | 2.261 |
| clg_density_twostage_b_p2 | 27 | monitor_mse_plus_density | 2 | 0.590002 | 0.597513 | 0.124448 | 12211.5 | 5.179513 | 2.276 |
| clg_density_twostage_a_p2 | 56 | val_sc_log_pearson_sparse | 31 | 0.589611 | 0.596964 | 0.124914 | 12530.4 | 5.176046 | 2.275 |
| clg_density_w5_ld01 | 28 | val_sc_log_pearson_sparse | 3 | 0.589348 | 0.597053 | 0.124031 | 12132.0 | 5.400250 | 2.324 |
| clg_density_w5_ld002 | 28 | val_sc_log_pearson_sparse | 3 | 0.589291 | 0.597039 | 0.123610 | 11997.3 | 5.434477 | 2.331 |
| clg_density_w2_ld01 | 31 | val_sc_log_pearson_sparse | 6 | 0.588843 | 0.596730 | 0.124475 | 12054.1 | 5.450891 | 2.335 |
| clg_density_w2_ld002 | 30 | val_sc_log_pearson_sparse | 5 | 0.587682 | 0.595779 | 0.123554 | 12095.9 | 5.652274 | 2.377 |

补充（两阶段 Phase-1）：

- `clg_density_twostage_a_p1` / `clg_density_twostage_b_p1` 未启用 density（`lambda_density=0`），因此 `val_density` 记录为 0，仅用于提供 Phase-2 的初始化模型。

## 结论（针对本轮“density 可收敛”目标）

1) **训练确实更长了**（多数 run 跑到 28–56 epoch，最长 54 epoch），但 **`val_density_last` 仍远离 0**（约 4.45–5.65），表示“期望边数与真值边数”的相对偏差仍很大（约 2.1–2.4 的量级）。

2) 在当前设置下，**稀疏口径指标（topk/sparse pearson）提升幅度很小**，且不同 `lambda_density/warmup` 的排序并不稳定；两阶段训练的 Phase-2 也未表现出明确优势。

3) 从“可收敛”角度看，本轮结果更像是：**density 项被加入了，但没有被有效优化到一个较小的区间**。这与上一轮（B/D 仅 16 epoch，density 刚起效就早停）相比，问题从“训练太短”转变为“密度约束仍未真正收敛”。

## 下一步建议（最小改动优先）

为避免继续“盲扫 λ_density”，下一轮建议优先补齐诊断，否则很难判断 density 不收敛的根因：

1) **记录密度匹配的可解释量**
   - `ratio = sum(p_hat)/k`（上三角）及其分位数（p10/p50/p90）
   - 这能直接回答：当前是“预测过稠密”还是“塌到过稀疏”。

2) **记录 mask 质量而非只看权重相关**
   - `precision@k / recall@k`（对 `A_true>0`）
   - （若可行）AUPRC

3) 若确认 `sum(p_hat) >> k` 且 density 始终压不下去：
   - 可以在当前早停口径（sparse pearson）下，**反向尝试更大的 `lambda_density`**（例如 0.05/0.1）并配合更慢 ramp（或更长 patience），验证是否仅是“权重太小导致不起作用”。
   - 或降低 `L_edge` 的正样本权重（pos_weight=5.0）/改 focal，减少“过预测正边”的驱动力（需单独对照）。

4) 若确认“总边数能对齐，但位置不对”：
   - 再考虑升级为期望度匹配（`L_deg`）以约束“概率质量落在哪些节点/模块附近”。

