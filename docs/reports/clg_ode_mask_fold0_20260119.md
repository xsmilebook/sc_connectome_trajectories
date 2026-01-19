# fold0 测试报告：mask 学习与稀疏对齐（2026-01-19）

## 目的

验证在“最终预测矩阵使用带 soft mask 的期望权重”定义下（`A_pred = p_hat * w_hat`），引入以下正则是否能稳定提升稀疏口径指标：

- **密度/边数约束**：对 `p_hat` 施加期望边数接近真值的约束（`lambda_density` + warmup/ramp）。
- **零边幅值惩罚**：对 `A_pred` 在真零边位置施加轻量幅值惩罚（`lambda_zero_log` + warmup/ramp）。

本次仅做 **fold0** 快速对照（短训练），用于判定方向与量级，而非最终模型选择。

## 提交方式（用户执行）

批量提交脚本：

```bash
bash scripts/submit_clg_ode_mask_fold0_batch.sh
```

对应的 4 个子实验脚本见 `scripts/submit_clg_ode_mask_fold0_{a,b,c,d}.sh`。

## 结果汇总（test_sc_metrics.json）

评估文件位置：

- `outputs/results/clg_ode/runs/<run>/fold0/test_sc_metrics.json`

其中 `<run>` 为：
`clg_ode_mask_fold0_a_control` / `clg_ode_mask_fold0_b_density` / `clg_ode_mask_fold0_c_zero` / `clg_ode_mask_fold0_d_density_zero`。

| 方案 | run | epochs | density_factor(last) | zero_factor(last) | train_density(last) | train_zero_log(last) | sc_log_mse | sc_log_mae | sc_log_pearson | sc_log_pearson_pos | sc_log_pearson_topk | sc_log_pearson_sparse | ecc_l2 | ecc_pearson |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A control | clg_ode_mask_fold0_a_control | 40 | 1.0 | 1.0 | 0.0 | 0.0 | 0.124016 | 0.148261 | 0.863571 | 0.824677 | 0.595877 | 0.587981 | 12159.1 | 0.999100 |
| B density | clg_ode_mask_fold0_b_density | 16 | 0.3 | 1.0 | 4.668728 | 0.0 | 0.125125 | 0.147315 | 0.863593 | 0.825867 | 0.600252 | 0.592787 | 11257.8 | 0.999132 |
| C zero | clg_ode_mask_fold0_c_zero | 40 | 1.0 | 1.0 | 0.0 | 0.020951 | 0.123424 | 0.148321 | 0.863766 | 0.824764 | 0.594844 | 0.586505 | 12035.1 | 0.999108 |
| D density+zero | clg_ode_mask_fold0_d_density_zero | 16 | 0.3 | 0.3 | 4.662680 | 0.019297 | 0.126019 | 0.148624 | 0.863340 | 0.825263 | 0.599458 | 0.592125 | 11759.8 | 0.999096 |

说明：

- `density_factor(last)` / `zero_factor(last)` 为 warmup/ramp 后在 **最后一个 epoch** 的启用系数（0 表示未启用；1 表示完全启用）。
- B/D 两个带 density 的实验在 warmup 后不久即早停，实际仅经历了较短的 density 约束阶段。

## 结论（针对这次 fold0 快测）

1) **`A_pred = p_hat * w_hat` 的口径修正后，整体 SC 指标已回到合理区间**（`sc_log_pearson ~ 0.86`），不再出现“dense 预测导致相关极低”的现象。

2) **密度约束（B）对稀疏口径指标更敏感**：`sc_log_pearson_topk / sc_log_pearson_sparse` 与 `ecc_l2` 有小幅改善，但训练在 density 开启后更容易早停，提示当前 `lambda_density` 或 warmup 配置仍偏激进。

3) **零边惩罚（C）对 MSE 更敏感**：`sc_log_mse` 最小，但对 top-k/sparse 指标改善不明显（幅度较小）。

4) **密度 + 零边组合（D）并未优于单独项**：在本次配置下（同样较早进入早停），未呈现叠加收益。

## 下一步建议（用于下一轮提交的参数方向）

- 若继续验证“强制学习 mask”，建议缩短 warmup 并降低强度以避免早停：
  - `DENSITY_WARMUP_EPOCHS`：从 10 降到 2–5
  - `LAMBDA_DENSITY`：从 0.05 降到 0.005–0.02
  - 或提高 `PATIENCE`/`MAX_EPOCHS`，保证 density 有足够训练时间稳定下来
- `ZERO_LOG_*` 的量级当前较稳，可与更温和的 density 同时开启再观察。

