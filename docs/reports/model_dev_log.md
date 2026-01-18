# 模型开发日志（Model Dev Log）

## 2026-01-16

### Smoke 结果（clg_ode_smoke_job13697506）

- 运行目录：`outputs/results/clg_ode/runs/clg_ode_smoke_job13697506`
- 配置：fold=0，max_epochs=8，patience=3，topo_scale_quantile=0.9，topo_warmup_frac=0.2，GradNorm(manifold+topo)
- epoch=1：train_topo_raw=3.318e7，train_topo=17.24，topo_scale=5.03e6，warmup=0.5，w_m=0.10，w_t=2.00
- epoch=2：train_topo_raw=3.354e7，train_topo=1.98，topo_scale=9.61e6，warmup=1.0，w_m=0.10，w_t=2.00
- epoch=3：train_topo_raw=3.359e7，train_topo=1.46，topo_scale=1.37e7，warmup=1.0，w_m=0.10，w_t=2.00
- 结论：topo 原始量级维持在 3e7，但经 q0.9 归一化 + log1p 后压缩到 1~2，loss 稳定且 GradNorm 生效。

### 下一步决策

1) 对比测试 q0.8：检查 `train_topo` 与 `topo_scale` 是否更稳、收敛更快。
2) 若 `w_m` 长期贴近下限，考虑将 `gradnorm_weight_min` 提高到 0.2 或降低 `gradnorm_lr` 以平滑权重更新。

### Smoke 结果（clg_ode_smoke_job13697578）

- 运行目录：`outputs/results/clg_ode/runs/clg_ode_smoke_job13697578`
- epoch=4：train_loss=0.485，val_loss=0.467
- topo 量级：train_topo_raw=3.414e7，train_topo=1.202，topo_scale=1.770e7
- GradNorm：w_m=0.10，w_t=2.00，warmup=1.0
- 结论：拓扑压缩与 GradNorm 稳定，loss 量级正常，可进入完整训练。

### Fold0 快速对照（10 组，残差跳连 + log-MSE）

- 运行目录：`outputs/results/clg_ode/runs/clg_ode_fast_residual_a/b/c`、`clg_fast_tau*`、`clg_fast_lmse*`、`clg_fast_nores_lmse01`、`clg_fast_dim48`
- 共同设置：fold0、`adjacent_pair_prob=1.0`、禁用 topo/KL/vel/acc、短时训练（≤6h 目标）
- 最优（按 `sc_log_mse`）：`clg_fast_lmse015`，`sc_log_mse=0.127624`，`sc_log_pearson=0.864544`
- 对比 identity baseline：`sc_log_mse=0.126836`、`sc_log_pearson=0.864430`，快速对照仍未超越
- 观察：`residual_tau` 与 `lambda_full_log_mse` 在该设置下影响极小；关闭 residual 的对照并未明显劣化
