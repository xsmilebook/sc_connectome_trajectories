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
