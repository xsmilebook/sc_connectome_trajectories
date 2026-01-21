# fold0 对照与消融汇总报告：Fixed-support Residual + Conservative Innovation（2026-01-21）

## 目的与口径

- 目标：在 **不做交叉验证**（仅 fold0）的前提下，对比 Baselines 与消融，验证：
  - fixed-support（继承 `A0` 支持集）是否能在稀疏口径上稳定、避免无约束稠密化；
  - residual + Δt 门控是否必要；
  - `L_small`（残差收缩）是否带来增益；
  - conservative innovation（保守新增边）在固定 `K_new/TopM` 的强约束下是否改善“新边/零边区域”而不破坏整体指标。
- 统一评估：读取各 run 的 `fold0/clg_ode_results.json:test_sc_metrics`（测试集固定 `t0→t1`）。
- 指标说明：
  - `sc_log_mse/sc_log_pearson`：`log1p(A)` 上三角向量的 MSE / Pearson；
  - `sc_log_pearson_topk/sc_log_pearson_sparse`：top-k 稀疏口径相关（k 取真值正边数）；
  - `ecc_*`：ECC 曲线相似性（与既有实现一致）。

## 运行与模型定义

| 标签 | run 目录（fold0） | 关键设置（摘要） |
|---|---|---|
| B0 |（Identity baseline）| `A_pred=A0`（测试集与 CLG-ODE 一致的 20% subject split，random_state=42；该基线数值来自既有报告）|
| B1 | `outputs/results/clg_ode/runs/clg_baseline_original/fold0/` | 原始 CLG-ODE 风格：无 residual、无 fixed-support、无 innovation |
| B1' | `outputs/results/clg_ode/runs/clg_focal_ld01/fold0/` | `edge_loss=focal` 的对照（用于替代调 `pos_weight`） |
| A1 | `outputs/results/clg_ode/runs/clg_fs_residual/fold0/` | fixed-support + residual（Δt 门控开），无 innovation，含 `L_small`（`lambda_delta_log=0.01`）|
| B2 | `outputs/results/clg_ode/runs/clg_fs_no_dt_gate/fold0/` | 去 Δt 门控（residual scale 恒为 1），无 innovation |
| C2 | `outputs/results/clg_ode/runs/clg_fs_no_Lsmall/fold0/` | 去 `L_small`（`lambda_delta_log=0`），无 innovation |
| B3 | `outputs/results/clg_ode/runs/clg_fs_no_residual/fold0/` | fixed-support 但无 residual skip（直接预测并按 `A0` 支持集 mask） |
| D2 | `outputs/results/clg_ode/runs/clg_fs_innov_default/fold0/` | fixed-support + residual + conservative innovation（`TopM=400, K_new=80, δ=P95, τ=0.10, g(dt)=min(1,dt/1y)`；epoch≥10 才启用 innovation） |

## 核心结果（测试集）

> 数值越好：`sc_log_mse/ecc_l2` 越小越好；其余相关系数越大越好。

| 标签 | sc_log_mse | sc_log_pearson | sc_log_pearson_pos | sc_log_pearson_topk | sc_log_pearson_sparse | ecc_l2 | ecc_pearson |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.126836 | 0.864430 | 0.822734 | 0.574224 | 0.560635 | 10929.320 | 0.998740 |
| B1 | 1.543607 | 0.165683 | 0.162174 | 0.073667 | 0.088269 | 638377.217 | 0.983969 |
| B1' | 0.132057 | 0.862502 | 0.826777 | 0.608941 | 0.603967 | 11221.808 | 0.999121 |
| A1 | 0.119231 | 0.863273 | 0.820370 | 0.577879 | 0.575806 | 10591.973 | 0.998854 |
| B2 | 0.119271 | 0.863199 | 0.820112 | 0.578376 | 0.576421 | 10653.400 | 0.998841 |
| C2 | 0.119213 | 0.863325 | 0.820423 | 0.577759 | 0.575598 | 10574.013 | 0.998857 |
| B3 | 0.266387 | 0.674739 | 0.616615 | 0.114262 | 0.288959 | 42915.230 | 0.995579 |
| D2 | 0.119354 | 0.863085 | 0.820234 | 0.578156 | 0.575673 | 10701.172 | 0.998835 |

## 关键结论

1) **fixed-support + residual 是决定性改进**  
`B1 → A1/C2` 的提升巨大（MSE 从 ~1.54 降到 ~0.119），说明“继承 `A0` 支持集 + residual 学变化”是必要条件；`B3` 也明显劣于 residual 版本，进一步支持 residual 的必要性。

2) **Δt 门控在该配置下带来轻微增益**  
`A1` 相比 `B2` 略优（差距小但稳定），说明 dt gate 的“短间隔更稳”方向是合理的。

3) **`L_small`（lambda_delta_log）在该轮设置下没有收益，甚至略有负效应**  
`C2`（去 `L_small`）在 `sc_log_mse` 与相关性上略好于 `A1`。如果以当前指标为主，建议交付版默认关闭 `L_small`（或将其作为可选项）。

4) **D2（保守新增边）在当前默认超参下未带来总体指标增益**  
`D2` 与 `A1/C2` 在 `sc_log_mse/pearson/topk/sparse` 上基本持平或略差。根据 `metrics.csv` 的 best 监控 epoch，innovation 的 `val_new_kept_mean≈80`，说明模块确实在“放行新边”，但这一变化未转化为更好的全局测试指标。

5) **与 identity baseline（B0）的关系**  
- 在 `sc_log_mse` 上：`A1/B2/C2/D2` **明显优于** `B0`。  
- 在 `sc_log_pearson`（全上三角相关）上：当前模型 **略低于** `B0`。  
- 在稀疏口径（`topk/sparse`）上：`A1/B2/C2/D2` **略优于** `B0`。  
这意味着：如果报告主卖点是“稀疏口径一致性/拓扑稳定”，当前 fixed-support 系列已经具备可用优势；若必须在全量相关上超越 identity，需要进一步针对性优化（见下一节建议）。

## 建议（若仍有时间）

- **交付版推荐默认模型**：优先用 `C2`（fixed-support + residual + dt gate，且 `lambda_delta_log=0`），innovation 暂作为可选模块。  
- 若要让 D2 更“有用”：
  - 先补充 **新边专用指标**（`new_edge_precision/recall@K` 或 PR/AUPRC，在 `A0=0 & A1>0` 子集上），否则“新增边是否有效”很难仅靠全局指标判断；
  - 在不改 `K_new` 的前提下，可尝试更保守的阈值（提高 `δ` 分位数或增大 `τ` 的反向效果）与更强 `λ_new_sparse`，观察新增边是否主要出现在真正变化区域；
  - 若新增边主要破坏全量相关，可考虑只在长间隔（Long 桶）启用或增强 gate（更强依赖 dt）。

## 复现说明

- 本报告对应的对照/消融脚本均位于 `scripts/`，并在 `docs/workflow.md` 中列出。
- 各 run 的最终数值以 `fold0/clg_ode_results.json:test_sc_metrics` 为准。

