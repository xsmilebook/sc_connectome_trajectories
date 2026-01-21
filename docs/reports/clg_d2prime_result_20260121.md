# D2′ 实验结果检查与结论（fold0，2026-01-21）

## 实验目标（本轮）

1) **不牺牲 C2 的主干指标**：`sc_log_mse / sc_log_pearson / sc_log_pearson_topk / sc_log_pearson_sparse / ecc_*` 基本持平或更好。  
2) **让创新模块在新增边任务上明显变好**：即使全局指标不大动，也能用“新增边专用指标”证明模块有效。

## 本次完成的 D2′（rerun1）

- run：`outputs/results/clg_ode/runs/clg_d2prime_long_freeze_rerun1/fold0/`
- 设置要点（与设计一致）：
  - 起点为 C2 配置（fixed-support + residual + dt gate，`lambda_delta_log=0`）
  - 更保守 innovation：`TopM=200`、`K_new=40`、`δ=P97.5`、`τ=0.07`、`λ_new_sparse=0.20`
  - 仅长间隔启用：`g(dt)=clip((dt_months-9)/9,0,1)`
  - epoch≥10 冻结主干，仅训练 innovation head

## 结果对比（测试集）

对照基线：

- C2：`outputs/results/clg_ode/runs/clg_fs_no_Lsmall/fold0/test_sc_metrics.json`
- D2：`outputs/results/clg_ode/runs/clg_fs_innov_default/fold0/test_sc_metrics.json`
- D2′（本次）：`outputs/results/clg_ode/runs/clg_d2prime_long_freeze_rerun1/fold0/test_sc_metrics.json`

### 主干指标（必须不掉）

| 模型 | sc_log_mse | sc_log_pearson | topk | sparse | ecc_l2 | ecc_pearson |
|---|---:|---:|---:|---:|---:|---:|
| C2 | 0.119213 | 0.863325 | 0.577759 | 0.575598 | 26492.167 | 0.999386 |
| D2 | 0.119354 | 0.863085 | 0.578156 | 0.575673 | 26125.349 | 0.998699 |
| D2′ | 0.158971 | 0.858323 | 0.570745 | 0.544777 | 10594.876 | 0.998728 |

结论：**D2′ 显著劣于 C2/D2**（`sc_log_mse` 明显变大，`pearson/topk/sparse` 均下降）。本轮的“主干不掉”目标未达成。

### 新增边指标（证明 innovation 是否有效）

D2′ 产出的新增边指标（写入 `test_sc_metrics.json`）：

- `new_edge_precision_at_knew = 0.0325`
- `new_edge_recall_at_knew = 0.000329`
- `new_edge_auprc = 0.0364`

结论：新增边相关指标处于**非常低**水平（接近随机/不可用），本轮“新增边任务明显变好”的目标未达成。

## 诊断：为何 D2′ 会明显拖累主干

最核心原因：**该 D2′ run 并非从已训练收敛的 C2 权重出发，而是从头训练并在 epoch 10 就冻结主干**。  
这会导致主干在尚未达到 C2 的收敛水平前就停止学习，最终整体指标显著变差；后续仅训练 innovation head 无法弥补主干欠拟合。

（与日志一致：epoch 11 打印 “Freeze backbone…”，之后训练主要由 innovation loss 驱动。）

## 建议的最小修正（仍保持“只跑 1 个新实验”的精神）

若你仍希望验证 “D2′：更保守创新 + 冻结主干 + 仅长间隔启用”：

1) **必须从 C2 的 best checkpoint 继续**（这才符合“从 C2 出发”的原意）  
   - `RESUME_FROM=outputs/results/clg_ode/runs/clg_fs_no_Lsmall/fold0/clg_ode_fold0_best.pt`
2) 冻结主干可以从 epoch 0 开始（或 1），并将 `NEW_SPARSE_WARMUP_EPOCHS=0`（避免“冻结但 innovation 未启用”的空转期）。
3) 训练轮数可缩短（例如 `MAX_EPOCHS=40`），以节省时间且降低扰动风险。

这样才能更可靠地测试：在主干指标基本不变的前提下，新增边指标是否得到可解释提升。

