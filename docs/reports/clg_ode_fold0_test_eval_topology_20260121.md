# CLG-ODE fold0：测试集扩展评估（加入度/强度分布拓扑指标）

本文档汇总 fold0 需纳入报告的 baselines / 消融 / 创新模块模型，并在测试集上补齐 **度分布（degree）** 与 **节点强度分布（strength）** 等拓扑/结构指标，用于补足仅用 ECC 的证据不足问题。

## 1. 评估设置与口径

- 数据划分：与训练时一致（同一 `random_state` 的 outer split；fold0 的 GroupKFold）。
- 测试对选择：与现有实现一致（测试阶段使用固定的 `t0→t1` 对，避免随机性）。
- SC 评估域：`log1p(A)` 为主（`sc_log_*` 系列）。
- 拓扑/结构口径：
  - `ecc_*`：对预测矩阵做 top-k 稀疏化（k=真实正边数）后计算 ECC 相似度（与既有口径一致）。
  - `degree/strength`：同样在 **top-k 稀疏化后的预测矩阵** 上计算，以避免“边数不一致”导致的伪差异。
- 指标补齐方式：使用评估-only 作业对已有 run 目录复评估，输出写入 `test_sc_metrics_ext.json`（不重训）。

对应的评估脚本与提交方式见 `docs/workflow.md`（“测试集复评估（补充拓扑指标；不重训）”一节）。

## 2. 纳入报告的模型（fold0）

下表覆盖 `docs/final_plan.md` 与 `docs/reports/clg_fold0_fixedsupport_innovation_report_20260121.md` 中用于对比的核心模型集合，并加入 D2′（resume C2 + freeze）：

- B1：`clg_baseline_original`（原始 CLG-ODE 风格对照）
- B1′：`clg_focal_ld01`（edge focal 的对照）
- A1：`clg_fs_residual`（fixed-support + residual + dt gate + L_small）
- B2：`clg_fs_no_dt_gate`（去 dt gate）
- C2：`clg_fs_no_Lsmall`（去 L_small；当前主干最优候选）
- B3：`clg_fs_no_residual`（fixed-support 但无 residual）
- D2：`clg_fs_innov_default`（C2 + conservative innovation）
- D2′：`clg_d2prime_resume_c2_long_freeze`（从 C2 resume，冻结主干，仅训 innovation；更保守新边）

## 3. 可报告的指标清单（本次补齐后）

### 3.1 权重重建（整体 + 稀疏口径）

- `sc_log_mse`, `sc_log_mae`, `sc_log_pearson`, `sc_log_pearson_pos`
- `sc_log_pearson_topk`, `sc_log_pearson_sparse`

### 3.2 ECC 拓扑相似度（已有）

- `ecc_l2`, `ecc_pearson`

### 3.3 新增：度/强度分布拓扑指标（本次新增）

在 top-k 稀疏化口径下计算（预测边数与真值一致）：

- **degree（度）**：`deg_mae`, `deg_rmse`, `deg_pearson`, `deg_ks`
- **strength（节点强度，加权度）**：`strength_mae`, `strength_rmse`, `strength_pearson`, `strength_ks`

其中 `*_ks` 为一维分布的 KS 距离（越小越好）。

### 3.4 新增边/零边区域（仅对启用 innovation 的模型有意义）

- `mse_zero`, `mse_zero_strict`, `mse_new_region`
- `new_edge_precision_at_knew`, `new_edge_recall_at_knew`, `new_edge_auprc`

说明：当 `innovation_enabled=false` 时，上述 `new_edge_*` 指标在实现里会被记录为 0；**报告中应标为 N/A**，避免误读。

## 4. 结果汇总（fold0 测试集）

数据来源：各 run 目录下 `test_sc_metrics_ext.json`。

| ID | run | sc_log_mse↓ | sc_log_pearson_sparse↑ | topk↑ | ecc_pearson↑ | deg_mae↓ | deg_pearson↑ | strength_mae↓ | strength_pearson↑ | new_edge_AUPRC↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C2 | `clg_fs_no_Lsmall` | 0.119213 | 0.575598 | 0.577759 | 0.999386 | 7.66 | 0.9087 | 154.37 | 0.8904 | N/A |
| A1 | `clg_fs_residual` | 0.119231 | 0.575806 | 0.577879 | 0.998876 | 7.69 | 0.9086 | 155.07 | 0.8904 | N/A |
| B2 | `clg_fs_no_dt_gate` | 0.119271 | 0.576421 | 0.578376 | 0.999214 | 7.85 | 0.9080 | 158.41 | 0.8905 | N/A |
| D2 | `clg_fs_innov_default` | 0.119354 | 0.575673 | 0.578156 | 0.998699 | 7.72 | 0.9054 | 154.93 | 0.8904 | 0.036590 |
| D2′ | `clg_d2prime_resume_c2_long_freeze` | 0.120985 | 0.574152 | 0.576003 | 0.999385 | 7.67 | 0.9077 | 154.31 | 0.8881 | 0.036451 |
| B1′ | `clg_focal_ld01` | 0.133726 | 0.603959 | 0.608830 | 0.999153 | 6.99 | 0.9120 | 317.31 | 0.8875 | N/A |
| B3 | `clg_fs_no_residual` | 0.264022 | 0.288771 | 0.114038 | 0.969299 | 7.62 | 0.9033 | 435.08 | 0.5505 | N/A |
| B1 | `clg_baseline_original` | 0.474533 | 0.089119 | 0.073857 | 0.983969 | 42.31 | 0.5032 | 530.28 | 0.3688 | N/A |

## 5. 结论（面向报告写作）

### 5.1 当前“最优主干”结论（不引入新增边）

- 以 `sc_log_mse` 与 `sc_log_pearson_sparse/topk` 为主口径：**C2（`clg_fs_no_Lsmall`）为当前最稳的主干方案**；A1/B2 与其非常接近。
- `degree/strength` 指标在 C2/A1/B2/D2/D2′ 之间差异很小，说明 fixed-support + residual 系列在“结构分布”层面较稳定；B1/B3 明显更差。

### 5.2 创新模块（新增边）是否带来可见收益

- 在本次更保守与冻结主干的 D2′ 下：`new_edge_auprc≈0.03645`，与 D2 的 `≈0.03659` 基本相当；同时 D2′ 的主干指标略有退化（`sc_log_mse` 上升，`sparse/topk` 下降）。
- 因此：就 fold0 当前结果而言，**尚不能用测试集指标证明 innovation 在“新增边任务”上明显变好**；报告中应如实写为“未观察到提升/需进一步改进”，或将创新模块定位为可选探索项并明确下一步改进方向（见 `docs/final_plan.md` 的指标建议与 D2′ 设计动机）。

### 5.3 Baseline/消融的定性结论（用于讲故事）

- B1（原始）明显劣于 fixed-support + residual 系列，体现“继承稀疏支持集 + 学变化”的必要性。
- B3（无 residual）在权重与结构上均显著变差，支持 residual 的必要性。

## 6. 建议写入正文的“最佳方案”表述（示例）

- Full / Best trunk：C2（fixed-support + residual + dt gate，`lambda_delta_log=0`）。
- Innovation 模块：D2 与 D2′ 在新增边 AUPRC 上未体现收益（fold0），可作为“探索性模块”呈现，需按新增边子集指标继续迭代。

