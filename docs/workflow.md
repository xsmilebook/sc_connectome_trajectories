# 工作流（单数据集）

本文档描述在 `ARCHITECTURE.md` 固定目录结构下，本项目的**数据落盘约定**与**最小可复现运行流程**。本项目仅使用一个数据集，因此 `data/` 下不再以数据集名称（如 `ABCD`）划分子目录。

## 目录与落盘约定

- `data/raw/`: 外部输入（不由本仓库脚本产生，必要时可为软链接/挂载点）
- `data/interim/`: 中间产物（可重新生成；例如 FreeSurfer 重建产物）
- `data/processed/`: 可复用的处理后数据（建模的稳定输入；例如 SC、形态学表格）
- `outputs/`: 运行产物（训练结果、图表、日志等）

## 配置（configs/）

本项目使用 `configs/paths.yaml` 作为路径注册表：

- FreeSurfer 相关的大体量数据根目录保留为 HPC 上的绝对路径（例如 `/GPFS/.../ABCD/...`），**不要求迁移**。
- 仓库内可管理的默认路径使用相对路径（`data/raw`、`data/interim`、`data/processed`、`outputs/logs` 等）。
- Bash 脚本可通过 `python -m scripts.render_paths ...` 读取 `configs/paths.yaml` 并导出环境变量。
- 训练与提交脚本使用以下键：`local.data.sc_connectome_schaefer400`、`local.data.morphology`、`local.data.subject_info_sc`、`local.outputs.clg_ode`、`local.outputs.vector_lstm_baseline`、`local.outputs.gnn_baseline`、`local.containers.torch_gnn`。

## Atlas（Schaefer annot）

形态学提取需要 Schaefer2018 的 FreeSurfer `fsaverage` `.annot` 文件（左右半球各 1 个）。请确保 `ATLAS_DIR` 指向包含以下文件的目录：

- `lh.Schaefer2018_400Parcels_17Networks_order.annot`
- `rh.Schaefer2018_400Parcels_17Networks_order.annot`

建议将 atlas 文件放在 `data/external/atlas/`（Git 会忽略），并在 `configs/paths.yaml` 配置 `local.atlas.schaefer400_annot_dir`。HPC 脚本（如 `src/preprocess/run_schaefer400_morphology.sh`）会在未显式提供 `ATLAS_DIR` 时从该配置读取默认值。

推荐（但不强制）的子目录示例：

- `data/interim/freesurfer/`: FreeSurfer `SUBJECTS_DIR`
- `data/processed/sc_connectome/<atlas>/`: 结构连接组（按被试×时间点组织）
- `data/processed/morphology/`: 形态学 ROI 指标（Schaefer400）
- `data/processed/table/`: 人口学与对齐用表格
- `outputs/results/`: 训练与评估输出（JSON/pt 等）
- `outputs/figures/`: 可视化输出
- `outputs/logs/`: 运行日志（脚本 stdout/stderr）

## 数据输入格式

### 结构连接组（SC）

训练入口读取 `--sc_dir` 下的 `.csv` 文件：

- 文件内容：`N×N` 实数矩阵（默认 `N=400`），**无表头/无索引列**。
- 文件命名：建议包含 session 信息以便按时间排序，例如：
  - `<subid>_ses-baselineYear1Arm1.csv`
  - `<subid>_ses-2YearFollowUpYArm1.csv`
  - `<subid>_ses-4YearFollowUpYArm1.csv`

时间点排序逻辑见 `src/data/utils.py`（`session_sort_key`）。

CLG-ODE 训练阶段会对 SC 进行以下预处理（不修改原始 CSV 文件）：

- 对称化并清零对角线。
- 取 `log(1 + A)` 作为模型输入与权重回归的目标域。
- 不做额外 Top-K 或阈值稀疏化。
- 基于原始权重域计算全局强度协变量 `s` 与 `s_mean`。

### 形态学（Schaefer400 Morphology）

- `--morph_root` 下递归查找形如 `Schaefer400_Morphology_<subid>.csv` 的文件（实现见 `src/preprocess/export_morphology_tables.py`；入口为 `python -m scripts.export_morphology_tables`）。
- 该文件的生成可使用 `python -m scripts.extract_schaefer400_morphology`（实现见 `src/preprocess/extract_schaefer400_morphology.py`；依赖 FreeSurfer 命令行工具链）。

### 被试信息表（subject_info_sc.csv）

CLG-ODE 训练需要 `--subject_info_csv`，建议放在 `data/processed/table/subject_info_sc.csv`，至少包含：

- `scanid`（推荐形如 `<subid>_ses-...` 以与 SC/形态学对齐）
- `age`、`sex`、`siteid`（CLG-ODE covariates；具体编码见 `src/data/clg_dataset.py`）

## 预处理流程（最小集合）

1) FreeSurfer 重建（如果形态学需要）
   - 入口脚本：`src/preprocess/freesurfer.sh`
   - 建议将 `SUBJECTS_DIR` 指向 `data/interim/freesurfer/`
   - 如遇到作业中断导致“误判完成并跳过”（存在残留 `scripts/recon-all.done`），可在提交时设置 `FREESURFER_FORCE=1` 强制重跑；脚本也会在检测到 `recon-all.done` 但关键产物缺失时自动重跑。

2) 提取 Schaefer400 形态学（可选）
   - 入口：`python -m scripts.extract_schaefer400_morphology ...`
   - 生成 `Schaefer400_Morphology_<subid>.csv` 后，将其放入 `data/processed/morphology/`（或其子目录）

3) 汇总形态学覆盖情况（可选）
   - 入口：`python -m scripts.export_morphology_tables ...`
   - 输出建议写入 `data/processed/table/`

### 形态学汇总表（已生成）

当前统计文件已位于 `data/processed/table/`：

- `subject_info_morphology_success.csv`：成功生成形态学指标的被试清单与路径索引。
- `subject_info_sc_without_morphology.csv`：SC 表中缺失形态学指标的被试清单。
- `subject_info_sc.csv`：结构连接组与协变量对齐的基础表。
- `sublist_by_site/`：按 session/site 分组的被试子清单，用于批量提交形态学提取任务。

## 训练与结果输出

### VectorLSTM baseline（时序）

本基线使用现有 `VectorLSTM`，按 CLG-ODE 相同的被试划分（random_state=42，20% test），并过滤缺失形态学的扫描。

```bash
python -m scripts.train_vector_lstm_baseline
```

如需提交到集群（由用户提交）：

```bash
sbatch scripts/submit_vector_lstm_baseline.sh
```

输出目录：`outputs/results/vector_lstm_baseline/runs/<run>/fold{0..4}/`。

### GNN baseline（图预测）

轻量两层 GCN，节点特征为 `log1p(degree)`，通过内积解码下一时刻 SC。

```bash
python -m scripts.train_gnn_baseline
```

如需提交到集群（由用户提交）：

```bash
sbatch scripts/submit_gnn_baseline.sh
```

输出目录：`outputs/results/gnn_baseline/runs/<run>/fold{0..4}/`。

### CLG-ODE

注意（重要）：本项目的深度学习训练（含 CLG-ODE）**禁止在宿主机上直接运行**（例如直接 `python -m scripts.train_clg_ode` 或尝试 `pip/conda install torch`），因为当前集群节点的系统/工具链过旧，难以可靠安装与运行 PyTorch。请统一使用 **Slurm + 容器（Singularity/Apptainer）** 的方式训练，并由用户提交 `sbatch` 任务。

推荐提交方式（单卡，按 fold 训练；由用户提交）：

```bash
sbatch scripts/submit_clg_ode.sh
```

如需用 Slurm array 一次性提交 5 个 fold（0-4），可由用户执行：

```bash
sbatch --array=0-4 scripts/submit_clg_ode.sh
```

运行目录与记录追溯：

- 训练脚本默认在 `--results_dir/runs/<timestamp>_job<jobid>/` 下创建独立运行目录（不需要手动指定），并将该运行目录作为本次训练的实际输出位置。
- 运行目录中会保存：
  - `args.json`：完整 CLI 参数快照
  - `run_meta.json`：时间戳、Slurm jobid、world size、git 信息等元数据
- `metrics.csv`：按 fold×epoch 记录的 train/val 总损失、`L_manifold/L_vel/L_acc/L_topo`（含 `train_topo_raw/val_topo_raw`）、`vel/acc` 触发计数（`*_vel_count`/`*_acc_count`），以及拓扑缩放/GradNorm/warmup 相关字段。
  - `test_sc_metrics.json`：测试集 SC 评估指标（`sc_log_mse/sc_log_mae/sc_log_pearson/sc_log_pearson_pos/sc_log_pearson_topk/sc_log_pearson_sparse/ecc_l2/ecc_pearson`）
- 如需固定运行目录名称（便于复现实验分组），可使用 `--run_name <name>` 覆盖默认命名。

训练实现要点（与 `docs/reports/implementation_specification.md` 一致）：

- ODE 使用 `Δt = age(t) - age0`，`age0` 作为协变量输入动力学函数。
- `s_mean` 默认启用，可用 `--disable_s_mean` 关闭。
- 形态学特征按 ROI×Metric 列在训练集上做 Z-score（含可选 ICV/TIV 归一化）。
- 拓扑特征（ECC 向量）仅作条件输入，不参与训练 loss。
- 拓扑损失在训练期使用分位数尺度（q0.8/q0.9）+ `log1p` 压缩，并通过 GradNorm 仅对 `L_manifold/L_topo` 动态加权；`L_topo` 额外使用 20% cosine warmup。
- 采用多起点配对采样：相邻访视 70%，任意 i<j 组合 30%。
- 默认启用轻量去噪增强（train-only）：形态学高斯噪声（`morph_noise_sigma=0.05`）与 SC 正边 dropout（`sc_pos_edge_drop_prob=0.02`）。
- 支持 1/2/3 个时间点的被试共同训练（Tier 3/2/1）；对应 `L_manifold/L_vel/L_acc` 的启用与 warmup 见 `docs/methods.md`。
- 测试评估当前仅覆盖 SC；单时间点被试使用重建输出评估，多时间点被试使用 `i→j` 预测输出评估。
- 测试评估采用固定的 `t0→t1` 对（避免随机性），ECC 评估时对预测矩阵做 top-k 稀疏化（k 为真实正边数）。
- 为突出“短间隔稳定、长间隔改进”，建议在汇报时按测试对的 `dt_months` 做分位数分层：Short（≤P33）、Long（≥P67）（可选 Mid），并在表/图注中写出对应月份阈值与样本量。
- 拓扑损失（Betti curve）为实验性增强项，默认启用；细节见 `docs/methods.md`。
- 训练阶段对预测权重执行 top-k 稀疏化用于 `L_weight` 与 `L_topo`，与真实稀疏结构对齐（见 `docs/methods.md`）。
- 可选残差跳连：`--residual_skip` 启用 log 空间残差（`log1p(a0) + s(dt)*tanh(delta)`），`s(dt)=dt/(dt+tau)` 由 `--residual_tau` 控制。
- 可选全边 log-MSE：`--lambda_full_log_mse` 用于对齐 `test_sc_metrics` 的评估口径。
- 可选零边抑制与残差收缩：`--lambda_zero_log` 用于压制零边幅值，`--lambda_delta_log` 用于收缩预测到 identity。
- 可选稀疏度/边数约束（soft mask）：`--lambda_density` 约束 `p_hat` 的期望正边数接近真值（训练更稳、与稀疏评估口径一致）。
- `lambda_zero_log` 与 `lambda_density` 推荐配合 warmup/ramp，避免初期塌缩：`--zero_log_warmup_epochs/--zero_log_ramp_epochs`，`--density_warmup_epochs/--density_ramp_epochs`。
- 残差幅度上限：`--residual_cap` 控制 `tanh` 输出幅度上限。

按 fold 拆分提交（单卡替代多卡）：

- 使用 `--cv_fold` 仅训练单个 fold（0-based），例如：

```bash
python -m scripts.train_clg_ode --cv_fold 0
```

- 可用 Slurm array 提交 5 个独立任务（每个任务 1 张卡），避免 DDP 复杂性。

可选参数示例：

```bash
python -m scripts.train_clg_ode \
  --topo_bins 32 \
  --adjacent_pair_prob 0.7 \
  --disable_s_mean
```

集群提交脚本（Slurm + Singularity）：

```bash
sbatch scripts/submit_clg_ode.sh
```

常用参数（通过环境变量传给 `submit_clg_ode.sh`；由用户提交）：

```bash
LAMBDA_DENSITY=0.05 \
DENSITY_WARMUP_EPOCHS=10 \
DENSITY_RAMP_EPOCHS=20 \
LAMBDA_ZERO_LOG=0.05 \
ZERO_LOG_WARMUP_EPOCHS=10 \
ZERO_LOG_RAMP_EPOCHS=20 \
sbatch scripts/submit_clg_ode.sh
```

继续训练（单折，需提供 checkpoint 路径）：

```bash
RESUME_FROM="outputs/results/clg_ode/runs/<run>/fold0/clg_ode_fold0_best.pt" \
  sbatch scripts/submit_clg_ode_continue_fold0.sh
```

## 推荐唯一新增实验：D2′（从 C2 resume 的保守新增边）

如果你的目标是“**不牺牲 C2 主干指标**，但让创新模块在**新增边任务**上更有说服力”，推荐只追加一个实验 D2′：

- 关键点：**从 C2 checkpoint 恢复（resume）并从 epoch 0 冻结主干**，只训练 innovation head（避免从零训练时主干未收敛就冻结导致全局指标变差）。
- 提交脚本（fold0，默认从 C2 resume；由用户提交）：

```bash
sbatch scripts/submit_clg_ode_d2prime_fold0.sh
```

如需覆盖 checkpoint 路径（例如你自己的 C2 run 目录）：

```bash
RESUME_FROM="outputs/results/clg_ode/runs/<your_c2_run>/fold0/clg_ode_fold0_best.pt" \
  sbatch scripts/submit_clg_ode_d2prime_fold0.sh
```

该脚本默认优先使用 `q_ai8`，若不可用再回落到 `q_ai4`，并通过 `torchrun` 启动单卡训练（自动选择 `master_port` 避免端口冲突）。可按需调整 `#SBATCH --gres`。
提交前请确保日志目录存在：`mkdir -p outputs/logs/clg_ode`。

短跑 smoke 模板（单折、短 epoch）：

```bash
sbatch scripts/submit_clg_ode_smoke.sh
```

快速对照实验（fold0，用户提交）：

```bash
sbatch scripts/submit_clg_ode_fast_fold0_a.sh
sbatch scripts/submit_clg_ode_fast_fold0_b.sh
sbatch scripts/submit_clg_ode_fast_fold0_c.sh
sbatch scripts/submit_clg_ode_fast_fold0_d.sh
```

一次性批量提交（包含上面 4 个脚本 + 9 个扩展对照）：

```bash
bash scripts/submit_clg_ode_fast_fold0_batch.sh
```

Mask 学习与稀疏化对齐测试（fold0，用户提交；聚焦 `p_hat` mask + 密度/零边约束）：

```bash
sbatch scripts/submit_clg_ode_mask_fold0_a.sh
sbatch scripts/submit_clg_ode_mask_fold0_b.sh
sbatch scripts/submit_clg_ode_mask_fold0_c.sh
sbatch scripts/submit_clg_ode_mask_fold0_d.sh
```

一次性批量提交：

```bash
bash scripts/submit_clg_ode_mask_fold0_batch.sh
```

对应结果解读见：`docs/reports/clg_ode_mask_fold0_20260119.md`。

Fixed-support + 保守新增边（fold0，用户提交；推荐交付版默认口径）：

```bash
sbatch scripts/submit_clg_ode_fixedsupport_innovation_fold0.sh
```

唯一推荐新增实验（D2′，fold0，用户提交）：

```bash
sbatch scripts/submit_clg_ode_d2prime_fold0.sh
```

对照与消融（fold0，用户提交）：

```bash
# B0: Identity baseline（CLG-ODE test split）
sbatch scripts/submit_identity_baseline_sc_eval.sh

# B1: CLG-ODE baseline（no residual / no fixed-support / no innovation）
sbatch scripts/submit_clg_ode_baseline_original_fold0.sh

# A1: fixed-support + residual（no innovation）
sbatch scripts/submit_clg_ode_fixedsupport_residual_fold0.sh

# B2: remove residual dt gate
sbatch scripts/submit_clg_ode_fixedsupport_no_dt_gate_fold0.sh

# C2: remove L_small (residual shrinkage)
sbatch scripts/submit_clg_ode_fixedsupport_no_Lsmall_fold0.sh

# B3: fixed-support without residual skip
sbatch scripts/submit_clg_ode_fixedsupport_no_residual_fold0.sh
```

说明：

- `--fixed_support`：主干仅在 `A0>0` 支持集上学习/预测，避免无约束稠密化。
- `--innovation_enabled`：仅在 `A0=0` 的边上做候选 TopM 与 TopK(K_new) 放行（保守新增边）。
- D2′ 关键取舍：更强的长间隔 gate、更小的 TopM/K_new、更高的阈值分位数、更硬阈值温度、更强稀疏惩罚，并在 epoch≥10 后冻结主干仅训 innovation head（避免扰动 C2 主干指标）。
- 训练期会在 `metrics.csv` 追加记录：`train_new_edge/train_new_sparse/train_new_reg/train_new_q_mean/train_new_kept_mean`（以及对应的 `val_*`）。

## density 收敛性实验（10 个任务）

目标：让 `lambda_density` 这条线**可收敛**（更长训练、更温和、更对齐 early stopping 指标），避免“刚起效一点就被 early stop 掐断”。

批量提交（10 个实验，用户执行）：

```bash
bash scripts/submit_clg_ode_density_10exp_batch.sh
```

内容：

- 8 组网格：`DENSITY_WARMUP_EPOCHS ∈ {2,5}` × `LAMBDA_DENSITY ∈ {0.002,0.005,0.01,0.02}`，固定 `MAX_EPOCHS=100`、`PATIENCE=25`，并将 early stop 监控改为 `val_sc_log_pearson_sparse`（需要每 epoch 计算 val SC 指标）。
- 2 组两阶段训练（Phase-1 → Phase-2）：先学主任务，再用更小学习率微调并打开 density/zero 约束；Phase-2 分别用 `val_sc_log_pearson_sparse` 和 `monitor_mse_plus_density` 作为 early stop 监控。

注意：

- `scripts/submit_clg_ode_density_twostage_fold0_{a,b}.sh` 会提交两个作业并用 `--dependency=afterok` 串联；Phase-2 使用 Phase-1 的 `clg_ode_fold0_best.pt` 作为 `RESUME_FROM`。

本轮实验结果解读见：`docs/reports/clg_ode_density_convergence_fold0_20260120.md`。

## mask 根因定位实验（10 个任务）

目标：不再盲扫 `lambda_density`，而是定位：

- `sum(p_hat)/k` 是不是长期 `>>1`（过稠密，通常是 `L_edge` 推出来的）
- 即使密度对齐了，top-k 位置是否正确（`precision@k/recall@k/AUPRC`）

批量提交（10 个实验，用户执行）：

```bash
bash scripts/submit_clg_ode_mask_rootcause_10exp_batch.sh
```

包含：

- 1 个诊断基线：`scripts/submit_clg_ode_maskdiag_fold0.sh`（输出 `mask_ratio/mask_mean_p/mask_p10/p50/p90/precision@k/recall@k/AUPRC`）
- 9 个网格：`pos_weight ∈ {1,2,5}` × `lambda_density ∈ {0,0.01,0.05}`（见 `scripts/submit_clg_ode_posweight_density_grid_fold0_batch.sh`）
- 1 个 focal 对照：`scripts/submit_clg_ode_focal_density_fold0.sh`（作为替代“调 pos_weight”的方案）

诊断指标写入每个 run 的 `metrics.csv`：

- `train_mask_ratio/val_mask_ratio`（`sum(p_hat)/k`，上三角定义一致）
- `train_mask_p10/p50/p90`（`p_hat` 分位数）
- `train_mask_precision_at_k/val_mask_precision_at_k`、`train_mask_auprc/val_mask_auprc`

本轮实验结果解读见：`docs/reports/clg_ode_mask_rootcause_fold0_20260120.md`。

可选环境变量（不改脚本也能快速调整）：
`FOLD_ID=0`，`MAX_EPOCHS=8`，`PATIENCE=3`，`BATCH_SIZE=2`，`TOPO_SCALE_Q=0.9`，`TOPO_WARMUP_FRAC=0.2`，`RUN_TAG=smoke`。

一键提交包装脚本（直接走默认 smoke 参数）：

```bash
scripts/submit_clg_ode_smoke_now.sh
```

如需按 fold 分拆提交（Slurm array，单卡每 fold）：

```bash
sbatch --array=0-4 scripts/submit_clg_ode.sh
```

结果将写入统一的运行根目录（`runs/<time>_job<array_job_id>/fold{0..4}/`）。

如需让同一模型的不同 fold 强制落在同一目录，可在提交前显式设置统一时间戳：

```bash
export RUN_DATE=20260116
export RUN_TIME=011137
sbatch --array=0-4 scripts/submit_clg_ode.sh
```

日志与诊断提示：

- 若出现 `Address already in use`（`torchrun` 端口冲突），确保使用最新提交脚本（自动随机端口）。
- 若 `train_vel/val_vel/train_acc` 长期为 0，查看 `metrics.csv` 的 `*_vel_count`/`*_acc_count`，并检查 fold 的样本长度分布（脚本会在 fold 开始时打印）。

训练脚本会在 `--results_dir` 下保存：

- 最优 fold 的模型权重（`.pt`）
- 训练/验证摘要（JSON）

## 集群 GPU 运行规范

本仓库在当前集群上运行深度学习/GNN 任务时，统一遵循 `docs/cluster_gpu_usage.md`：

- 仅通过 Slurm 队列申请 GPU（不在登录节点直接训练）。
- 使用 Singularity 容器 `singularity exec --nv`，避免在计算节点动态安装/编译依赖。
- Slurm 日志建议写入 `outputs/logs/`，训练结果写入 `outputs/results/`（与本仓库结构一致）。

### 容器构建（torch+CUDA+GNN）

推荐使用 `scripts/containers/torch_gnn.def` 通过远程构建生成 `.sif`，容器路径由 `configs/paths.yaml` 的 `local.containers.torch_gnn` 控制（默认 `data/external/containers/torch_gnn.sif`）。

```bash
bash scripts/build_torch_gnn_container.sh
```

## 数据分层统计（Tier 1/2/3）

如需统计“严格文件存在性”下可用于 CLG-ODE 训练的被试数量（按可用 session 数分层），可运行：

```bash
python -m scripts.report_clg_ode_tiers
```

该脚本默认读取 `configs/paths.yaml` 的 `local.data.sc_connectome_schaefer400` 与 `local.data.morphology`，并在 `docs/reports/` 下生成：

- `clg_ode_dataset_tiers.md`
- `clg_ode_dataset_tiers.json`
- `clg_ode_dataset_tiers_subjects.csv`

## 更新说明

当目录约定或关键输入输出路径变更时，请在 `docs/sessions/` 记录变更内容、原因与影响范围。
