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
- CLG-ODE 训练与提交脚本使用以下键：`local.data.sc_connectome_schaefer400`、`local.data.morphology`、`local.data.subject_info_sc`、`local.outputs.clg_ode`、`local.containers.torch_gnn`。

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

### VectorLSTM baseline

```bash
python -m scripts.train \
  --sc_dir data/processed/sc_connectome/schaefer400 \
  --results_dir outputs/results/vector_lstm
```

### CLG-ODE

```bash
python -m scripts.train_clg_ode \
  --sc_dir data/processed/sc_connectome/schaefer400 \
  --morph_root data/processed/morphology \
  --subject_info_csv data/processed/table/subject_info_sc.csv \
  --results_dir outputs/results/clg_ode
```

运行目录与记录追溯：

- 训练脚本默认在 `--results_dir/runs/<timestamp>_job<jobid>/` 下创建独立运行目录（不需要手动指定），并将该运行目录作为本次训练的实际输出位置。
- 运行目录中会保存：
  - `args.json`：完整 CLI 参数快照
  - `run_meta.json`：时间戳、Slurm jobid、world size、git 信息等元数据
  - `metrics.csv`：按 fold×epoch 记录的 train/val 总损失与 `L_manifold/L_vel/L_acc` 组件
  - `test_sc_metrics.json`：测试集 SC 评估指标（`sc_log_mse/sc_log_mae/sc_log_pearson/ecc_l2/ecc_pearson`）
- 如需固定运行目录名称（便于复现实验分组），可使用 `--run_name <name>` 覆盖默认命名。

训练实现要点（与 `docs/reports/implementation_specification.md` 一致）：

- ODE 使用 `Δt = age(t) - age0`，`age0` 作为协变量输入动力学函数。
- `s_mean` 默认启用，可用 `--disable_s_mean` 关闭。
- 形态学特征按 ROI×Metric 列在训练集上做 Z-score（含可选 ICV/TIV 归一化）。
- 拓扑特征（ECC 向量）仅作条件输入，不参与训练 loss。
- 采用多起点配对采样：相邻访视 70%，任意 i<j 组合 30%。
- 默认启用轻量去噪增强（train-only）：形态学高斯噪声（`morph_noise_sigma=0.05`）与 SC 正边 dropout（`sc_pos_edge_drop_prob=0.02`）。
- 支持 1/2/3 个时间点的被试共同训练（Tier 3/2/1）；对应 `L_manifold/L_vel/L_acc` 的启用与 warmup 见 `docs/methods.md`。
- 测试评估当前仅覆盖 SC；单时间点被试使用重建输出评估，多时间点被试使用 `i→j` 预测输出评估。
- 测试评估采用固定的 `t0→t1` 对（避免随机性），ECC 评估时对预测矩阵做 top-k 稀疏化（k 为真实正边数）。

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

该脚本默认申请 `q_ai4` 的 1 张 GPU，并通过 `torchrun` 启动单卡训练。可按需调整 `#SBATCH --gres` 与 `#SBATCH -t`。

如需按 fold 分拆提交（Slurm array，单卡每 fold）：

```bash
sbatch --array=0-4 scripts/submit_clg_ode.sh
```

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
