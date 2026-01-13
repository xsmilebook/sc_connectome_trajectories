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

训练脚本会在 `--results_dir` 下保存：

- 最优 fold 的模型权重（`.pt`）
- 训练/验证摘要（JSON）

## 更新说明

当目录约定或关键输入输出路径变更时，请在 `docs/sessions/` 记录变更内容、原因与影响范围。
