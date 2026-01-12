# 工作流（单数据集）

本文档描述在 `ARCHITECTURE.md` 固定目录结构下，本项目的**数据落盘约定**与**最小可复现运行流程**。本项目仅使用一个数据集，因此 `data/` 下不再以数据集名称（如 `ABCD`）划分子目录。

## 目录与落盘约定

- `data/raw/`: 外部输入（不由本仓库脚本产生，必要时可为软链接/挂载点）
- `data/interim/`: 中间产物（可重新生成；例如 FreeSurfer 重建产物）
- `data/processed/`: 可复用的处理后数据（建模的稳定输入；例如 SC、形态学表格）
- `outputs/`: 运行产物（训练结果、图表、日志等）

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
