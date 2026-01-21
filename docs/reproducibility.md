# 可复现性与材料清单（结果/方法相关）

本文档面向“交付与复现”，整理：

1) 依赖与环境（包列表/配置文件）；  
2) 运行过程与复现实验的超参/命令；  
3) 数据位置（或链接）、测试样例数据打包方式；  
4) 模型 checkpoints 的定位方式（链接/路径）。

> 说明：本项目在当前集群上**不建议**在宿主机直接安装 PyTorch；训练/评估统一使用 **Slurm + Singularity/Apptainer 容器**（见 `docs/workflow.md`）。

## 1. 依赖与环境（包列表/配置文件）

### 1.1 容器定义（推荐作为“单一真源”）

- `scripts/containers/torch_gnn.def`
  - 基础镜像：`pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime`
  - 主要 Python 依赖：`numpy/pandas/scikit-learn/scipy/pytest`
  - 深度学习依赖：`torch==2.1.0+cu118`，`torch-geometric` + `pyg-lib/torch-scatter/torch-sparse/...`

### 1.2 requirements.txt（依赖清单）

- `requirements.txt`
  - 作为复现材料与容器构建的依赖清单（含 PyTorch CUDA wheel 源与 PyG wheel 源）。
  - 不建议在宿主机节点直接 `pip install -r requirements.txt`。

### 1.3 路径配置（数据/输出/容器）

- `configs/paths.yaml`
  - `local.data.sc_connectome_schaefer400`：SC 目录
  - `local.data.morphology`：形态学目录
  - `local.data.subject_info_sc`：被试信息表
  - `local.containers.torch_gnn`：容器 `.sif` 路径
  - `local.outputs.clg_ode`：CLG-ODE 输出根目录

## 2. 运行过程与复现实验超参/命令

### 2.1 CLG-ODE：训练（fold0）

以下脚本均为“一个模型一个 sbatch 任务”的入口，内部通过环境变量传参，并调用 `scripts/submit_clg_ode.sh`（容器内执行 `python -m scripts.train_clg_ode`）。

#### 纳入报告的 baseline / 消融 / 创新模块

- B1（原始对照）：`sbatch scripts/submit_clg_ode_baseline_original_fold0.sh`
- B1′（focal 对照）：`sbatch scripts/submit_clg_ode_focal_density_fold0.sh`
- A1（fixed-support + residual + L_small）：`sbatch scripts/submit_clg_ode_fixedsupport_residual_fold0.sh`
- B2（去 dt gate）：`sbatch scripts/submit_clg_ode_fixedsupport_no_dt_gate_fold0.sh`
- C2（去 L_small；当前主干候选）：`sbatch scripts/submit_clg_ode_fixedsupport_no_Lsmall_fold0.sh`
- B3（去 residual）：`sbatch scripts/submit_clg_ode_fixedsupport_no_residual_fold0.sh`
- D2（innovation 默认）：`sbatch scripts/submit_clg_ode_fixedsupport_innovation_fold0.sh`
- D2′（更保守 + resume C2 + freeze）：`sbatch scripts/submit_clg_ode_d2prime_fold0.sh`

> 每个 run 的**完整超参快照**会自动写入：`outputs/results/clg_ode/runs/<run>/fold0/args.json`。

### 2.2 测试集评估（补齐拓扑指标；不重训）

当你需要对既有 run 补齐扩展指标（degree/strength 等），使用评估-only 作业：

- 单个 run：

```bash
EVAL_RUN_DIR="outputs/results/clg_ode/runs/<run>/fold0" \
  sbatch scripts/submit_clg_ode_eval.sh
```

- 报告常用模型批量提交（每个模型一个 sbatch 任务）：

```bash
bash scripts/submit_clg_ode_eval_report_models_fold0.sh
```

输出写入 run 目录：

- `test_sc_metrics_ext.json`（扩展后的测试指标）
- `test_sc_metrics_ext_meta.json`（复评估元信息）

## 3. 数据（或链接）、测试样例数据

### 3.1 实验使用的数据位置（本地路径由配置管理）

- SC：`configs/paths.yaml` → `local.data.sc_connectome_schaefer400`
  - 文件：`<subid>_ses-*.csv`（`N×N`，无表头/索引，默认 `N=400`）
- 形态学：`configs/paths.yaml` → `local.data.morphology`
  - 文件：`Schaefer400_Morphology_<subid>.csv`
- 被试信息表：`configs/paths.yaml` → `local.data.subject_info_sc`
  - 文件：`subject_info_sc.csv`（含 `scanid/age/sex/siteid` 等）

若需要以“数据链接/共享”的形式交付，请将上述路径映射到你可分享的存储位置，并在交付材料中给出对应链接（本仓库默认不包含原始数据）。

### 3.2 测试样例数据（复制一个被试到 results）

可用脚本将 **一个被试** 的处理后数据复制到 `outputs/results/` 下，便于作为“最小测试样例/打包材料”：

```bash
bash scripts/prepare_test_sample_subject.sh
```

默认会自动选择 `data/processed/sc_connectome/schaefer400/` 中找到的第一个被试，并输出到：

- `outputs/results/test_sample_subject/<SUBID>/`

如需指定被试或目标目录：

```bash
bash scripts/prepare_test_sample_subject.sh --subid <SUBID> --dest outputs/results/test_sample_subject/<SUBID> --force
```

> 注意：该样例数据属于运行产物/拷贝，不纳入版本控制；也不要覆盖原始 `data/` 内容。

## 4. 模型 checkpoints（链接/路径）

### 4.1 训练产物位置

每个模型运行目录（fold0）：

- `outputs/results/clg_ode/runs/<run>/fold0/`

其中最关键的 checkpoint 文件：

- `clg_ode_fold0_best.pt`

### 4.2 报告模型的 checkpoint 路径（本地）

（示例）C2：

- `outputs/results/clg_ode/runs/clg_fs_no_Lsmall/fold0/clg_ode_fold0_best.pt`

其余模型同理：替换 `<run>` 为 `clg_baseline_original / clg_focal_ld01 / clg_fs_residual / ...`。

若需要“链接形式”交付 checkpoints，请将这些 `.pt` 文件上传到你可分享的存储（如对象存储/共享盘），并在报告中提供对应下载链接与哈希（可选）。

