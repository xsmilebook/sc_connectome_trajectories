# 工作流（工程）

本文档描述在 `ARCHITECTURE.md` 固定目录结构下，脚本入口的标准化调用方式。

## 标准 CLI 形式

`scripts/` 下所有可运行入口应接受：

- `--dataset <DATASET_NAME>`
- `--config <PATH_TO_CONFIGS_PATHS_YAML>`

示例：

```bash
python -m scripts.run_single_task --dataset EFNY --config configs/paths.yaml --dry-run
```

EFNY 相关假设配置在 `configs/paths.yaml` 的 `dataset` 段落中。数据集无关的分析默认值可配置在 `configs/analysis.yaml`。

## data/ 与 outputs/ 约定

- `data/raw/`: 原始输入（不由流水线脚本生成）
- `data/interim/`: 中间产物（例如 MRI 预处理或连接组中间结果）
- `data/processed/`: 可复用的处理后数据（如表格、FC 向量特征）
- `outputs/`: 运行产物（结果、图表）
- `outputs/logs/`: 运行日志（脚本日志、SLURM stdout/stderr）

部分外部输入（如 fMRIPrep 输出）可能位于仓库外；请在 `configs/paths.yaml` 的 `dataset.external_inputs` 下配置绝对路径。

当前本地 `ABCD` 数据目录按如下方式组织：

- `data/interim/ABCD/`: 例如 FreeSurfer 等中间产物
- `data/processed/ABCD/`: 例如 `table/`、`sc_connectome/` 等可复用处理后数据
- `outputs/figures/ABCD/`、`outputs/results/ABCD/`: 图表与结果

## 预处理流程（影像与行为）

本节描述推荐的预处理顺序与关键产物，细节以脚本为准。

### 影像预处理与功能连接

1) rs-fMRI 预处理（xcp-d）。
   - 脚本：`src/imaging_preprocess/batch_run_xcpd.sh`、`src/imaging_preprocess/xcpd_36p.sh`
   - 输入：`data/raw/MRI_data/`
   - 输出：`data/interim/MRI_data/xcpd_rest`
   - 当前参数要点（xcp_d-0.10.0）：`--file-format cifti`、`--output-type censored`、`--min-coverage 0`、`--smoothing 2`、`--despike n`、`--fd-thresh 100`。
2) task-fMRI 预处理（xcp-d：去噪 + 去除任务诱发 HRF）。
   - 目标：在 36P + 0.01–0.1 Hz 滤波的基础上，将任务诱发共激活信号作为 confounds 一并回归。
   - 脚本：
     - 单被试单任务：`src/imaging_preprocess/xcpd_task_36p_taskreg.sh`
     - 写死路径、可直接分享运行：`temp/run_xcpd_task_direct.sh`（不依赖 `configs/` 与 `scripts.render_paths`）
     - 写死路径、批量提交（SLURM）：`temp/batch_run_xcpd_task_direct.sh`（不依赖 `configs/` 与 `scripts.render_paths`）
      - 批处理（SLURM 提交）：`src/imaging_preprocess/batch_run_xcpd_task.sh`
      - task confounds 生成（Python）：`python -m scripts.build_task_xcpd_confounds`
   - 依赖：
     - xcp-d 容器版本：`xcp_d-0.10.0`
     - Psychopy 行为记录：`data/raw/MRI_data/task_psych/`（可通过配置覆盖）
     - fMRIPrep 结果：由 `configs/dataset_tsinghua_taskfmri.yaml` 指定各任务的 fMRIPrep 根目录（可为仓库外绝对路径）。
       - xcp-d 需要 fMRIPrep 输入目录包含 `dataset_description.json`；若原始目录无写权限或缺失该文件，`xcpd_task_36p_taskreg.sh` 会在工作目录下创建 wrapper（仅包含 `dataset_description.json`）并把 `sub-<label>` 目录 bind 进去。
     - FreeSurfer subjects dir（可选但建议）：`/ibmgpfs/cuizaixu_lab/liyang/BrainProject25/Tsinghua_data/freesurfer/`（包含 `sub-*/surf,label,mri,...`），脚本会 bind 到容器内 `/freesurfer` 并设置 `SUBJECTS_DIR=/freesurfer`。
   - 输出：
     - xcp-d 结果：`data/interim/MRI_data/xcpd_task/<task>/`
     - 自定义 confounds（供 `--datasets custom=/custom_confounds` 使用）：`data/interim/MRI_data/xcpd_task/custom_confounds/<task>/sub-<label>/`
       - 该目录是一个 BIDS derivative dataset（包含 `dataset_description.json` 与 `confounds_config.yml`）。
       - task regressors TSV 位于 `sub-<label>/func/`，且 **文件名必须与 fMRIPrep 的** `*_desc-confounds_timeseries.tsv` **完全一致**，xcp-d 才能自动匹配并加载。
   - 回归口径（按任务诱发 HRF 去除）：
     - block/state 回归量：canonical HRF（SPM HRF）卷积。
       - NBACK：`state_pure_0back`、`state_pure_2back`（如存在混合段：`state_mixed`）。
       - SWITCH：`state_pure_red`、`state_pure_blue`、`state_mixed`。
       - SST：
         - 若行为日志为 120 试次：仅生成单个 block（`state_part1`）。
         - 若行为日志为 180 试次：生成两个 block（`state_part1`/`state_part2`），且在第 90 与 91 试次之间通常存在约 15 s 的注视间隔。
         - 若存在 `Trial_loop_list`：优先用 `loop1/loop2`（或等价命名）识别两个 block；每个 loop 可能包含 1 行“无刺激”记录（不产生 stimulus 事件，但会影响 block 持续时间）。
     - event 回归量：FIR（以 TR 为 bin，窗口默认为 20 秒）。
       - 共通：`stimulus`（刺激呈现）、`response`（按键 onset = `key_resp.started + key_resp.rt`）。
       - SST 额外：`banana`（当 `bad` 列包含 banana 时，以 `Trial_image_3.started` 作为事件 onset）。
   - 关键对齐假设：
     - 扫描起点：使用 Psychopy CSV 中首个 `MRI_Signal_s.started` 作为时间 0；缺失时回退到 `Begin_fix.started`，再回退到 0。
     - NBACK/SWITCH 的 block 类型优先由 `Trial_loop_list` 推断（行为试次行中 `Task_img` 常为空）。
   - `xcp_d-0.10.0` 的 `--mode none` 要求显式设置若干参数，否则会直接报错；当前脚本固定为：
     - `--file-format cifti`、`--output-type censored`、`--combine-runs n`
     - `--warp-surfaces-native2std n`、`--linc-qc n`、`--abcc-qc n`
     - `--min-coverage 0`、`--create-matrices all`、`--head-radius 50`
     - `--lower-bpf 0.01 --upper-bpf 0.1`、`--bpf-order 2`
     - `--motion-filter-type lp --band-stop-min 6`、`--resource-monitor`
     - `--smoothing 2`、`--despike n`
     - `--fd-thresh 100`（避免实际 scrubbing）
2) 头动 QC 汇总。
   - 脚本：`src/imaging_preprocess/screen_head_motion_efny.py`
   - 输出：`data/interim/table/qc/rest_fd_summary.csv`
   - 有效 run 判定：`frame == 180`、`high_ratio <= 0.25` 且 `mean_fd <= upper_limit`。
   - `high_ratio` 为 `framewise_displacement > 0.3` 的比例；`upper_limit` 来自全体 FD 值的 `Q3 + 1.5*IQR`。
   - `valid_subject` 标准：有效 run 数 `valid_num >= 2`。
3) 单被试 FC 与 Fisher-Z。
   - 脚本：`src/imaging_preprocess/compute_fc_schaefer.py`、`src/imaging_preprocess/fisher_z_fc.py`
   - 输出：`data/interim/functional_conn/rest/` 与 `data/interim/functional_conn_z/rest/`
4) FC 向量化特征。
   - 脚本：`src/imaging_preprocess/convert_fc_vector.py`
   - 输出：`data/processed/fc_vector/`

### 行为数据预处理与指标计算

1) app 行为数据规范化（如需）。
   - 脚本：`src/behavioral_preprocess/app_data/format_app_data.py`
   - 输入：`data/raw/behavior_data/cibr_app_data/`
2) 行为指标计算。
   - 脚本：`src/behavioral_preprocess/metrics/compute_efny_metrics.py`
   - 输出：`data/processed/table/metrics/EFNY_beh_metrics.csv`
   - 列名规范化：见 `src/behavioral_preprocess/metrics/efny/io.py` 的 `normalize_columns`。
   - 任务映射：见 `src/behavioral_preprocess/metrics/efny/main.py`，按 sheet 名称映射到内部任务键。
   - 试次清洗：见 `src/behavioral_preprocess/metrics/efny/preprocess.py` 的 `prepare_trials`。
3) 人口学与指标合并。
   - 脚本：`src/behavioral_preprocess/app_data/build_behavioral_data.py`
   - 输出：`data/processed/table/demo/EFNY_behavioral_data.csv`

## 结果目录

`src/result_summary/` 下的汇总脚本默认使用：

- `outputs/results`

如有需要，可通过 `--results_root` 覆盖。

示例：

```bash
python -m src.result_summary.summarize_real_perm_scores --dataset EFNY --config configs/paths.yaml --analysis_type both --atlas <atlas> --model_type <model>
```

## 快速自检（dry-run）

在仓库根目录执行：

```bash
python -m scripts.run_single_task --dataset EFNY --config configs/paths.yaml --dry-run
```

该命令验证导入与配置解析，不会读取 `data/` 或写入 `outputs/`。

## 集群执行（SLURM）

提交脚本：

- `scripts/submit_hpc_real.sh`
- `scripts/submit_hpc_perm.sh`

这些脚本将 `#SBATCH --chdir` 设为集群项目根目录，并将 SLURM stdout/stderr 与任务日志写入 `outputs/logs/...`。
注意：SLURM 的 `#SBATCH --output/--error` 路径为静态字符串，无法展开环境变量，因此在脚本头部保持固定路径。

示例：

```bash
sbatch scripts/submit_hpc_real.sh
```

## EFNY 数据集说明（唯一数据集）

本节整合 EFNY 的数据约定、预处理假设与相关脚本顺序，避免分散在单独的数据集文档中。本节不报告任何结果，细节以源码为准。

### 1) 数据位置与约定

- 规范根目录：`data/raw/`、`data/interim/`、`data/processed/`、`outputs/`、`outputs/logs/`。
- 运行时产物：`data/` 与 `outputs/`（含 `outputs/logs/`）不纳入版本控制。
- 被试标识：常见字段为 `subid`/`subject_id`/`subject_code`，以具体脚本为准。

### 2) 行为数据与 EF 指标

输入与输出：

- 原始输入目录：`data/raw/behavior_data/cibr_app_data/`
- 输入格式：每位被试一个 Excel 工作簿（`*.xlsx`）
- 输出指标宽表：`data/processed/table/metrics/EFNY_beh_metrics.csv`

列名与任务规范：

- 列名统一与任务映射见 `src/behavioral_preprocess/metrics/efny/io.py` 与 `src/behavioral_preprocess/metrics/efny/main.py`。
- 任务名映射遵循：`*1back*` -> `oneback_*`，`*2back*` -> `twoback_*`，其余保留规范化名称。

试次级预处理（见 `src/behavioral_preprocess/metrics/efny/preprocess.py`）：

- 缺失 `correct_trial` 时用 `answer == key` 计算。
- `rt` 转数值（`errors='coerce'`），可选按 `rt_min`/`rt_max` 过滤并进行 ±3 SD 修剪。
- 有效试次比例低于阈值（`min_prop`）时输出 `NaN`。

输出列名模式：

- `{task_name}_{metric_name}`（如 `FLANKER_ACC`, `oneback_number_dprime`）。

### 行为数据探索性分析（单被试试次统计）

目的：对单个 Excel 工作簿（单被试）按指标计算口径统计试次数量与条件分布，用于建模可行性评估与数据质量自检。

默认输入文件由 `configs/paths.yaml` 的 `dataset.behavioral.reference_game_data_file` 指定，位于 `dataset.behavioral.app_data_dir` 下。

口径：

- 探索性统计默认不进行反应时过滤/修剪，避免单被试过滤导致条件分布不稳定；`Included` 表示可被规则归类的试次（可能包含任务特定截断）。

运行：

```bash
python -m scripts.eda_behavior_trials --dataset EFNY --config configs/paths.yaml
```

输出：

- 报告写入 `configs/paths.yaml` 的 `docs_reports_root`（默认 `docs/reports/`），文件名根据工作簿自动生成。
- 可通过 `--excel-file`（绝对路径或仓库相对路径）与 `--report-out` 覆盖默认输入/输出位置。

### 行为刺激一致性分组（全样本）

目的：基于每位被试 Excel 工作簿中各 sheet 的 `正式阶段正确答案` 列进行一致性分组。允许部分被试缺少某些任务（sheet）：只要两名被试**共同存在**的任务中答案序列完全一致，则可分到同一组（缺失的任务不作为判定依据）。

注意：已观测到 SST sheet 可能出现 97 行（最后一行无效）；当前分组比较会忽略该无效行。详见 `docs/reports/app_data_format.md`。

运行：

```bash
python temp/group_app_stimulus_groups.py --dataset EFNY --config configs/paths.yaml
```

输入与输出：

- 输入目录：`data/raw/behavior_data/cibr_app_data/`（由 `configs/paths.yaml` 的 `dataset.behavioral.app_data_dir` 控制）。
- 输出目录：`data/interim/behavioral_preprocess/stimulus_groups/`（由 `dataset.behavioral.interim_preprocess_dir` 控制）。
  - 每组一个 `group_###_sublist.txt`
  - 汇总清单：`groups_manifest.csv`、`groups_manifest.json`

### DDM/HDDM 决策与层级建模（PyMC-based）

目的：基于任务口径与样本特征生成 DDM/HDDM 决策文档，并在可行范围内进行层级 HDDM（回归）计算与基础效果评估（如条件效应后验区间、LOO 指标）。

核心文档：

- 决策与模型设计：`docs/reports/ddm_decision.md`

计算脚本（保存 traces 与结果表到 `data/processed/table/metrics/ssm/`）：

- 2AFC DDM（FLANKER / SST go-only / DT / EmotionSwitch）：`python -m scripts.fit_ssm_task`
- 4-choice 多选 SSM（ColorStroop / EmotionStroop；`race_no_z_4`）：`python -m scripts.fit_race4_task`

本项目将“决策文档”与“计算产物”解耦：决策文档稳定记录建模口径与参数设计；计算脚本可在本地或 SLURM 上运行并产出可下载的 posterior traces 与 summary。

运行示例（本地小规模 pilot）：

```bash
python -m scripts.fit_ssm_task --dataset EFNY --config configs/paths.yaml --job-index 4 --max-files 10 --draws 40 --tune 40 --chains 1 --seed 1
python -m scripts.fit_race4_task --dataset EFNY --config configs/paths.yaml --job-index 1 --max-files 10 --draws 40 --tune 40 --chains 1 --seed 1
```

说明：

- 输入目录默认为 `data/raw/behavior_data/cibr_app_data/`（由 `configs/paths.yaml` 的 `dataset.behavioral.app_data_dir` 控制）。
- 层级 SSM 计算成本高：可用 `--max-files` 做 pilot（先确认模型与依赖可运行），再扩展到全样本。
- 决策要点（以 `docs/reports/ddm_decision.md` 为准）：
  - `ColorStroop/EmotionStroop` 为 4-choice，主模型为 **4-choice 多选 SSM（`race_no_z_4`；LBA 替代）**；报告中将 choice 归并为 Target / Word / Other。
  - `DT/EmotionSwitch` 原始为 4-choice，但可按轴向/维度做 **2-choice 重编码** 后进行层级 DDM，并建立两个并行对照模型：
    - Model A（Mixing）：`v/a/t0 ~ block + rule + block:rule`（pure+mixed）
    - Model B（Switch）：`v/a/t0 ~ trial_type + rule + trial_type:rule`（mixed only）
  - `SST` 标准 DDM 不适用，需补充 **go-only 2AFC DDM**（仅 go trials）。
- 输出与可追溯性：
  - 每个 `task×model` 保存 posterior traces（`InferenceData` netcdf）与轻量 summary（CSV/JSON），保存位置默认在 `data/processed/table/metrics/ssm/`（由脚本统一管理路径）。

### 3) 神经影像预处理与 QC

预期顺序：

1) xcp-d（rest 预处理）
2) 头动筛查（QC 汇总）
3) 人口学预处理与 QC 合并

xcp-d：

- 脚本：`src/imaging_preprocess/batch_run_xcpd.sh`、`src/imaging_preprocess/xcpd_36p.sh`
- 被试列表：`data/processed/table/sublist/mri_sublist.txt`
- 输出目录：`data/interim/MRI_data/xcpd_rest`

task-fMRI xcp-d（任务回归）：

在集群上建议在仓库根目录（`data_driven_EF/`）执行以下命令。

**(A) THU（Tsinghua）数据：生成 sublist + 批量提交**

1) 生成被试列表（建议输出到独立文件，避免与其他来源混淆）：

```bash
python -m scripts.build_taskfmri_sublist \
  --dataset EFNY_THU \
  --config configs/paths.yaml \
  --dataset-config configs/dataset_tsinghua_taskfmri.yaml \
  --out data/processed/table/sublist/taskfmri_sublist_thu.txt
```

2) 批量提交 xcp-d（nback/sst/switch）：

```bash
bash src/imaging_preprocess/batch_run_xcpd_task.sh \
  data/processed/table/sublist/taskfmri_sublist_thu.txt \
  "nback sst switch" \
  configs/dataset_tsinghua_taskfmri.yaml \
  EFNY_THU
```

说明：`batch_run_xcpd_task.sh` 默认会跳过已成功完成的 `subject×task`（判定依据：输出目录下同时存在 `sub-<LABEL>.html`、`log/*/xcp_d.toml` 与 `func/*desc-denoised_bold.dtseries.nii`）。如需强制重跑，可在命令前加 `XCPD_FORCE=1`。
若同一被试同一任务存在多个 Psychopy CSV 记录，当前默认按 **文件名中的时间戳**（`YYYY-MM-DD_HHhMM.SS.mmm`）选择最新的 CSV（而不是按文件修改时间）。

**(B) EFNY-XY（Xiangya）数据：生成 sublist + 批量提交（输出分开保存）**

1) 生成被试列表：

```bash
python -m scripts.build_taskfmri_sublist \
  --dataset EFNY_XY \
  --config configs/paths.yaml \
  --dataset-config configs/dataset_xiangya_taskfmri.yaml \
  --out data/processed/table/sublist/taskfmri_sublist_xy.txt
```

2) 批量提交（产物将保存到 `data/interim/MRI_data/xcpd_task_xy/`，日志保存到 `outputs/logs/EFNY_XY/xcpd_task_xy/`）：

```bash
bash src/imaging_preprocess/batch_run_xcpd_task.sh \
  data/processed/table/sublist/taskfmri_sublist_xy.txt \
  "nback sst switch" \
  configs/dataset_xiangya_taskfmri.yaml \
  EFNY_XY
```

说明：XY 的 `task_psych_xy` 目录可能采用两种命名：

- BIDS 风格：`sub-<LABEL>`（例如 `sub-XY20240719168CTY`）。
- Psychopy 原始风格：`XY_YYYYMMDD_NUM_CODE`（例如 `XY_20240719_168_CTY` 或 `XY_20240724_173_CY_156`）。

本项目生成 sublist 时会写入 `<LABEL>`（不包含 `sub-` 前缀）；对于 Psychopy 原始风格会自动转换为 `XY<YYYYMMDD><NUM><CODE>` 并去掉下划线（例如 `XY_20240719_168_CTY -> XY20240719168CTY`，`XY_20240724_173_CY_156 -> XY20240724173CY156`），以匹配 fMRIPrep 的 `sub-XY...` 命名与 xcp-d `--participant_label` 的要求。

- 单被试单任务提交：

```bash
sbatch src/imaging_preprocess/xcpd_task_36p_taskreg.sh THU20231119141SYQ nback
```

头动 QC：

- 脚本：`src/imaging_preprocess/screen_head_motion_efny.py`
- 输出：`data/interim/table/qc/rest_fd_summary.csv`

人口学处理：

- 脚本：`src/behavioral_preprocess/app_data/preprocess_efny_demo.py`
- 输出：人口学清洗表与 demo+QC 合并表（以脚本为准）

### 4) 建模流程中的 EFNY 假设

- 评估使用嵌套交叉验证；所有预处理仅在训练折拟合并应用到留出数据。
- 真实与置换分析共用入口：`scripts/run_single_task.py`（真实 `task_id=0`，置换 `task_id>=1`）。
- 当前支持的模型：`adaptive_pls`、`adaptive_scca`、`adaptive_rcca`。

对齐要求：

- 脑特征、行为特征与被试列表必须基于 ID 明确对齐。
- 若存在重排/过滤，应显式合并与重排；仅靠行序一致性视为不可靠。

Atlas 选择（Schaefer）：

- 通过 `configs/paths.yaml` 的 `dataset.files.brain_file` 切换 Schaefer100/200/400。
- 路径模式：`fc_vector/<Atlas>/EFNY_<Atlas>_FC_matrix.npy`。

### 5) 更新说明

当预处理或文件约定变更时，请在 `docs/sessions/` 记录变更内容、原因与涉及脚本/路径。

