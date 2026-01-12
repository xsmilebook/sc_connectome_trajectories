# 方法（Methods）

本文档总结当前交叉验证与置换检验流程的实现细节，并补充预处理方法学假设。以下描述强调实现一致性，不包含结果性结论。

参考代码：

- `scripts/run_single_task.py`
- `src/models/evaluation.py`
- `configs/analysis.yaml`

## 预处理方法学细节

本节补充影像与行为预处理的关键方法学假设，确保实现与评估一致。

### 影像预处理与 FC 构建

- rs-fMRI 预处理由 xcp-d 执行（`src/imaging_preprocess/xcpd_36p.sh`），输出用于后续 FC 计算的清洗时序；当前输出采用 CIFTI（`--file-format cifti`），并将 `--min-coverage` 设为 `0`（不基于覆盖率剔除 parcels）。
- task-fMRI 预处理同样使用 xcp-d，但在去噪（36P + 0.01–0.1 Hz）基础上额外回归任务诱发 HRF（block canonical HRF + event FIR）；入口为 `src/imaging_preprocess/xcpd_task_36p_taskreg.sh`。其自定义任务 confounds 由 `python -m scripts.build_task_xcpd_confounds` 从 Psychopy 行为 CSV 构建为一个 BIDS derivative dataset，并通过 xcp-d 的 `--datasets custom=<folder>` 与 `--nuisance-regressors <yaml>`（36P + task 列）加载（要求 TSV basename 与对应 fMRIPrep confounds TSV 完全一致）。SST 的 block/state 划分以 `Trial_loop_list` 的 loop1/loop2（如存在）优先；否则按试次数量识别 120（单 block）与 180（双 block，90/91 之间存在注视间隔）。
- `xcp_d-0.10.0` 的 `--mode none` 需要显式提供若干参数（如 `--file-format`、`--output-type`、`--combine-runs`、`--linc-qc`、`--abcc-qc`、`--min-coverage`、`--warp-surfaces-native2std`、`--smoothing`、`--despike`）。本项目将 `--fd-thresh` 设为足够大的值（`100`）以避免实际 scrubbing，从而避免因 censoring 导致时序长度不一致。
- 为与已验证可运行的参考配置一致，本项目将 `--smoothing` 设为 `2`（单位 mm）并将 `--despike` 设为 `n`（禁用 despike）。
- 当前 `task-fMRI` 的输出采用 CIFTI（`--file-format cifti`），并将 `--min-coverage` 设为 `0`（不基于覆盖率剔除 parcels；当未选择 atlas 或跳过 parcellation 时该参数不影响结果）。
- 头动 QC 指标由 `src/imaging_preprocess/screen_head_motion_efny.py` 汇总；阈值与字段以脚本为准。
- FC 计算使用 Schaefer 分区（`src/imaging_preprocess/compute_fc_schaefer.py`），随后进行 Fisher-Z（`src/imaging_preprocess/fisher_z_fc.py`）。
- 向量化特征在 `src/imaging_preprocess/convert_fc_vector.py` 中生成，作为建模输入 X。

#### 头动 QC 与 valid_subject 标准

- 以 `*desc-confounds_timeseries.tsv` 中的 `framewise_displacement` 计算每个 run 的统计量。
- `high_ratio` 为 `framewise_displacement > 0.3` 的比例。
- 对每个 run，判定有效条件为：`frame == 180`、`high_ratio <= 0.25` 且 `mean_fd <= upper_limit`。
- `upper_limit` 使用全体 FD 值的箱线图上界：`Q3 + 1.5*IQR`（基于帧数为 180 的 run）。
- `valid_subject` 定义为有效 run 数 `valid_num >= 2`。
- `meanFD` 为有效 run 的 mean FD 均值。

### 行为数据与指标构建

- 原始 app 行为数据的列名与任务映射在 `src/behavioral_preprocess/metrics/efny/io.py` 与 `src/behavioral_preprocess/metrics/efny/main.py` 定义。
- 试次级处理与 QC 逻辑在 `src/behavioral_preprocess/metrics/efny/preprocess.py` 实现（如 RT 解析、过滤与有效试次比例）。
- 行为指标由 `src/behavioral_preprocess/metrics/efny/metrics.py` 计算，并由 `src/behavioral_preprocess/metrics/compute_efny_metrics.py` 汇总为宽表，作为建模输入 Y。

#### 行为数据清洗与映射流程

- `normalize_columns` 将常见中文列名映射为英文字段（`task`, `trial_index`, `subject_id`, `item`, `answer`, `key`, `rt` 等）。
- `subject_code` 由文件名去掉 `_GameData.xlsx` 得到（`subject_code_from_filename`）。
- 任务名映射由 `main.py` 按 sheet 名称解析并规范化。
- `prepare_trials` 会：
  - 若缺少 `correct_trial`，基于 `answer == key` 计算；
  - 将 `rt` 转为数值（`errors='coerce'`），并按 `[rt_min, rt_max]` 过滤；
  - 在可用 RT 上做 3 SD 修剪；
  - 若有效试次比例低于 `min_prop`，该任务标记为无效（`ok=False`）。

#### 指标计算总流程与定义口径

指标计算以 `configs/behavioral_metrics.yaml` 为配置入口，`compute_tasks` 指定任务类型与参数，`metrics` 限定需要输出的字段。`metrics.py` 中的 `get_raw_metrics` 按任务类型调用对应计算函数，并仅保留配置要求的指标。关键口径如下：

- 共通输入：以 `prepare_trials` 清洗后的试次表为输入，缺失列或无有效试次时返回空结果（全部为 `NaN`）。
- 正确性定义：若存在 `correct_trial` 列直接使用；否则由 `answer == key` 推导。
- RT 口径：默认仅在正确试次上统计（当函数显式筛选 `corr`），并在 `[rt_min, rt_max]` 与 3 SD 修剪后计算均值与标准差。
- d′ 口径：使用命中率与虚报率的正态分位差（`d′ = Z(H) - Z(FA)`），并对极端比例做 0.5 校正。

#### 各任务类型的指标计算流程

1) N-back（`type: nback`）

- 试次分类：以 `item` 序列定义 `target`（与 `n_back` 步前相同）与 `nontarget`。
- 指标：
  - `ACC`：全部有效试次正确率。
  - `RT_Mean`/`RT_SD`：全部有效试次 RT 均值与标准差。
  - `Hit_Rate`：`target` 试次正确率。
  - `FA_Rate`：`nontarget` 试次错误率。
  - `dprime`：由 `Hit_Rate` 与 `FA_Rate` 计算。

2) 冲突任务（`type: conflict`；Flanker/ColorStroop/EmotionStroop）

- 条件划分：由 `item` 解析 `congruent` 与 `incongruent`，具体解析规则按任务名区分（例如 Flanker 的末尾方向、Stroop 的图片-文字一致性/序号规则）。
- 指标：
  - `ACC`/`RT_Mean`/`RT_SD`：整体正确率与 RT 统计。
  - `Congruent_ACC`/`Congruent_RT`、`Incongruent_ACC`/`Incongruent_RT`：条件内统计。
  - `Contrast_RT = Incongruent_RT - Congruent_RT`。
  - `Contrast_ACC = Incongruent_ACC - Congruent_ACC`。

3) 任务切换（`type: switch`；DCCS/DT/EmotionSwitch）

- 规则序列：
  - DCCS：以 `item` 的首字符代表规则。
  - DT：以 `item` 是否包含 `T/t` 映射为 `TN/CN` 规则。
  - EmotionSwitch：以 `item` 数字区间映射 emotion/gender 规则。
- 切换定义：与前一试次规则不同为 `switch`，相同为 `repeat`，并可用 `mixed_from` 丢弃前期非混合段。
  - 当前 block 约定（EFNY app）：DCCS 的 pure block 为 1–20（mixed 从 21 开始）；DT 与 EmotionSwitch 的 pure block 为 1–64（mixed 从 65 开始）。
- 指标：
  - `ACC`/`RT_Mean`/`RT_SD`：整体统计。
  - `Repeat_ACC`/`Repeat_RT`、`Switch_ACC`/`Switch_RT`：条件内统计。
  - `Switch_Cost_RT = Switch_RT - Repeat_RT`。
  - `Switch_Cost_ACC = Switch_ACC - Repeat_ACC`。

4) 停止信号任务（`type: sst`）

- 基于 `SSRT`/`ssd_var` 划分 stop 与 go 试次；stop 试次正确性以“未按键”为抑制成功。
- go 试次 RT 进行 `rt_max` 与 3 SD 修剪，并要求有效比例不低于 `min_prop`。
- 指标：
  - `ACC`：所有试次正确率。
  - `Stop_ACC`：stop 试次抑制成功率。
  - `Mean_SSD`：stop 试次 SSD 均值。
  - `SSRT`：采用积分法，`SSRT = Go_RT_quantile(p) - Mean_SSD`，其中 `p = 1 - Stop_ACC`。
  - `Go_RT_Mean`/`Go_RT_SD`：go 正确试次 RT 统计。

5) Go/No-Go 与 CPT（`type: gonogo`）

- 试次分类：`answer` 为 true/yes/1 视为 go，false/no/0 视为 nogo。
- go 试次 RT 进行 `rt_max` 与 3 SD 修剪，并要求有效比例不低于 `min_prop`。
- 指标：
  - `ACC`：全体正确率。
  - `Go_ACC`/`NoGo_ACC`：go/nogo 正确率。
  - `Go_RT_Mean`/`Go_RT_SD`：go 有效试次 RT 统计。
  - `dprime`：`Go_ACC` 作为命中率，`1 - NoGo_ACC` 作为虚报率。

6) ZYST（`type: zyst`）

- 试次解析：`trial_index` 解析为 `(trial, subtrial)`，仅保留含 0/1 子试次的 trial。
- 反应率门槛：`resp_var` 有效数量需达到 `min_resp`。
- 指标：
  - `ACC`/`RT_Mean`/`RT_SD`：整体统计。
  - `T0_ACC`/`T1_ACC`：子试次 0/1 的正确率。
  - `T1_given_T0_ACC`：在 T0 正确的条件下 T1 正确率。
  - `T0_RT`/`T1_RT`：子试次 RT 均值。

7) FZSS（`type: fzss`）

- 正确性：以 `answer == key` 定义正确试次。
- 指标：
  - `ACC`/`RT_Mean`/`RT_SD`：整体正确率与正确试次 RT 统计。
  - `Miss_Rate`：在 `answer == right` 子集中，`key != right` 的比例。
  - `FA_Rate`：在 `answer == left` 子集中，`key == right` 的比例。
  - `Correct_RT_Mean`/`Correct_RT_SD`：正确试次 RT 统计。

8) KT（`type: kt`）

- 指标：
  - `ACC`/`RT_Mean`/`RT_SD`：整体统计。
  - `Overall_ACC` 与 `Mean_RT` 为与 `ACC`、`RT_Mean` 等价的重复输出，用于下游兼容。

## 嵌套交叉验证（真实数据）

入口：`scripts/run_single_task.py`（`task_id=0`）调用 `src/models/evaluation.py` 中的 `run_nested_cv_evaluation`。

### 1) 数据输入与配置

- 输入：脑影像特征 X、行为特征 Y、可选协变量 C。
- CV 默认值来自 `configs/analysis.yaml`，可由命令行覆盖：
  - `evaluation.cv_n_splits`, `evaluation.inner_cv_splits`
  - `evaluation.cv_shuffle`, `evaluation.inner_shuffle`
  - `evaluation.outer_shuffle_random_state`, `evaluation.inner_shuffle_random_state`
  - `evaluation.score_metric`（当前为 `mean_canonical_correlation`）
- 模型超参数候选由 `scripts/run_single_task.py` 根据模型类型（adaptive PLS / sCCA / rCCA）构建，并作为参数网格传入嵌套 CV。

### 2) 外层 CV

- 使用 `KFold(n_splits=outer_cv_splits, shuffle=outer_shuffle, random_state=outer_random_state)`。
- 对每个外层折：
  - 内层 CV 仅在外层训练集上选择超参数。
  - 用最优参数在完整外层训练集上重训。
  - 在外层测试集上评估。

### 3) 内层 CV（超参数选择）

- 使用 `KFold(n_splits=inner_cv_splits, shuffle=inner_shuffle, random_state=inner_random_state)`，数据来自外层训练集。
- 对每个参数候选：
  - 在内层训练集拟合模型。
  - 在内层验证集计算评分。
  - 以各内层折评分均值选择最优参数。
- 评分指标为各成分典型相关的均值（`mean_canonical_correlation`）。

### 4) 折内预处理（仅用训练拟合）

所有预处理均在训练集拟合，并应用于对应验证/测试集：

- 缺失值处理：按训练集特征均值填补 NaN。
- 协变量回归：`ConfoundRegressor(standardize=True)` 在训练集拟合并应用到验证/测试集。
- 可选标准化：对 X、Y 各自使用 `StandardScaler`。
- 可选 PCA：对 X、Y 分别 `PCA(n_components=...)`，仅用训练集拟合。

这些步骤在 `run_nested_cv_evaluation` 内部，对内层与外层折分别执行。

### 5) 嵌套 CV 输出

`run_nested_cv_evaluation` 输出包含：

- 外层折的最优参数与内层 CV 统计。
- 外层测试集典型相关及其均值。
- 外层训练集典型相关（用于诊断）。
- 若模型提供，则包含加载（loadings）。

汇总输出包括：

- `outer_mean_canonical_correlations` 与 `outer_std_canonical_correlations`
- 外层测试相关矩阵（用于后续汇总）

结果保存路径：

```
outputs/results/real/<atlas>/<model_type>/seed_<seed>/
```

## 置换检验（逐成分）

当 `task_id > 0` 时，`scripts/run_single_task.py` 进入置换流程。

### 1) 置换种子与打乱方式

- `permutation_seed = random_state + task_id`
- 对 Y 做被试行打乱，X 保持不变。

### 2) 真实数据摘要的依赖

置换检验依赖真实数据的逐成分摘要，由以下脚本生成：

- `src/result_summary/summarize_real_loadings_scores.py`

读取路径：

```
outputs/results/summary/<atlas>/<model_type>/
```

摘要包含真实数据的逐成分得分与被试得分，用于后续逐成分置换。

### 3) 逐成分置换流程

令 `n_components` 为真实摘要中的成分数量。对每个 `k = 1..n_components`：

- 构造附加协变量：使用真实数据的被试得分中 `1..k-1` 成分：
  - `X_scores[:, :k-1]` 与 `Y_scores[:, :k-1]`
  - 若存在原始协变量，则与其拼接。
- 构建参数网格并强制 `n_components=1`。
- 使用 `run_nested_cv_evaluation` 在 `Y_perm` 上执行嵌套 CV。
- 取 `outer_mean_canonical_correlations` 的第一个元素作为该 k 的置换得分。

置换结果保存每个 k 的得分与嵌套 CV 细节。

### 4) 输出位置

置换结果保存路径：

```
outputs/results/perm/<atlas>/<model_type>/seed_<seed>/
```

元数据包含置换种子与 `configs/analysis.yaml` 中的 `permutation_n_iters`（用于可追溯性记录）。

