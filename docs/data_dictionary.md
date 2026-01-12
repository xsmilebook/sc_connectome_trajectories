# 数据字典

本文档提供本项目常用输入/输出文件的**路径约定**与**字段口径**，用于减少“同名不同义”与跨表对齐错误。由于本项目仅使用单一数据集，`data/` 下不再按数据集名称划分子目录。

## 目录约定（摘要）

- 原始输入：`data/raw/`
- 中间产物：`data/interim/`（例如 FreeSurfer）
- 处理后数据：`data/processed/`（训练与复现的稳定输入）
- 运行产物：`outputs/`（结果、图表、日志）

## 结构连接组（SC）

**路径建议：**

- `data/processed/sc_connectome/schaefer400/`

**文件格式：**

- 单文件对应一个被试×时间点（session）的 SC 矩阵。
- CSV 内容为 `N×N` 浮点矩阵（默认 `N=400`），无表头/无索引列。

**命名建议：**

- `<subid>_ses-baselineYear1Arm1.csv`
- `<subid>_ses-2YearFollowUpYArm1.csv`
- `<subid>_ses-4YearFollowUpYArm1.csv`

训练时按文件名解析 `subid` 与 `sesid`（见 `src/data/utils.py`）。

## 形态学（Schaefer400）

**路径建议：**

- `data/processed/morphology/`（可包含任意层级子目录）

**文件命名：**

- `Schaefer400_Morphology_<subid>.csv`

`python -m scripts.export_morphology_tables`（实现见 `src/preprocess/export_morphology_tables.py`）会在 `--morph_root` 下递归查找该命名模式，并推断：

- `subid`：从文件名提取（`sub-...`）
- `sesid`/`siteid`：从路径字符串中推断（存在不确定性，建议最终以 `subject_info_sc.csv` 为准）

## 被试信息表（subject_info_sc.csv）

**路径建议：**

- `data/processed/table/subject_info_sc.csv`

**用途：**

- 作为建模对齐表；CLG-ODE 训练会读取其 `age/sex/siteid` 等字段作为 covariates。

**关键字段（建议最低集合）：**

- `scanid`: 推荐为 `<subid>_ses-...`（用于与 SC/形态学对齐）
- `subid`: `sub-...`
- `sesid`: `ses-...`
- `age`: 连续变量（单位与口径由数据源决定，需在分析中保持一致）
- `sex`: 类别变量（编码口径见 `src/data/clg_dataset.py`）
- `siteid`: 站点/设备/中心标识（类别变量）

## 衍生表（可选）

由 `src/preprocess/export_morphology_tables.py` 生成，默认建议输出到 `data/processed/table/`：

- `subject_info_morphology_success.csv`: 成功提取形态学的 `scanid` 列表与基础字段
- `subject_info_sc_without_morphology.csv`: SC 具备但形态学缺失的条目（用于补算/排查）
