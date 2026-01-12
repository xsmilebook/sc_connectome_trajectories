# 数据字典（建设中）

本文件提供关键表与常用字段的轻量级索引。EFNY 的数据集假设与模式见 `docs/workflow.md` 的 EFNY 说明部分。

## EFNY（当前重点）

权威数据集说明：

- `docs/workflow.md`（EFNY 数据集说明）

常用表（路径为约定，见 `configs/paths.yaml`）：

- `data/processed/table/demo/EFNY_demo_processed.csv`: 清洗后的基本人口学信息。
- `data/processed/table/demo/EFNY_demo_with_rsfmri.csv`: 人口学信息与 rs-fMRI 质控合并（如包含 `meanFD` 等运动摘要字段）。
- `data/processed/table/metrics/EFNY_beh_metrics.csv`: 按被试汇总的行为指标宽表。
- `data/processed/table/demo/EFNY_behavioral_data.csv`: 人口学 + 行为指标合并表（如有生成）。
- `data/processed/table/sublist/sublist.txt`: 分析用被试列表（如有生成）。
- `data/processed/table/sublist/rest_valid_sublist.txt`: 质控后可用的 rs-fMRI 被试列表（如有生成）。

## QC 与 FC 相关文件

### QC

- `data/interim/table/qc/rest_fd_summary.csv`: rs-fMRI 头动 QC 汇总表（每 run 与被试级指标）。
- `data/processed/table/demo/EFNY_demo_with_rsfmri.csv`: 人口学与 rs-fMRI QC 合并表（用于建模的协变量与过滤）。

### FC

- `data/interim/functional_conn/rest/Schaefer*/<sub>_Schaefer*_FC.csv`: 单被试 FC 矩阵（CSV）。
- `data/interim/functional_conn_z/rest/Schaefer*/<sub>_Schaefer*_FC_z.csv`: Fisher-Z 后的单被试 FC 矩阵。
- `data/processed/fc_vector/Schaefer*/EFNY_Schaefer*_FC_matrix.npy`: 向量化 FC 特征矩阵（被试 × 特征）。
- `data/interim/avg_functional_conn_matrix/GroupAverage_Schaefer*_fc.csv`: 组平均 FC（未 Z）。
- `data/interim/avg_functional_conn_matrix/GroupAverage_Schaefer*_fisher_z.csv`: 组平均 FC（Z 后）。

## 约定

- 标识列因数据源而异；跨表合并时优先使用明确的 `subid` 列。
- 行为数值指标应以数值列存储；缺失值需统一表示（例如空值/NA 解析为 NaN）。
