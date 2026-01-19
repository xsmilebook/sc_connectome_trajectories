# 文档索引

本目录包含 `sc_connectome_trajectories` 项目的方法、流程与决策记录。除根目录文档外，本仓库约定 `docs/` 下内容使用中文撰写。

## 关键文件

- `docs/workflow.md`: 可复现实验与数据流转约定（单数据集；`data/` 下不再按数据集名称分子目录）。
- `docs/methods.md`: 模型与训练/评估口径（面向方法学一致性）。
- `docs/data_dictionary.md`: 关键表格与文件格式索引（字段/命名约定）。
- `docs/cluster_gpu_usage.md`: 当前集群 GPU 使用规范（Slurm + Singularity；避免登录节点运行与在线安装依赖）。
  - 包含端口冲突、日志目录等常见报错的处理建议。
- `docs/research/research_plan.md`: 研究计划与阶段性设想（可能与实现存在时间差）。
- `docs/reports/implementation_specification.md`: CLG-ODE 方法学最终规范（锁定；实现以此为准）。
- `docs/reports/clg_ode_dataset_tiers.md`: CLG-ODE 训练数据的 Tier 1/2/3 可用性统计（严格文件存在性口径）。
- `docs/reports/clg_ode_cv_merge_20260116.md`: CLG-ODE 5-fold 结果合并与分析（2026-01-16）。
- `docs/reports/baseline_lstm_gnn_20260119.md`: VectorLSTM 与 GNN baseline 结果汇总（2026-01-19）。

## sessions 与 notes

- `docs/sessions/`: 按日期记录的会话日志（变更原因、影响范围、回滚方式）。
- `docs/notes/`: 用户随手记录与自由想法（不保证结构化）。

## 运行日志与产物

- 运行日志建议写入 `outputs/logs/`。
- 结果与图表建议写入 `outputs/results/` 与 `outputs/figures/`。
