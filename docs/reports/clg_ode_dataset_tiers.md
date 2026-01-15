# CLG-ODE 训练数据分层统计（严格文件存在性）

- 生成时间：2026-01-15 18:02:03
- SC 目录：`/ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/data/processed/sc_connectome/schaefer400`
- Morphology 目录：`/ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/data/processed/morphology`

## 统计口径
- 以 `sc_dir` 中的 SC CSV 文件为起点，按文件名解析 `scanid=sub-..._ses-...`。
- 以 `morph_root` 下的 `Schaefer400_Morphology_sub-*.csv` 为 morphology 输入，按路径解析 session。
- **严格要求**：同一 `scanid` 的 SC 与 morphology 文件同时存在，才计入可训练访视。

## 结果汇总
- SC 可用访视数：7692
- Morphology 可用访视数：11263
- SC∩Morphology 可用访视数：7692
- 可训练被试数（至少 1 个访视）：4566

## 按被试可用 session 数分布
- 1 个 session：1824 名被试
- 2 个 session：2358 名被试
- 3 个 session：384 名被试

## Tier 定义与数量
- Tier 1（≥3 个时间点）：用于学习加速度/非线性项（`L_acc`）。
- Tier 2（2 个时间点）：用于学习速度场（`L_vel`）。
- Tier 3（1 个时间点）：用于学习群体流形分布/去噪（`L_manifold`）。

- Tier 1：384
- Tier 2：2358
- Tier 3：1824

## 丢弃原因（如有）
- missing_sc_for_morph：3571

## 产出文件
- 详细被试列表：`docs/reports/clg_ode_dataset_tiers_subjects.csv`
- 机器可读摘要：`docs/reports/clg_ode_dataset_tiers.json`
