# 评估方案与消融实验设计文档

**方法：Fixed-support Residual CLG-ODE + Conservative Innovation（保守新增边）**
**目标：证明**

1. *mask 继承 baseline 结构* 能显著提升稀疏口径一致性与拓扑稳定性；
2. *残差/Δt 门控* 能保证短间隔不劣于 identity，同时在长间隔改善；
3. *保守新增边* 能在不破坏稀疏/拓扑的前提下，改善零边区域的误差与新增边召回。

---

## 1. 总体评估方案（推荐你明天交付就用这个）

### 1.1 数据切分与任务设置（展示优点的“好看”设置）

* **主任务（最能体现你方法优势）**：同被试纵向预测 `t0 → t1`

  * **Δt 分层（用分位数切分）**：按测试对的 `dt_months` 分布计算分位数阈值并分桶
    * **Short**：`dt_months ≤ P33(dt_months)`，强调 *不劣于 identity* + 拓扑不漂移
    * **Long**：`dt_months ≥ P67(dt_months)`，强调 *学到变化*（delta 的优势更明显）
    * （可选）**Mid**：其余样本（用于完整性展示，但不是重点）
  * **报告格式（必须写清具体范围与样本量）**：例如 “Short（≤4.8 months, n=xxx），Long（≥13.2 months, n=xxx）”（用你实际计算得到的 P33/P67 替换）。
* **评估口径**：上三角、去对角；log 域为主（`log1p(A)`）
* **不做交叉验证（节省时间）**：只跑并报告单一 `fold0`（或你当前最稳的一个 fold）即可；如需补充稳健性，优先在 *同配置* 下追加 1 个 fold 做“趋势一致性”核对（不要求 5-fold 平均）。

> 你想突出“继承 mask + 学变化”的优点，就要把评估按 Δt 分层展示：短间隔看稳定性，长间隔看改进幅度。

---

## 2. 方法方案（最终模型建议，作为“Full Model”）

### 2.1 主干：Fixed-support residual（继承 A0 mask，只学变化）

* **硬 mask**：`m0 = 1(A0>0)`
* **预测**：在 log 域预测
  [
  \hat y = y0 + \Delta \hat y,\quad \text{仅在 } m0=1 \text{ 上学习}
  ]
* **输出**：`m0=0` 默认 0（不会引入伪阳性）

### 2.2 保守新增边模块（Conservative Innovation，尽量不乱加边）

* 只在 `m0=0` 的边上 **候选集筛选**（Top-M by score）
* **硬上限**：每个样本最多新增 `K_new` 条边（TopK(q)）
* **Δt 门控**：`g(Δt)` 保证短间隔不乱新增、Δt=0 时新增边=0
* **强保守正则**：惩罚 `mean(q)`，压低新增倾向

#### 2.2.1 创新模块固定超参（E=79,800；N=400 parcellation）

> N=400 时，上三角边数 `E = N(N-1)/2 = 79,800`。以下默认值以“非常保守新增边”为目标。

* **硬上限**：`K_new = 80`（约占全边数 `0.1%`）
* **候选集大小**：`TopM = 400`（`= 5 × K_new`）
  * 含义：仅在 `m0=0` 的边里先挑 Top 400 条最可能的新边，再从中最多放行 80 条。
* **Δt 门控（以月份计）**：
  * `g(dt) = min(1, dt_months / 12)`
  * `dt=0 → 0`；`dt=6 → 0.5`；`dt≥12 → 1`
* **创新概率（保守阈值）**：
  * 记创新 head 的原始输出为 `l_new_ij`（raw logit/score，pre-sigmoid，且未乘 `g(dt)`、未减去 `δ`），则：
    * `q_ij = g(dt) * sigmoid((l_new_ij - δ)/τ)`
  * `τ = 0.10`
  * `δ = P95(s_new over candidate TopM)`
    * 计算域：每个样本、仅在 `m0=0` 的候选 TopM 边集合上计算 P95（更保守）；若来不及可退化为 batch 级 P95。
* **损失（偏 precision、强保守）**：
  * edge classification：focal（`gamma=2, alpha=0.25`）
  * 新增稀疏惩罚：`L_new_sparse = mean(q)`（只在候选集上）
  * `λ_new_sparse = 0.10`
  * （可选）新增真阳性权重回归：若开启，`λ_new_reg = 1.0`（只在 true new edges）
* **启用时序（避免破坏主干）**：
  * epoch 0–10：`λ_new_sparse = 0`（只训主干 residual）
  * epoch 10–20：线性 ramp 到 `0.10`
  * epoch ≥20：保持 `0.10`

---

## 3. 对照实验方案（Baselines，建议至少 6 个，展示优势够用）

> 命名上你可以直接在 runs 里用这些标签，写报告更清晰。

### 3.1 必备 Baselines（强烈建议）

**B0. Identity baseline**

* `A_pred = A0`
* 目的：证明你方法 *至少不劣于* 最强 baseline（短间隔尤其重要）

**B1. CLG-ODE 原始输出（不继承 mask / 或 learned mask + weight）**

* 目的：对比“从头学 mask”会导致位置差、稀疏口径差

**B2. 仅修正输出口径：A_pred = p*w（soft mask）**

* 目的：证明“不是因为 w_hat 全正导致 dense 崩盘”，你已经做过类似诊断
* 作为“传统 mask 学习路线”的代表

**B3. Fixed-support 但不做 residual（直接预测 y1 on m0）**

* `ŷ = f(z(t1))`，在 m0=1 回归 y1
* 目的：突出 residual（学习变化）带来的稳定性与 dt=0 一致性

### 3.2 可选“更好看”的 Baselines（增强说服力）

**B4. A0 + 线性漂移（简单 age/dt 线性回归）**

* 目的：证明非线性/ODE 的增益（尤其长间隔）

**B5. 仅在模块/网络均值层面预测（粗粒度 baseline）**

* 目的：证明你的模型能在“边级别”做得更好

---

## 4. 消融实验方案（Ablations：逐个证明模块有效）

下面给一个“最推荐的消融树”，每个消融都能对应一个你想强调的优点。

### A. 继承 mask 是否必要？

**A1. Full model（继承 mask + residual +（可选）新增）**
**A2. 去掉继承 mask：改为 learned mask（p_hat 学位置）**

* 预期：A2 的 `precision@k/AUPRC/topk/sparse pearson` 明显更差
* 目的：证明“位置学不动”→ 继承 mask 是关键

### B. Residual/Δt 门控是否必要？

**B1. residual + dt 门控（Full 的主干）**
**B2. residual 但无 dt 门控（Δt=0 不强制为 0）**
**B3. 无 residual（直接预测 y1）**

* 预期：B1 在短间隔最稳；B3 更易漂移拓扑；B2 可能出现 dt 小也乱改
* 目的：突出“短间隔不输 identity”的设计点

### C. 小变化先验是否必要？

**C1. 带 `L_small = |Δy|`**
**C2. 去掉 L_small**

* 预期：C2 可能在长间隔 MSE 变好一点但拓扑更漂（你可以强调 C1 的稳健）
* 目的：突出“变化可解释且稳定”

### D. 新增边模块是否有效（保守、不会破坏整体）

**D1. 无新增边（m0 外全 0）**
**D2. 保守新增边（候选集 + TopK(K_new) + mean(q) 惩罚 + dt gate）**
**D3. 新增边但去掉保守机制之一（用于证明每个保守组件必要）**

* D3a：去掉候选集（全 m0=0 都考虑）→ 假阳性上升
* D3b：去掉 TopK 上限 → 新增边数量失控
* D3c：去掉 `mean(q)` 惩罚 → q 偏大，稠密化
* D3d：去掉 dt gate → Δt 很小也新增（会很难看）
* 目的：证明“我们能新增，但非常克制，且各保守组件都必要”

#### D2′（推荐唯一新增实验）：更保守 + 冻结主干 + 仅长间隔启用

目标：**不牺牲 C2 的主干指标**（`sc_log_mse/pearson/topk/sparse/ecc` 基本持平或更好），同时让创新模块在“新增边任务”上明显变好（即使全局指标不大动，也能证明模块有效）。

- 起点：**从已训练好的 `C2` checkpoint 恢复（resume）**（fixed-support + residual + dt gate，且 `lambda_delta_log=0`），保证主干已收敛且指标稳定。
- 仅改创新模块（更保守、更不扰动）：
  - 仅长间隔启用：`g(dt)=clip((dt_months-9)/9, 0, 1)`（dt<9mo 不新增；dt=18mo 满功率）
  - `K_new=40`，`TopM=200`
  - `δ=P97.5`（逐样本、仅 `m0=0`、仅候选 TopM）
  - `τ=0.07`
  - `λ_new_sparse=0.20`（只在候选集上 `mean(q)`）
- 训练策略（不破坏主干）：
  - **epoch 0 起：冻结主干，仅训练 innovation head**（因为主干来自 C2 checkpoint，不再需要 warmup）
  - 同时将 `new_sparse_warmup_epochs=0`，避免“冻结但 gate=0 导致无梯度更新”的无效阶段。
- 对应提交脚本（fold0，默认从 C2 resume）：`scripts/submit_clg_ode_d2prime_fold0.sh`（可用环境变量 `RESUME_FROM=...` 覆盖 checkpoint 路径）

### E. 训练策略消融（可选但很有“工程说服力”）

**E1. Phase-2 仅训练 innovation head（freeze 主干）**
**E2. Phase-2 全量训练**

* 目的：证明“保守新增边不会破坏主干拟合”

---

## 5. 评测指标集（覆盖：权重、稀疏位置、拓扑、谱/几何、变化合理性）

> 建议你把指标分成 6 组，每组挑 3–6 个核心指标在正文里画图/表格，其余放 Supplement。

### 5.1 边权重重建指标（整体）

在 `log1p(A)` 域：

* **MSE / MAE**：`sc_log_mse`, `sc_log_mae`
* **Pearson / Spearman**：`sc_log_pearson`, `sc_log_spearman`
* **Pos-only**（仅真正边 `A_true>0`）：`sc_log_pearson_pos`, `mse_pos`

> 作用：证明 “在保留稀疏性的同时，权重拟合不差”。

### 5.2 稀疏口径一致性（最关键，用来突出继承 mask）

* **Top-k 相关**：`sc_log_pearson_topk`（k = 真值正边数）
* **Sparse 相关**：`sc_log_pearson_sparse`（仅在真值正边位置计算）
* **密度偏差**：`mask_ratio=sum(p)/k`（若有 p）或 `nnz(A_pred)/k`（对 hard 输出）
* **零边误差**：`mse_zero`（在 A_true=0 位置的 log1p 输出）

> 作用：证明“我们不会乱加边”，以及“新增边模块只在必要时改善 zero 区域”。

### 5.3 额外拓扑/分布指标（用于补足 ECC 之外的结构证据）

为避免仅用 ECC 指标过于单薄，建议在测试评估中补充 **度分布/强度分布** 的对齐指标（在与真值相同边数的 top-k 稀疏化口径下计算）：

- `deg_mae / deg_rmse / deg_pearson / deg_ks`：二值图度分布的误差、相关与 KS 距离
- `strength_mae / strength_rmse / strength_pearson / strength_ks`：加权度（节点强度）分布的误差、相关与 KS 距离

### 5.3 Mask/位置分类指标（若涉及新增边或 learned mask）

* `precision@k`, `recall@k`
* **AUPRC**（强烈建议主指标，稀疏分类比 AUROC 更合适）
* 新增边专用：

  * `new_edge_precision`, `new_edge_recall`, `new_edge_auprc`（针对 A0=0 & A1>0）

> 作用：证明“从头学位置很差”，以及“保守新增边在不增伪阳性的情况下提高召回”。

#### D2/D2′ 必须补齐的新增边评测（避免收益被淹没）

仅看全图 `sc_log_* / ecc_*` 会把新增边的价值淹没（新增边集合很小）。建议在 `test_sc_metrics.json` 中补充：

1) 新增边子集指标（核心）  
定义：`S_new={(i,j): A0=0 & A1>0}`（上三角）

- `new_edge_precision_at_knew`：在 `m0=0` 边里按 `q` 排序取 top-`K_new`，命中真新增边比例
- `new_edge_recall_at_knew`：同上，覆盖真新增边的比例
- `new_edge_auprc`：在 `m0=0` 边上用 `q` 做二分类 PR-AUC

2) 零边/新增区域误差（反映假阳性与新增区域拟合）

- `mse_zero`：在 `A1=0` 的边上，`log1p(A_pred)` 的 MSE（越小越好）
- `mse_zero_strict`：在 `A0=0 & A1=0` 的边上，`log1p(A_pred)` 的 MSE
- `mse_new_region`：在 `A0=0` 区域上 `log1p(A_pred)` vs `log1p(A1)` 的 MSE（更敏感）

### 5.4 图论拓扑指标（全局 + 节点分布相似性）

建议都在 weighted 图上（并统一阈值/稀疏化策略，保持可比）：

**全局：**

* density / total strength（总权重）
* global efficiency、local efficiency
* characteristic path length（或 harmonic mean distance）
* clustering coefficient（加权版）
* modularity Q（基于固定分区或 Louvain，注意可重复性）
* assortativity（可选）

**分布相似性（更好看，也更稳定）：**

* degree/strength 分布：KS 距离、Wasserstein 距离、相关
* betweenness / eigenvector centrality 分布相似性
* participation coefficient、within-module z-score（如果有模块）

> 作用：证明“我们保持网络组织结构稳定（不漂移）”，这是继承 mask + 小变化正则的核心卖点。

### 5.5 谱/几何一致性（非常适合“展示优势”，且不太贵）

* Laplacian 特征值谱差异：`||λ_pred - λ_true||2`
* 前 k 个特征向量对齐：subspace similarity / cosine similarity
* diffusion distance / heat kernel trace（可选）
* communicability（可选）

> 作用：强调“整体网络动力学性质被保留”，对评委很友好。

### 5.6 高阶拓扑/曲线指标（如果你已有 ECC/Euler 曲线就继续用）

* ECC（edge clustering coefficient curve）：

  * `ecc_l2`, `ecc_pearson`
* Euler characteristic curve（若你算过）：

  * `euler_l2`, `euler_pearson`
* 三角形数/聚团相关的曲线（可选）

> 作用：把“拓扑保持”讲得更高级（且与你之前框架一致）。

---

## 6. 推荐的展示形式（让优点更突出、但不需要非常严谨）

### 6.1 主表（正文一张表够）

按模型（B0/B1/A1/D2 等）列出：

* `sc_log_pearson_sparse/topk`
* `sc_log_mse`
* `mse_zero`
* `new_edge_precision/recall`（若启用新增边）
* `modularity_Q_diff`、`efficiency_diff`（选 1–2 个拓扑差异）

### 6.2 三张图（非常“讲得通”）

1. **Δt 分层性能图**：短间隔 vs 长间隔（强调 dt 门控 & residual）
2. **拓扑稳定性雷达图/条形图**：Q、效率、聚类等差异（强调不漂移）
3. **新增边 PR 曲线**：对比 D1 vs D2（强调“保守但有效”）

---

## 7. 最小可交付实验清单（DDL 版本：跑得动、结论够用）

如果你明天要交，我建议你至少跑并报告以下 8 个：

### Baselines（4 个）

* B0 Identity：`A_pred=A0`
* B1 原始 CLG-ODE（learned mask / 或你现有最接近原版）
* B2 soft mask：`A_pred=p*w`
* B3 fixed-support 但无 residual（direct y1 on m0）

### 你方法 & 消融（4 个）

* A1 Full：fixed-support + residual + dt gate + L_small（无新增边）
* B2 消融：去 dt gate
* C2 消融：去 L_small
* D2 Full+Innovation：保守新增边（候选集 + TopK + mean(q) + dt gate）

> 这 8 个足以把“继承 mask + 学变化 + 保守新增”的故事讲完整。

---

## 8. 你写结论时的“高胜率措辞模板”（建议直接抄）

* “Compared to learned masking, fixed-support inheritance substantially improves sparse/top-k alignment and prevents spurious densification.”
* “Residual learning with Δt-gating guarantees identity-consistency at Δt≈0 while allowing gradual deviations at larger Δt.”
* “The conservative innovation module introduces only a small number of new edges, improving zero-edge reconstruction and new-edge recall without compromising global topology.”
