# 方法学指令规范 (Implementation Specification)

> **文档说明**
>
> 本文档定义了 Brain Network ODE 模型的最终方法学决策。
> * **状态**：**已定稿 (Finalized / Locked)**。
> * **用途**：作为 `methods.md` 的核心内容，或作为给合作者/AI Coding Agent 的**唯一执行指令**。
> * **核心决策**：明确选择 **Topo 方案 1**（拓扑仅作 Conditioning/评估，不作为训练 Loss）。

---

## 1. 总体目标 (General Objective)

*   **主任务**：纵向预测（Forecasting）。给定任一访视（Visit）的结构连接（SC）与形态学信息（Morphology），预测未来任意时间点的大脑网络状态。
*   **隐式目标**：模型需隐式学习按绝对年龄对齐的群体性发育轨迹，但不将其作为显式的主监督目标。

---

## 2. 时间变量与发育建模 (Time & Development)

### 2.1 时间定义（锁定）
模型使用 **相对时间（Delta Time）** 作为 ODE 的积分变量，而非绝对年龄。

*   **ODE 时间变量 $t$**：
    $$
    t = \Delta t = \text{age}(t) - \text{age}_0
    $$
    其中 $\text{age}_0$ 为当前预测起点（Baseline 或任一作为起点的访视）的绝对年龄。
*   **初始状态**：每个被试的 ODE 积分总是从 $t_0 = 0$ 开始。

### 2.2 年龄信息分工
为了避免重复建模并利用年龄信息：
1.  **绝对年龄不作为 ODE 的积分变量 $t$**。
2.  **绝对年龄 $\text{age}_0$ 作为显式协变量**输入动力学函数：
    $$
    \frac{dZ}{dt} = f\big(Z(t); \ \text{age}_0, \ \text{sex}, \ \text{site}, \ \dots\big)
    $$

### 2.3 群体发育轨迹的获取
群体轨迹不作为训练的直接 Loss，而是通过推断生成：
*   对任意目标绝对年龄 $\text{age}^*$，计算 $\Delta t^* = \text{age}^* - \text{age}_0$。
*   从起始状态 $Z(0|\text{age}_0)$ 积分至 $\Delta t^*$。
*   在群体水平上汇总预测结果（计算均值或分位数），形成“生长曲线式”的轨迹 $\mu(\text{age}^*)$。

---

## 3. 结构连接 (SC) 预处理策略

### 3.1 数据事实与决策
*   **数据特性**：SC 矩阵为硬稀疏（Hard Sparse），约 83.5% 元素为精确 0；非零权重均 $>0.01$，不存在微小噪声边。
*   **决策（锁定）**：**不进行任何额外的 Hard 稀疏化**（禁用 Top-K 或额外阈值截断），以保留所有潜在的发育信号。

### 3.2 数值处理流程
1.  **对称化**：
    $$ A \leftarrow \frac{A + A^\top}{2} $$
2.  **去对角**：置 $A_{ii} = 0$。
3.  **权重变换**（Log domain）：
    $$ A_{\log} = \log(1 + A) $$

### 3.3 强度信息的处理
*   **归一化策略**：**不对 SC 做被试级（Subject-level）的强制总强度归一化**（如除以总和）。
*   **强度协变量提取（锁定）**：基于**原始权重域**（对称化 + 去对角后的 $A$，不取 log）显式计算全局强度标量 $s$：
    $$ s = \log\left(\sum_{i<j} A_{ij} + \epsilon\right) $$
*   **可选强度补充（可开关）**：可额外提供“正边平均强度”摘要（per-edge strength）：
    $$ s_{\text{mean}} = \log\left(\mathrm{mean}\big(A_{ij}\mid A_{ij}>0\big) + \epsilon\right) $$
*   **注入方式**：将 $s$ 作为协变量输入动力学函数 $f(\cdot)$ 或 Decoder。
    *   *目的*：保留强度随发育变化的真实信息，同时防止强度差异主导图结构的表征学习。

---

## 4. 形态学特征 (Morphology)

### 4.1 预处理规则（锁定）
*   **Z-score 标准化**：所有形态学特征必须进行 Z-score 处理。
*   **防泄漏机制**：Z-score 的均值和标准差**仅使用训练集（Training Set）统计量**计算：
    $$ x_z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}} $$
*   **统计粒度（锁定）**：按 **(ROI, Metric)** 的列粒度独立标准化（即每个 ROI×指标列分别估计 $\mu_{\text{train}},\sigma_{\text{train}}$）。

### 4.2 体积类特征特别处理
*   若存在 ICV (Intracranial Volume) 或 TIV 数据：
    1.  先做物理归一化（如 $\text{Volume} / \text{ICV}$）。
    2.  再做上述 Z-score 处理。

---

## 5. 协变量处理策略 (Covariates)

### 5.1 主策略（锁定）
*   **禁用 ComBat**：**不使用 ComBat / ComBat-GAM 作为主流程预处理**。ComBat 仅用于后续的敏感性分析或对照实验。
*   **端到端注入**：协变量直接作为条件输入模型。
    *   **本版本锁定协变量集合**：$\text{age}_0$（起点绝对年龄）、Sex、Site，以及 SC 全局强度标量 $s$（见 3.3）。
    *   其他扩展协变量（如 Motion/ICV/SES）保留为后续增强项，不纳入本版本实现。

### 5.2 注入位置
*   **必须注入**：动力学函数 $f(\cdot)$（影响 $dZ/dt$）。
*   **可选注入**：
    *   Encoder：影响初始状态 $Z(0)$。
    *   Decoder：用于建模条件均值或方差。

---

## 6. 模型架构与损失函数

### 6.1 编码器 (Encoder)
*   **输入**：使用被试特定（Subject-specific）的 $A_{t_0}$ 进行消息传递（Message Passing）。
*   **架构**：GNN (GCN 或 GraphSAGE)，层数推荐 2 层。
*   **输出**：从 GNN 输出分支得到两个潜在表示：
    *   $Z_{\text{morph}}(0)$
    *   $Z_{\text{conn}}(0)$

### 6.2 解码器 (Decoder)
采用连续权重回归策略：
*   **重建公式**：
    $$ \hat{A} = \text{Softplus}(Z_{\text{conn}} Z_{\text{conn}}^\top) $$
*   **共享打分与可学习标定（锁定）**：存在性与强度回归共享同一个内积分数 $s_{ij}=z_i^\top z_j$，但通过可学习标定参数映射到不同输出：
    $$
    p_{ij}=\sigma\big(\alpha(s_{ij}-\delta)\big),\quad
    \hat{w}_{ij}=\mathrm{softplus}(\gamma s_{ij}+\beta)
    $$
    默认初始化：$\alpha=10,\ \delta=0,\ \gamma=1,\ \beta=0$。
*   **结构与强度分离的重建损失（锁定）**：为避免大量零边主导训练，使用“两项式”损失：
    $$
    \mathcal{L}=\mathcal{L}_{\text{edge}}+\lambda_w\mathcal{L}_{\text{weight}},\quad \lambda_w=1.0
    $$
    1) **边存在性损失** $\mathcal{L}_{\text{edge}}$：对上三角（不含对角）进行**加权 BCE**（pos:neg = 5:1）。
    2) **边权强度损失** $\mathcal{L}_{\text{weight}}$：仅在真实为正的边集合上，对 $\log(1+A)$ 域的正边权重做 **Huber 回归**。
    *   Mask 范围：仅计算上三角区域（不含对角线）。

---

## 7. 拓扑信息 (Topology) - 方案 1

### 7.1 核心原则（锁定）
*   **角色**：拓扑特征仅作为 **Conditioning（条件输入）** 和 **Evaluation（评估解释）**。
*   **训练状态**：**拓扑不作为主训练损失函数**。
*   **禁用方法**：训练过程中**不计算** PD-Wasserstein 距离或 PDM Loss。

### 7.2 拓扑表示的获取
*   **输入域（锁定）**：对 $A_{\log}=\log(1+A)$ 进行拓扑摘要计算。
*   **Filtration（锁定）**：对 $A_{\log}$ 进行 **Edge-weight superlevel filtration**：
    $$ G(\tau) = \{ (i,j) \mid (A_{\log})_{ij} \ge \tau \} $$
*   **阈值序列（锁定）**：仅使用 $A$ 的**非零边**（$A_{ij}>0$）权重集合，在 $A_{\log}$ 域上取 $K=32$ 个分位数阈值：
    $$ q\in[0.05,0.95]\ \text{等距取点},\quad \tau_k=\text{Quantile}\big((A_{\log})_{ij}\mid A_{ij}>0;\ q_k\big) $$
*   **默认拓扑向量（锁定）**：Euler characteristic curve（欧拉示性数曲线），用 clique complex 的低阶近似：
    $$
    \mu(A) = \big[\chi(\tau_1),\ldots,\chi(\tau_K)\big],\quad
    \chi(\tau)\approx V - E(\tau) + T(\tau)
    $$
    其中 $V=N$，$E(\tau)$ 为阈值图的边数，$T(\tau)$ 为阈值图的三角形数量（2-simplices）。
*   **三角形计数细则（锁定）**：对每个阈值 $\tau$，先将 $A_{\log}$ 二值化为无向简单图
    $$ B(\tau)=\mathbb{1}[A_{\log}\ge\tau] $$
    忽略权重，仅在 clique complex 下统计 2-simplices；实现上使用稀疏矩阵乘法进行三角形计数（例如基于 $B$ 的乘法与迹/按边局部计数聚合）。
*   **标准化（锁定）**：对 $\mu(A)$ 的每个 $\tau$ 维度使用训练集统计量做标准化（防泄漏）。

### 7.3 使用方式
1.  **预计算**：在数据加载阶段预计算拓扑向量 $\mu(A)$（作为每个访视的全局特征）。
2.  **条件注入**：将 $\mu(A)$ 作为全局特征向量（Global Condition）输入模型；默认注入动力学函数 $f(\cdot)$，可选同步注入 Encoder/Decoder。

---

## 8. 训练目标与采样策略

### 8.1 训练任务（Forecasting）
采用 **多起点预测（Multi-start Forecasting）** 策略：
1.  随机采样同一被试的一对访视 $(i, j)$，满足时间顺序 $j > i$。
2.  输入：$t_0$ 时刻的数据 $(A_i, X_i)$。
3.  积分：从 $Z_i(0)$ 积分至 $\Delta t = \text{age}_j - \text{age}_i$。
4.  监督：重建 $t_j$ 时刻的数据 $(\hat{A}_j, \hat{X}_j)$ 并计算 Loss。
5.  **配对采样混合（锁定）**：70% 概率采样相邻访视对（adjacent），30% 概率在所有可用 $(i<j)$ 对中均匀采样。

### 8.2 缺失值处理
*   利用 ODE 的连续性天然处理不规则采样。
*   **不进行**伪造时间点的插补作为训练目标。
