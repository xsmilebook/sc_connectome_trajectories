# AI Agent 使用 GPU 开展深度学习的集群规范指南

（北京脑中心 HPC · Singularity · PyTorch / GNN）

---

## 1. 文档适用对象与目标

### 适用对象

* 自动化 **AI agent / LLM agent**
* 需要在集群上 **自主提交 GPU 深度学习任务**
* 使用 **容器化环境（Singularity）**
* 无 root / 无管理员权限

### 目标

AI agent 能够：

1. 正确识别 **集群可用 GPU 类型与队列**
2. 合理选择 **CUDA / PyTorch / GNN 版本**
3. 通过 **Slurm + Singularity** 申请并使用 GPU
4. 避免常见违规与失败模式（资源、网络、环境）

---

## 2. 集群 GPU 资源概览（AI 必须掌握）

根据集群手册 **第 1 章 & 第 4.5 节**，当前可用于深度学习的 GPU 资源如下 ：

### 2.1 GPU 节点与队列

| 队列名       | GPU 类型              | GPU 数  | 单 GPU 显存 | CPU / GPU | 内存 / GPU | 适用任务   |Hostname|
| --------- | ------------------- | ------ | -------- | --------- | -------- | ------------- | --- |
| `q_ai4`   | NVIDIA Tesla V100   | 4 / 节点 | 32 GB    | 9 cores   | 126 GB   | 中等规模 DL / GNN | ai03, ai04,ai06|
| `q_ai8`   | NVIDIA Tesla V100   | 8 / 节点 | 32 GB    | 4 cores   | 108 GB   | 多卡训练 / DDP    | ai01,ai02|
| `q_gpu_c` | NVIDIA RTX 6000 Ada | 8 / 节点 | 48 GB    | 32 cores  | 183 GB   | 大模型 / 高显存     | 暂无  |

> ⚠️ **AI agent 只能在以上队列申请 GPU**，不能在 `q_cn / q_fat` 等 CPU 队列请求 GPU。优先使用`q_ai4`

---

## 3. AI Agent 的核心约束（必须遵守）

### 3.1 资源申请规则（硬约束）

* GPU 必须通过 Slurm 申请：

  ```bash
  --gres=gpu:<N>
  ```
* **禁止**在登录节点直接运行深度学习程序
* **禁止**绕过 Slurm 直接占用 GPU

### 3.2 GPU = CPU + 内存的“捆绑资源”

集群采用 **GPU 驱动型资源分配**（手册 4.5 节）：

> 申请 GPU 时，CPU 和内存会自动分配，不需要、也不应手动指定 `-c / --mem`

示例（正确）：

```bash
srun -p q_ai4 --gres=gpu:1 ...
```

示例（不推荐）：

```bash
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
```

---

## 4. 容器与 CUDA / PyTorch 的选择原则（AI 决策逻辑）

### 4.1 核心兼容性原则（AI 必须内化）

> **容器中的 CUDA Runtime ≤ 集群 GPU Driver 支持的 CUDA**

由于：

* 集群 GPU Driver 由管理员统一维护
* 用户无法升级 driver

**推荐策略（稳定优先）：**

| 项目      | 推荐                               |
| ------- | -------------------------------- |
| CUDA    | 11.8（最稳）                         |
| PyTorch | 2.0 – 2.2                        |
| cuDNN   | 随官方镜像                            |
| GNN     | PyTorch Geometric / DGL 官方 wheel |

---

## 5. AI Agent 推荐的容器策略（强制）

### 5.1 构建策略（AI 不能在集群现编环境）

**禁止行为**：

* 在计算节点 `pip install torch-scatter`
* 在集群上编译 CUDA extension

**唯一推荐流程**：

```text
Dockerfile（本地 / remote）
   ↓
singularity build --remote
   ↓
torch_gnn.sif
   ↓
Slurm + singularity exec --nv
```

---

## 6. AI Agent 的标准运行模板（GPU 作业）

### 6.1 Slurm 作业模板（单 GPU）

```bash
#!/bin/bash
#SBATCH -J gnn_train
#SBATCH -p q_ai4
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o outputs/logs/%j.out
#SBATCH -e outputs/logs/%j.err

module load singularity

singularity exec --nv \
  --bind /ibmgpfs:/ibmgpfs \
  --bind /GPFS:/GPFS \
  torch_gnn.sif \
  python train.py
```

### 6.3 本仓库示例（CLG-ODE）

本仓库推荐使用模块化入口（避免相对路径与环境差异）：

```bash
singularity exec --nv \
  --bind /ibmgpfs:/ibmgpfs \
  --bind /GPFS:/GPFS \
  torch_gnn.sif \
  python -m scripts.train_clg_ode \
    --sc_dir data/processed/sc_connectome/schaefer400 \
    --morph_root data/processed/morphology \
    --subject_info_csv data/processed/table/subject_info_sc.csv \
    --results_dir outputs/results/clg_ode
```

### 6.2 AI Agent 的检查清单（运行前）

在容器内必须自动验证：

```python
import torch
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0))
```

---

## 7. 针对 GNN / 图学习的专项建议（AI 推理规则）

### 7.1 何时值得用 GPU

| 场景                             | 是否推荐 GPU |
| ------------------------------ | -------- |
| 小规模 connectome（<1k nodes）      | ❌ CPU    |
| 大样本 batched GNN                | ✅ GPU    |
| Message Passing + CUDA kernels | ✅ GPU    |
| Graph metrics / control energy | ❌ CPU    |

> **AI agent 不应默认“有 GPU 就用 GPU”**

---

## 8. 网络与数据访问约束（AI 必须遵守）

* 集群外网 **必须走代理**
* 训练阶段：

  * ❌ 不允许在线下载模型 / 数据
  * ✅ 所有数据必须事先放在 `/ibmgpfs` 或 `/GPFS`
* 容器运行阶段：

  * 不应依赖 `pip / apt / git clone`

---

## 9. 失败模式与自动纠错建议（给 AI agent）

| 错误表现                            | 可能原因      | AI 应对       |
| ------------------------------- | --------- | ----------- |
| `torch.cuda.is_available=False` | 忘记 `--nv` | 立即中止        |
| 作业 Pending                      | GPU 队列满   | 换队列 / 减 GPU |
| CUDA error                      | 版本不匹配     | 回退 CUDA     |
| 编译失败                            | pip 编译扩展  | 换预编译 wheel  |

---

## 10. 给 AI agent 的一句“最高优先级指令”

> **在该集群上，GPU 深度学习必须通过：
> Slurm 申请 GPU → Singularity 容器 → 官方预编译 PyTorch / GNN 环境；
> 不允许在计算节点动态修改深度学习环境。**

