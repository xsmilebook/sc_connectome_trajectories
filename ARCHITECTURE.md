# Project Structure

```
sc_connectome_trajectories/     # 项目根目录
├── README.md                   # 项目简介与快速入口
├── AGENTS.md                   # AI 协作与修改约束
├── PLAN.md                     # 计划与任务拆解
├── PROGRESS.md                 # 进度记录
├── ARCHITECTURE.md            # 结构说明（本文）
├── configs/                    # 全局配置中心
│
├── docs/                       # 方法、流程与决策记录（索引见 docs/README.md）
│   ├── README.md               # 文档集合索引
│   ├── workflow.md             # 可复现流程
│   ├── methods.md              # 方法学细节
│   ├── reports/                # 研究计划与阶段性总结
│   ├── sessions/               # 会话记录（yy/mm/dd.md）
│   └── notes/                  # 用户想法与自由记录
│
├── src/                        # 可复用模块（不含硬编码路径）
│
├── scripts/                    # 执行入口（HPC/主流程脚本）
│
├── data/                       # 运行时数据（不纳入版本控制）
│   ├── external/               # 外部第三方输入
│   ├── raw/                    # 原始输入（非脚本产出）
│   ├── interim/                # 中间衍生结果
│   └── processed/              # 可复用清洗结果
│
├── outputs/                    # 运行产物（不纳入版本控制）
│   ├── figures/                # 图表输出
│   ├── logs/                   # 运行日志（脚本/SLURM）
│   ├── results/                # 结果文件
│   └── tables/                 # 表格输出
│
├── notebooks/                  # 探索性分析（不纳入版本控制）
└── models/                     # 序列化模型产物（不纳入版本控制）
```
