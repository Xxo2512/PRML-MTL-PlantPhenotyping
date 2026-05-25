# 面向植物表型视觉的异质同构多任务学习

> 华中科技大学 人工智能与自动化学院 — 2026 春《模式识别与机器学习》课程设计 选题

---

## 阶段 I 进度（W10–W12）

| 交付物 | 路径 | 主笔 | 状态 |
|---|---|---|---|
| 文献综述（MTDNN + 5 种 MTL 方法 + 共性差异表 + head 补齐） | [`docs/literature_review.md`](docs/literature_review.md) | A | ✅ |
| 系统设计稿 v1.0（架构、调度、接口、loss、训练流程、消融计划） | [`docs/design.md`](docs/design.md) | A | ✅ |
| 接口契约 | [`docs/api_contract.md`](docs/api_contract.md) | A | ✅ |
| 配置模板（base + 5 方法 + 消融 + 单任务） | [`configs/`](configs/) | A | ✅ |

**冻结项**：
- 任务键 `TASKS = ('seg', 'det', 'cnt', 'cls')`
- `SwinBackbone` / `InputAdapter` / `BaseTaskHead` / `MTLModel` / `CrossDatasetSampler` 签名
- yaml 顶层 schema

---

## 1. 项目概述

在植物表型视觉分析中，**分割 / 检测 / 计数 / 分类**是四大核心任务。现有多任务学习（MTL）方法多假设 *任务共享同一张输入图像* ，但在实际作物监测场景中，不同任务通常依赖**不同数据集**和**不同图像输入**——任务异质、数据异源，使得传统 MTL 难以直接迁移。

本项目以小麦视觉为研究对象，构建一个 **跨数据集 (cross-dataset) 多任务学习框架**：通过 MTDNN[1] 风格的数据调度方式让分割、检测、计数、分类四个任务在共享骨干 (Swin Transformer tiny / small) 上协同训练，并对 **5 种主流 MTL 方法** 进行适配与对比分析。

### 1.1 待复现方法
| 编号 | 方法 | 出处 | 关键思想 |
|---|---|---|---|
| M1 | **MTLoRA**[2] | CVPR 2024 | 低秩适配的高效 MTL |
| M2 | **TADFormer**[3] | CVPR 2025 | 任务自适应动态 Transformer |
| M3 | **DiTASK**[4] | CVPR 2025 | Diffeomorphic 多任务微调 |
| M4 | **TaskPrompter**[5] | ICLR 2023 | 空间-通道任务提示 |
| M5 | **PGT**[6] | TMM 2024 | Prompt-Guided Transformer |

### 1.2 任务-头-指标 对照
| 任务 | 数据集 | Task Head | 评测指标 |
|---|---|---|---|
| 分割 | Wheat Organ Segmentation | DPT[8] | mIoU, mAcc |
| 检测 | Wheat Head Detection (ground-based) | FCOS[7] | AP, AP50 |
| 计数 | Wheat Leaf Counting | PET[10] / FoMo4Wheat[9] 风格头 | MAE, RMSE, R² |
| 分类 | Wheat Disease Classification | MLP | mAP, BA |

---

## 2. 课程时间线 (第 10–19 周)

| 周次 | 阶段 | 关键交付物 | 状态门 |
|---|---|---|---|
| W10 | 需求分析 + 调研 | 调研报告 v0.1、数据集申请记录、文献综述笔记 | 组内评审 |
| W11 | 方案设计 | `docs/design.md`（架构图、调度方式、接口定义） | 提交指导老师 |
| W12 | 方案修改 | `docs/design.md` v1.0、API 冻结、`configs/` 模板 | 老师审阅通过 |
| W13 | 算法实现 (1) | 数据 pipeline + 共享 backbone + 4 个 task head 单任务 baseline 跑通 | 单任务指标达标 |
| W14 | 算法实现 (2) | 跨数据集调度器 (MTDNN-style) + 5 种 MTL 方法骨架 | 至少 2 种方法可前向 |
| W15 | 算法实现 (3) | 5 种 MTL 方法全部完成训练对接，统一日志/checkpoint | 全部方法可训练 |
| W16 | 系统测试 (1) | 完整训练 5 种方法，落表对比；消融实验设计 | 主表初稿 |
| W17 | 系统测试 (2) | 消融、可视化、超参回扫；Demo 脚本 | 结果冻结 |
| W18 | 报告撰写 | 课程设计报告 v1.0 + 演示 PPT + 视频 Demo | 组内交叉互审 |
| W19 | 答辩 | 答辩 PPT 终稿、Q&A 预演记录 | **答辩** |

> 关键点：**W12 设计冻结 → W15 全方法可训 → W17 结果冻结 → W19 答辩**。

---

## 3. 五人分工

组员为 **A / B / C / D / E**（A 为组长）。

| 角色 | 主责 MTL 方法 | 框架职能（共同代码贡献） | 报告章节主笔 |
|---|---|---|---|
| **A — 组长-架构** | **MTLoRA** | 共享骨干 (Swin)、训练主循环、cross-dataset 调度器、CI/规范 | 引言、系统架构、总结 |
| **B — 数据 & 检测** | **TADFormer** | 4 个数据集 dataloader & 增广、FCOS 检测头、mAP 评估 | 数据章节、检测实验 |
| **C — 分割 & 可视化** | **DiTASK** | DPT 分割头、可视化工具、TensorBoard 看板 | 分割实验、可视化分析 |
| **D — 计数 & 评测** | **TaskPrompter** | 计数头 (PET-style)、统一 metric 模块、消融驱动 | 计数实验、消融分析 |
| **E — 分类 & 部署** | **PGT** | 分类头、checkpoint/日志规范、推理 Demo、答辩 PPT | 分类实验、相关工作 |

### 协作约定
- 每人独立分支 `feat/<name>-<module>`，所有合入 `main` 必须经组长 (A) review。
- 共享代码（backbone / dataloader / metric）合入 `main` 后其他人 rebase；MTL 方法各自在 `models/mtl/<method>.py` 维护。
- 实验结果统一写入 `logs/results.csv`，由 D 负责合并。

---

## 4. 详细工作分解 (WBS)

### 阶段 I：调研与方案设计（W10–W12）
- [ ] 文献阅读：每人精读所主责方法原论文 + 1 篇相关综述
- [ ] 数据集获取：B 联系指导老师拿到 4 个小麦数据集，统计样本数 / 分辨率 / 标注格式
- [ ] 设计 cross-dataset 调度策略：参考 MTDNN[1]，决定 *按 batch 轮转* vs *按 step 加权采样*（A 主导）
- [ ] 接口冻结：`Backbone -> Adapter -> TaskHead` 的输入输出 shape、loss 接口
- [ ] 输出 `docs/design.md`，提交指导老师审阅 → 修订定稿

### 阶段 II：算法实现（W13–W15）
- [ ] **W13** 单任务 baseline：每个 task head 在自己数据集上单独训练跑通，作为后续 MTL 上限/下限对照
- [ ] **W14** 框架打通：cross-dataset sampler、共享 backbone forward、loss 加权（unweighted / DWA / Uncertainty 三选一作为默认）
- [ ] **W15** 五方法适配（每人一种）：保持原方法网络结构尽量不变，仅替换/补齐缺失的 4 个 task head；对 Swin tiny/small 预训练权重统一加载

### 阶段 III：实验与对比（W16–W17）
- [ ] 主表实验：5 方法 × 4 任务 = 20 组指标，加 4 个单任务 baseline 作 reference
- [ ] 消融 (D 主导，全员配合，每人至少 1 项)：
  - 调度策略（轮转 vs 加权）
  - Loss 加权方式（uniform vs DWA vs Uncertainty）
  - Backbone 规模（tiny vs small）
  - 输入分辨率
  - 任务子集（去掉某一任务对其它任务的影响 → 异质性分析）
- [ ] 可视化 (C)：分割 mask、检测框、计数密度图、Grad-CAM 分类
- [ ] 推理 Demo (E)：单脚本完成"输入图像 → 4 任务输出"

### 阶段 IV：报告与答辩（W18–W19）
- [ ] 课程设计报告：按章节主笔 → 组长 (A) 整稿 → 全员交叉互审
- [ ] 答辩 PPT (E 主笔，A 复核)：方法图 / 主表 / 关键消融 / Demo 视频
- [ ] 答辩预演 1 次，记录预期问答清单

---

## 5. 仓库结构

```
PRML-MTL-PlantPhenotyping/
├── configs/          # 各方法 / 各实验的 yaml 配置
├── data/             # 数据软链接（数据本体不入仓）
├── datasets/         # 4 个任务的 Dataset / 跨数据集 Sampler
├── models/
│   ├── backbone/     # Swin tiny / small
│   ├── heads/        # FCOS / DPT / MLP / PET-style
│   └── mtl/          # mtlora.py / tadformer.py / ditask.py / taskprompter.py / pgt.py
├── scripts/          # 训练 / 评测 / 可视化 / 数据准备
├── utils/            # metrics / loss / logging / scheduler
├── checkpoints/      # 训练产物（.gitignore）
├── logs/             # TensorBoard + results.csv
├── docs/             # design.md / report.md / 答辩 PPT
├── train.py
├── evaluate.py
├── test.py
└── requirements.txt
```

---

## 6. 参考文献

[1] Liu et al. *Multi-task Deep Neural Networks for Natural Language Understanding.* ACL 2019.
[2] Agiza, Neseem, Reda. *MTLoRA: Low-rank Adaptation Approach for Efficient Multi-task Learning.* CVPR 2024.
[3] Baek et al. *TADFormer: Task-Adaptive Dynamic Transformer for Efficient Multi-task Learning.* CVPR 2025.
[4] Mantri et al. *DiTASK: Multi-task Fine-tuning with Diffeomorphic Transformations.* CVPR 2025.
[5] Ye, Xu. *TaskPrompter: Spatial-Channel Multi-task Prompting for Dense Scene Understanding.* ICLR 2023.
[6] Lu et al. *Prompt Guided Transformer for Multi-task Dense Prediction.* IEEE TMM 2024.
[7] Tian et al. *FCOS: A Simple and Strong Anchor-free Object Detector.* TPAMI 2020.
[8] Ranftl, Bochkovskiy, Koltun. *Vision Transformers for Dense Prediction (DPT).* ICCV 2021.
[9] Han et al. *FoMo4Wheat: Toward Reliable Crop Vision Foundation Models with Globally Curated Data.* arXiv:2509.06907, 2025.
[10] Liu, Lu, Cao, Liu. *Point-Query Quadtree for Crowd Counting, Localization, and More (PET).* ICCV 2023.
