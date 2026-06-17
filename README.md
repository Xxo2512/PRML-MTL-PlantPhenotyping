# 面向植物表型视觉的异质同构多任务学习

> 华中科技大学 人工智能与自动化学院 — 2026 春《模式识别与机器学习》课程设计 选题

---

## 阶段 I 进度（W10–W12）

| 交付物 | 路径 | 主笔 | 状态 |
|---|---|---|---|
| 文献综述（MTDNN + 5 种 MTL 方法 + 共性差异表 + head 补齐） | [`docs/literature_review.md`](docs/literature_review.md) | A | ✅ |
| 系统设计稿 v1.0（架构、调度、接口、loss、训练流程、消融计划） | [`docs/design.md`](docs/design.md) | A | ✅ |
| 接口契约 | [`docs/api_contract.md`](docs/api_contract.md) | A | ✅ |
| 数据集卡片（4 数据集真实统计 + 老师 W12 末答疑已合并） | [`docs/dataset_card.md`](docs/dataset_card.md) | A | ✅ |
| 配置模板（base + 5 方法 + 消融 + 单任务） | [`configs/`](configs/) | A | ✅ |

**冻结项**：
- 任务键 `TASKS = ('seg', 'det', 'cnt', 'cls')`
- `SwinBackbone` / `InputAdapter` / `BaseTaskHead` / `MTLModel` / `CrossDatasetSampler` 签名
- yaml 顶层 schema

---

## 阶段 II 进度（W13–W15）

### 已完成

| 模块 | 路径 | 主笔 | 说明 |
|---|---|---|---|
| 4 个 Dataset | [`datasets/wheat_{cls,seg,det,cnt}.py`](datasets/) | A/B | cls=ImageFolder, seg=mask PNG, det=YOLO txt, cnt=VOC XML (叶尖) |
| InputAdapter | [`datasets/input_adapter.py`](datasets/input_adapter.py) | A | 4 任务统一归一化到 `[3,384,384]`，keep-ratio resize + 同步 box/point 变换 |
| CrossDatasetSampler | [`datasets/cross_dataset.py`](datasets/cross_dataset.py) | A | RR / PS (α=0.5) / HM 三种调度 |
| Swin-T Backbone | [`models/backbone/swin.py`](models/backbone/swin.py) | A | timm `swin_tiny_patch4_window7_224`，自动走 hf-mirror |
| FCOS det head | [`models/heads/det_fcos.py`](models/heads/det_fcos.py) | B | 多尺度 FPN + centerness + Focal/IoU/BCE |
| DPT seg head | [`models/heads/seg_dpt.py`](models/heads/seg_dpt.py) | C | Reassemble + Fusion blocks，含可视化接口 |
| PET-style cnt head | [`models/heads/cnt_pet.py`](models/heads/cnt_pet.py) | D | quadtree 划分 + 点查询匈牙利匹配 |
| MLP / PGT cls head | [`models/heads/cls_mlp.py`](models/heads/cls_mlp.py), [`cls_pgt.py`](models/heads/cls_pgt.py) | E | 支持 ordinal CE（生育期 6 类有序）|
| vanilla MTLModel | [`models/mtl/base.py`](models/mtl/base.py) | A | 共享 backbone + 独立 head 基线 |
| **MTLoRA** | [`models/mtl/mtlora.py`](models/mtl/mtlora.py) | A | TA-LoRA + TS-LoRA 注入 Swin 的 qkv/proj/fc1/fc2（48 层 LoRA, rank=16, ~5.65M LoRA 参数）|
| **TADFormer** | [`models/mtl/tadformer.py`](models/mtl/tadformer.py) | B | 任务嵌入 + 每 block 动态调制 (γ, β) |
| **DiTASK** | [`models/mtl/ditask.py`](models/mtl/ditask.py) | C | backbone 输出后挂可逆微分同胚映射 Φ_t |
| **PGT** | [`models/mtl/pgt.py`](models/mtl/pgt.py) | E | 每 block prompt + prompt-guided cross-attention |
| LossAggregator | [`utils/losses.py`](utils/losses.py) | A | `uniform` + Kendall'18 `uncertainty` 可学权重 |
| Visualize | [`utils/visualize.py`](utils/visualize.py) | C | seg mask 叠加 / det boxes / cnt 热图 / cls Grad-CAM → TensorBoard |
| Metrics | [`utils/metrics.py`](utils/metrics.py) | D | seg mIoU/mAcc, det AP/AP50, cnt MAE/RMSE/R², cls mAP/BA |
| Logger | [`utils/logger.py`](utils/logger.py) | E | TensorBoard + `logs/results.csv` |
| 训练入口 | [`train.py`](train.py) | A | smoke / 短训 / single-task baseline / 全量 通用；支持 `--tag` / `--single_task` / `--save_every` |
| 评测入口 | [`evaluate.py`](evaluate.py) | A | 按 ckpt+config 跑 4 任务指标，结果落 `logs/results.csv` |
| 实验脚本 | [`scripts/run_a_experiments.sh`](scripts/run_a_experiments.sh) | A | 服务器一键 single×4 + vanilla + mtlora 训练+评测 |

### 待办

- M2 (TADFormer) / M3 (DiTASK) / M5 (PGT) 已合入主干，但**尚未进入主表对比训练**（A 的 `run_a_experiments.sh` 当前只覆盖 vanilla + MTLoRA）；W15 末由各方法主笔追加自己的 `run_<x>_experiments.sh`，复用同样的 ckpt/eval 流程。
- M4 (TaskPrompter) — `models/mtl/taskprompter.py` 尚未实现（D 主笔）。

### W15 主表实验（A 部分已完成 @ AutoDL RTX 4090, 2026-06-17）

`bash scripts/run_a_experiments.sh` 已跑完，结果落 `logs/results.csv`。

| # | exp tag | config | scope | epochs | 状态 |
|---|---|---|---|---|---|
| 1 | `single_seg_5ep` | vanilla.yaml | seg only | 5 | ✅ |
| 2 | `single_det_5ep` | vanilla.yaml | det only | 5 | ✅ |
| 3 | `single_cnt_5ep` | vanilla.yaml | cnt only | 5 | ✅ |
| 4 | `single_cls_5ep` | vanilla.yaml | cls only | 5 | ✅ |
| 5 | `vanilla_10ep_ps` | vanilla.yaml | 4 任务 PS 调度 | 10 | ✅ |
| 6 | `mtlora_20ep_ps` | mtlora.yaml | 4 任务 PS 调度 | 20 | ✅ |

> **W14 sampler 修复**：依据老师反馈，PS 模式由"每步独立 P(t)∝|D_t|^α 抽样"改为"构造 α-balanced 任务序列 + shuffle"，epoch 长度从 `max(sizes)` 改为 `Σ|D_t|`。当前 α=0.5 下 4 任务 epoch_len=8367 (bs=8)，每 epoch 各 task 步数 = seg 627 / det 1343 / cnt 868 / cls 5529（小任务被多次复用、cls 被随机欠采，避免 cls 一家独大）。详见 [`datasets/cross_dataset.py`](datasets/cross_dataset.py)。

> **W15 指标/loss/head 修复**（2026-06-17）：原实现有 5 处问题：
> - `cls/mAP` 每 batch 算后平均，退化到 ~1/C 随机水平 → 改为跨 batch 累积后整体算
> - `det/AP` == `det/AP50`（placeholder）→ 改为 COCO 风格 10 个 IoU 阈值平均
> - `cnt/R²` 用 batch.mean() 作为 ss_tot → batch 太小爆负值，改为全集 mean
> - `cnt head` 第一版 (density+MSE×1000) 让模型收敛到"输出全零"退化解 → 第二版重平衡 (MSE+L1)
> - `cnt_pet.py` 名为 PET 但实现是 density regression（老师指出）→ **第三版改写为真 PET：N=256 learnable point queries + Transformer decoder + Hungarian matching + CE/SmoothL1 loss**，遵循 Liu et al. ICCV 2023
>
> 详见 `utils/metrics.py` / `models/heads/cnt_pet.py`。

### 主表（PET cnt head + 修指标后的最终版本）

| Method (tag) | seg/mIoU↑ | seg/mAcc↑ | det/AP50↑ | det/AP@[.5:.95]↑ | cnt/MAE↓ | cnt/RMSE↓ | cnt/R²↑ | cls/acc↑ | cls/mAP↑ | cls/BA↑ |
|---|---|---|---|---|---|---|---|---|---|---|
| single_seg_5ep | **0.634** | 0.708 | — | — | — | — | — | — | — | — |
| single_det_5ep | — | — | **0.791** | **0.397** | — | — | — | — | — | — |
| single_cnt_5ep (PET) | — | — | — | — | 23.41 | 28.84 | 0.522 | — | — | — |
| single_cls_5ep | — | — | — | — | — | — | — | 0.986 | **0.9997** | 0.986 |
| vanilla_10ep_ps | 0.670 | 0.767 | **0.803** | **0.395** | 15.92 | 21.17 | 0.742 | 0.995 | 0.9998 | 0.995 |
| **mtlora_20ep_ps** | **0.677** | **0.775** | 0.773 | 0.354 | **14.22** | **18.95** | **0.794** | **0.998** | **1.0000** | **0.998** |

### Δm 综合指标（Maninis et al. CVPR 2019，老师建议）

公式：$\Delta m = \frac{1}{T} \sum_{t=1}^{T} (-1)^{l_t} \cdot \frac{M_t^{\text{MTL}} - M_t^{\text{STL}}}{M_t^{\text{STL}}}$（$l_t=1$ 若指标越低越好，否则 0）

由 `scripts/compute_delta_m.py` 自动从 `logs/results.csv` 计算：

| 方法 | seg/mIoU | det/AP50 | cnt/MAE | cls/acc | **Δm** |
|---|---|---|---|---|---|
| **mtlora_20ep_ps** | +6.72% | -2.34% | +39.28% | +1.21% | **+11.22%** ← 最优 |
| vanilla_10ep_ps | +5.61% | +1.43% | +32.03% | +0.86% | +9.98% |

两种方法 Δm 都为正 → MTL 综合优于单任务基线；MTLoRA 综合提升幅度高出 Vanilla 约 1.24 个百分点。

### 迁移效应（diagonal 单任务 vs MTL）

| Task | 单任务 baseline (PET 5ep) | Vanilla MTL | MTLoRA | 迁移结论 |
|---|---|---|---|---|
| seg / mIoU | 0.634 | 0.670 (+3.6) | **0.677 (+4.3)** | ✅ **正迁移**，多任务帮助分割 |
| seg / mAcc | 0.708 | 0.767 (+5.9) | **0.775 (+6.7)** | ✅ **正迁移** |
| det / AP50 | 0.791 | **0.803 (+1.2)** | 0.773 (-1.8) | ⚠️ Vanilla 略好；MTLoRA 轻微负迁移 |
| det / AP@.5:.95 | **0.397** | 0.395 (-0.2) | 0.354 (-4.3) | ⚠️ MTLoRA 在严格 IoU 下显著弱 |
| cnt / MAE | 23.41 (PET 欠训练) | 15.92 (-7.5) | **14.22 (-9.2)** | ✅ **强正迁移**，-39% |
| cnt / R² | 0.522 | 0.742 (+0.22) | **0.794 (+0.27)** | ✅ **强正迁移** |
| cls / acc | 0.986 | 0.995 | **0.998** | 接近饱和, 信息量低 |

### 关键发现

1. **MTLoRA Δm = +11.22% 超过 Vanilla +9.98%**：MTLoRA 用 44% 可训练参数 (17.4M vs 39.3M) 反而综合表现更好。
2. **cnt 单项贡献 ~88% 的 Δm**：mtlora cnt +39.28% / 总 Δm 11.22% × 4 = cnt 占 9.82 / 11.22 ≈ 88%。这部分提升是**三层叠加**：
   - 跨任务迁移（seg/det 共享 backbone 学的物体定位特征帮助 cnt）
   - 训练量等价（mtlora 20ep × 868 = 17360 cnt batches vs single 940，**18 倍**）
   - PET 在单任务 5ep 下欠拟合（MAE 23.4，远差于密度回归基线的 15.7），让 single baseline 异常弱
   - **需要 W17 消融**：跑 single_cnt 在 20-50 ep 看 PET 真正的"单任务上限"，从而严格区分三层贡献
3. **seg 稳定正迁移**：+4.3 mIoU / +6.7 mAcc。多任务监督让 backbone 学到更通用视觉表示，分割像素级分类受益。
4. **det 对 MTL 敏感**：Vanilla 持平 single；MTLoRA 因冻结 backbone + 低秩约束，无法为 dense prediction 提供细粒度空间特征，AP@[.5:.95] 跌 4.3 个点。是 MTLoRA 在 dense 任务上的已知短板。
5. **cls 饱和无信息**：6 类生育期视觉特征差异极大，单任务和 MTL 都拿 99%+。**这一列不该作为方法对比的依据**；要么换更难数据集，要么报告时弱化。**注意**：cls 数据集 val 中有 24% 的 scene 与 train 共享（同一片地块不同处理变体），val acc 99% 部分包含数据泄露。

### Smoke 验证（2026-06-16 / 17, RTX 4090）

`python train.py --steps 8 --no_save`（服务器）均 PASS：
- **vanilla** (`swin_tiny`, 4 任务, bs=2)：trainable 39.16M / total 39.16M，~5 it/s
- **MTLoRA** (`swin_tiny`, 4 任务, bs=2, rank=16)：注入 48 个 LoRALinear，trainable 17.30M / total 44.81M（backbone 冻结，仅 LoRA + heads），~6 it/s
- **PET cnt head 升级 smoke**：trainable 28.02M，loss 从 6.88 → ~1.5 健康下降，~12 it/s


---

## W13–W15 各组员 to-do（截至 2026-06-16）

> 框架已经"通"了，head 升级与各方法实现基本完成；W15 末进入主表训练与消融阶段。所有改动走 PR + A review，遵循 [`docs/api_contract.md`](docs/api_contract.md)。

### A（组长 / 架构 / MTLoRA）
- [x] 写 `evaluate.py`：4 个 metric (cls top-1, seg mIoU, det 简化 IoU/recall, cnt MAE) + 写 `logs/results.csv`
- [x] 把 vanilla 在 4 任务各跑 5 epoch (single-task baseline) 入表 — `run_a_experiments.sh` 跑全
- [x] `models/mtl/mtlora.py` — TA-LoRA + TS-LoRA 注入 Swin attn/FFN，48 层 LoRALinear，按 task 路由
- [ ] best-ckpt 选择（目前只保存 last）

### B（数据 & 检测 / TADFormer）
- [x] 升级 `models/heads/det_fcos.py` 占位版 → 完整 **FCOS** 头
- [x] 在 `utils/metrics.py` 补 **AP / AP50**
- [x] `models/mtl/tadformer.py` — 任务嵌入 + 每 block 动态调制 (γ, β)
- [ ] 加 `run_b_experiments.sh`：在主表里追加 TADFormer × 4 任务

### C（分割 & 可视化 / DiTASK）
- [x] 升级 `models/heads/seg_dpt.py` FPN-lite → 完整 **DPT** 头 (Reassemble + Fusion blocks)
- [x] 写 `utils/visualize.py`：4 任务可视化工具 → TensorBoard
- [x] `models/mtl/ditask.py` — backbone 输出后挂可逆微分同胚映射 `Φ_t`
- [ ] 加 `run_c_experiments.sh`：在主表里追加 DiTASK × 4 任务

### D（计数 & 评测 / TaskPrompter）
- [x] 升级 `models/heads/cnt_pet.py` 占位版 → **真 PET 点查询头**（learnable queries + Transformer decoder + Hungarian + CE/SmoothL1, 第三版，按老师反馈）
- [x] 在 `utils/metrics.py` 锁死统一 metric 模块
- [x] 写 `scripts/compute_delta_m.py` Δm 综合指标（按老师建议加）
- [ ] `models/mtl/taskprompter.py` — 每 block 在 token 序列前拼 spatial prompt + FiLM channel prompt
- [ ] 加 `run_d_experiments.sh`

### E（分类 & 部署 / PGT）
- [x] cls head 加 ordinal CE 选项（生育期 6 类有序）→ [`models/heads/cls_pgt.py`](models/heads/cls_pgt.py)
- [x] 写 `utils/logger.py`：统一 TensorBoard + `logs/results.csv` 写入 + checkpoint 规范
- [x] `models/mtl/pgt.py` — 每 block prompt + prompt-guided cross-attention
- [ ] 写 `scripts/demo.py`：单图输入 → 4 任务输出
- [ ] 加 `run_e_experiments.sh`

### 协同约定（再次重申）
- 分支：`feat/<name>-<module>`，PR 合入 `main` 前必须经组长 (A) review
- 共享代码（backbone / dataloader / metric / loss）变更走小 PR；个人 MTL 方法各自在 `models/mtl/<method>.py` 维护，不互相干扰
- 实验结果写 `logs/results.csv`，D 负责合并
- 例会：每周一晚 21:00（30 min）；W13 末例会上各组员 demo 自己升级后的 head 在单任务下的 loss/metric 数

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
- [x] **W13** 单任务 baseline：4 任务 head 各自跑通；当前已落 `single_{seg,det,cnt,cls}_5ep` (运行中)
- [x] **W14** 框架打通：cross-dataset sampler、共享 backbone forward、loss 加权（uniform / uncertainty）
- [x] **W15** 五方法适配：MTLoRA / TADFormer / DiTASK / PGT 已合入主干；TaskPrompter 待 D 完成

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
