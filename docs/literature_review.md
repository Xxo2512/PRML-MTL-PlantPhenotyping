# 文献综述（W10–W11）

> 本文档面向项目"跨数据集异质同构 MTL"框架的设计需求，重点提炼**对我们的统一框架有约束作用的细节**：每篇方法在哪里插入额外参数、对 backbone 有什么假设、原始任务集合与我们四任务的差异、能否直接套用 cross-dataset 调度。
> 5 人各自精读所主责方法时，可基于本文做扩展。

---

## 0. 跨数据集 MTL 的核心难点（贯穿全篇）

现有 MTL 工作几乎都假设 **任务共享同一张输入图像**（NYUD-v2、PASCAL-Context、Taskonomy……），forward 一次拿到所有任务的预测。我们的设定不同：

- 4 个任务 = 4 个**异源**数据集（Wheat Organ Seg / Wheat Head Det / Wheat Leaf Cnt / Wheat Disease Cls）
- 输入分辨率、归一化、增广策略不同
- 标注密度差异大（pixel-wise mask / sparse boxes / sparse points / single label）

这意味着：
1. **数据调度**必须在 batch/step 层面解决，而非在样本层面共享。参考 MTDNN[1]。
2. **共享骨干**必须能稳定接受来自 4 个分布的输入；通常做法是给每个任务一个轻量 `InputAdapter`（resize + 归一化 + 颜色增广策略），把所有任务统一到 backbone 期望的输入空间。
3. **每步 loss** 只对当前 batch 的那个/那些任务计算；backbone 的梯度由该 batch 的任务信号驱动。

---

## 1. MTDNN — 调度蓝本 [1]

**核心**：NLP 多任务学习，shared BERT encoder + per-task heads。

**调度（我们最关心的部分）**：
1. 给每个任务 t 准备独立 dataloader，得到 batch 列表 `B_t = [b_{t,1}, …, b_{t,N_t}]`。
2. 把所有 task 的所有 batch 索引混合：`P = [(t, i) for t in tasks for i in range(N_t)]`，每个 epoch 开始**整体 shuffle** 一次。
3. 训练循环：依序取 `(t, i)`，从 task t 取 batch，前向 + 仅用 task t 的 loss 反传，更新 backbone 与 head_t。

**对我们的启示**：
- 这是 **proportional-by-batch-count** 的天然实现：大数据集自然贡献更多 step。
- 不需要把异源数据塞到同一个 forward；和我们的 cross-dataset 设定天然契合。
- **隐患**：backbone 在不同 step 看到的数据分布会"摆动"。需要小学习率 + 适度的梯度裁剪，且建议每若干 step 做一次 **task 轮转的微批 (micro-batch)**，让一次 optimizer.step() 包含多个任务的梯度。
- 我们的默认调度即基于此（详见 design.md §3）。

---

## 2. MTLoRA [2] — A 主责，CVPR 2024

**架构**：
- Backbone：Swin Transformer（v1，tiny / small / base，ImageNet 预训练）
- 在 backbone 的注意力 / FFN 线性层中插入两类 LoRA：
  - **TA-LoRA**（Task-Agnostic）：所有任务共享，捕获通用结构
  - **TS-LoRA**（Task-Specific）：每任务一份，捕获任务专属偏置
- Forward：`y = W₀ x + α_TA · BA_TA(x) + α_t · BA_t(x)`，其中 `W₀` 冻结，仅训练 LoRA 与 task heads。

**原始任务**：NYUD-v2 (semseg/depth/normal/edge) + PASCAL-Context (semseg/parts/saliency/normal)。**全是 dense prediction**。

**对我们的适配点**：
| 方面 | 原 MTLoRA | 我们的改造 |
|---|---|---|
| 任务类型 | 全 dense | 加 cls / cnt（cls 头取 stage4 池化；cnt 头用 PET 风格点查询） |
| 输入 | 同一图 | InputAdapter 预处理 4 个数据集到 384×384 |
| 调度 | 单 batch 一次 forward 出 N 任务 | 改为 MTDNN 风格 round-robin；TS-LoRA 仅当前任务激活，TA-LoRA 始终参与 |
| Head | DPT-like 解码器 | 检测换 FCOS、计数换 PET head、分类换 MLP |
| 训练 | LoRA + heads 训练，骨干冻结 | 同；额外训练 InputAdapter |

**关键超参**：LoRA rank（默认 r=16）、α（默认 16）、应用层（默认 q,k,v,o + MLP）。

---

## 3. TADFormer [3] — B 主责，CVPR 2025

**架构**：Task-Adaptive Dynamic Transformer。基于 ViT/Swin，**动态地**为每个任务在每个 block 调制 token 表征。

**核心机制**：
- 引入 task embedding `e_t`
- 在每个 transformer block，用 `e_t` 生成 **dynamic gate / scale / shift** 调制 token：`h_t = (1 + γ(e_t)) · h + β(e_t)`
- 相比 TaskPrompter 的"加 prompt"，TADFormer 是"调制已有 token"，对 token 长度无影响。

**对我们的适配点**：
- 天然支持"按任务切换"，与 round-robin 调度兼容性最好——只需在每个 step 把 `task_id` 传进 backbone。
- task embedding 维度需与 backbone hidden dim 对齐；Swin 有多 stage，需要为每 stage 维护一份调制器。
- 参数量增长温和：`#tasks × dim × 2`（γ、β）每 block。

**注意**：原文实验同样在 NYUD-v2/PASCAL，需补 cls/cnt head。

---

## 4. DiTASK [4] — C 主责，CVPR 2025

**架构**：Diffeomorphic Multi-task Fine-Tuning。把任务特定的"特征变换"建模为**微分同胚映射**（光滑且可逆），保持 backbone 学到的几何/拓扑结构。

**核心机制**：
- 在 backbone 输出特征 `f` 上应用 `f_t = Φ_t(f)`，其中 `Φ_t` 是参数化的可逆光滑映射
- 参数化方式（论文用）：基于 SVD 的旋转/拉伸组合，或归一化流式的耦合层
- 训练时 backbone 冻结或低秩微调，`Φ_t` per-task 训练

**对我们的适配点**：
- 适配相对干净：`Φ_t` 作用在 backbone 输出 → task head 输入之间，**与 head 类型无关**，所以 cls/det/cnt/seg 都能套。
- 但要求 backbone 输出形状统一；多 stage 输出时，每 stage 一份 `Φ_t^{(s)}`。
- 对计算资源较敏感（SVD/可逆映射），上 Swin-tiny 比较稳。

---

## 5. TaskPrompter [5] — D 主责，ICLR 2023

**架构**：Spatial-Channel Multi-Task Prompting，基于 ViT。

**核心机制**：
- 每任务在每 block 注入两类 prompt：
  - **Spatial Prompt**：拼接到 token 序列前，参与 self-attention
  - **Channel Prompt**：作用在 channel 维度（FiLM-style scale/shift）
- 共享 ViT backbone，prompt 是唯一可训练（或低代价训练）部分

**对我们的适配点**：
- ViT-only 的设计；如果坚持 Swin，需要把 spatial prompt 改造为 stage-wise 的可学习 token，并兼容 window attention（建议用 ViT 直接跑，或在 Swin shifted-window 内插入 prompt token 到每个 window）。
- 与 round-robin 兼容：每 step 只激活 task t 的 prompt。
- 显存友好。

---

## 6. PGT [6] — E 主责，TMM 2024

**架构**：Prompt-Guided Transformer（dense prediction）。在 TaskPrompter 思路上做了化简和改进，强调 prompt 之间的交互（task-aware prompt fusion）。

**核心机制**：
- 仍基于 ViT/Swin，每任务一组可学习 prompt
- 引入 **prompt-guided cross-attention**：用任务 prompt 作为 query 去查询 shared visual tokens，让 prompt 能"主动选择"特征
- 解码端用统一的 transformer decoder

**对我们的适配点**：
- 思路与 TaskPrompter 相近，但更强调 **prompt-feature 交互**；可作为 TaskPrompter 的对照组。
- 适配工作与 TaskPrompter 接近；E 在实现时可以与 D 共用 prompt 基础设施。

---

## 7. 五方法共性 / 差异速查表

| 维度 | MTLoRA | TADFormer | DiTASK | TaskPrompter | PGT |
|---|---|---|---|---|---|
| Backbone 假设 | Swin | ViT/Swin | 任意 | ViT | ViT/Swin |
| 改动位置 | 注意力/FFN 层内 LoRA | 每 block 调制 | backbone 输出后 Φ_t | 每 block 加 prompt | 每 block + decoder |
| 是否冻结 backbone | 可冻结 | 视实现 | 多冻结 | 可冻结 | 可冻结 |
| 与 round-robin 调度 | ✅ 直接 | ✅ 直接 | ✅ 直接 | ✅ 直接 | ✅ 直接 |
| 显存/算力 | 低 | 低 | 中 | 低 | 中 |
| 缺失 head 类型 | cls / cnt | cls / cnt | cls / cnt | cls / cnt | cls / cnt |
| 适配 cross-dataset 难度 | 低 | 低 | 中 | 中（ViT-only 时低） | 中 |

> 5 种方法都把 **task 维度的可学习参数** 放在共享 backbone 之外/之上，因此可以套到同一个 cross-dataset 调度框架。这正是本项目"异质同构"标题的着力点：异质（数据/任务）但同构（统一调度+统一接口）。

---

## 8. 任务头补齐（4 任务统一接口）

所有 5 种方法原本都不覆盖 cls + cnt，因此 **task head 由我们自己补齐并统一**：

| 任务 | Head | 输入 | 输出 | 损失 |
|---|---|---|---|---|
| Seg | DPT[8] | 多 stage feature | HxW logit | CE + (可选) Dice |
| Det | FCOS[7] | 多 stage feature | (cls, ctr, reg) per-pixel | Focal + IoU + BCE |
| Cnt | PET[10] 风格 | 高分辨 stage feature | 点查询坐标 + 计数 | L1 + Hungarian + 计数 L1 |
| Cls | MLP | stage4 GAP | C-way logit | CE |

> Head 由 B/C/D/E 各自负责实现并提交到 `models/heads/`，A 提供 `BaseTaskHead` 抽象类。

---

## 参考文献

[1] Liu et al. *Multi-task DNNs for NLU.* ACL 2019.
[2] Agiza, Neseem, Reda. *MTLoRA.* CVPR 2024.
[3] Baek et al. *TADFormer.* CVPR 2025.
[4] Mantri et al. *DiTASK.* CVPR 2025.
[5] Ye, Xu. *TaskPrompter.* ICLR 2023.
[6] Lu et al. *PGT.* IEEE TMM 2024.
[7] Tian et al. *FCOS.* TPAMI 2020.
[8] Ranftl, Bochkovskiy, Koltun. *DPT.* ICCV 2021.
[9] Han et al. *FoMo4Wheat.* arXiv 2509.06907, 2025.
[10] Liu et al. *PET.* ICCV 2023.
