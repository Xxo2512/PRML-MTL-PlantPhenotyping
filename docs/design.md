# 系统设计文档 (design.md) — v1.0
---

## 0. 目标与范围

构建统一的**跨数据集（cross-dataset）多任务学习框架**，让 4 个异源小麦视觉任务在共享 Swin 骨干上协同训练；并以同一调度 / 接口 / 评测在 5 种主流 MTL 方法（MTLoRA / TADFormer / DiTASK / TaskPrompter / PGT）上做适配与对比。

---

## 1. 系统总览

```
                   ┌──────────────────────────────────────────┐
                   │           Training Loop (train.py)        │
                   └────────────────┬─────────────────────────┘
                                    │
            ┌──────────────────────┴──────────────────────────┐
            ▼                                                  ▼
   Cross-Dataset Sampler                              Loss Aggregator
   (MTDNN-style, §3)                                  (Uniform/DWA/Uncertainty, §5)
            │                                                  ▲
            ▼                                                  │
   ┌──────────────────────────────────────────────┐           │
   │             MTLModel (§4)                     │           │
   │                                               │           │
   │  ┌──────────────────────────────────────┐    │           │
   │  │   InputAdapter[task]                  │    │           │
   │  │   (resize/normalize/augment)          │    │           │
   │  └────────────────┬─────────────────────┘    │           │
   │                   ▼                            │           │
   │  ┌──────────────────────────────────────┐    │           │
   │  │   SharedBackbone (Swin-T/S)           │    │           │
   │  │   + MTL-specific adapter (§6)         │    │           │
   │  │   [LoRA / dyn-modulation / prompts]   │    │           │
   │  └────────────────┬─────────────────────┘    │           │
   │     stage1..stage4 features                   │           │
   │                   ▼                            │           │
   │  ┌──────────────────────────────────────┐    │           │
   │  │   TaskHead[task]                      │────┼───────────┘
   │  │   seg=DPT / det=FCOS / cnt=PET / cls │    │  per-task loss
   │  └──────────────────────────────────────┘    │
   └──────────────────────────────────────────────┘
```

---

## 2. 4 个数据集 / 输入规约

| 任务 (key) | 原始数据集 | 典型分辨率 | 标注 | InputAdapter 输出 |
|---|---|---|---|---|
| `seg` | Wheat Organ Segmentation | ~1024×1024 | per-pixel mask | RandomCrop→384×384 |
| `det` | Wheat Head Detection (ground) | ~1024×1024 | bbox | Resize→384×384, keep ratio + pad |
| `cnt` | Wheat Leaf Counting | 任意 | point | Resize 短边 384 + CenterCrop 384 |
| `cls` | Wheat Disease Classification | 224–1024 | image-label | Resize→256 + RandomCrop 224，再 pad 到 384 |

> 训练时所有 task 的 InputAdapter 把图像统一吐成 `B × 3 × 384 × 384` 张量；归一化用 ImageNet mean/std（Swin 预训练对齐）。
> 测试用 InputAdapter 关闭随机增广，分割与检测保留 keep-ratio resize（不丢失精度）。

---

## 3. 跨数据集调度策略

### 3.1 三种候选

| 模式 | 描述 | 优点 | 缺点 |
|---|---|---|---|
| **RR**（round-robin）| 固定顺序 [seg, det, cnt, cls] 轮流 | 简单稳定 | 大小数据集贡献相同，偏向小集 |
| **PS**（proportional sampling）| `P(t) ∝ |D_t|^α`，α=0.5 默认 | 兼顾小/大集 | 随机性大，需更多 epoch |
| **HM**（heterogeneous megabatch）| 一个 optimizer step 内 4 任务都 forward 一次，loss 求和 | backbone 信号最丰富，梯度稳定 | 显存翻 4 倍 |

### 3.2 默认与回退

- **默认**：**RR + 梯度累积**。每 4 个 micro-step 视作一个 outer-step：4 个任务各前向反传一次，累加梯度后再 `optimizer.step()`。等价于 HM 的低显存近似。
- **当显存允许**：切到 HM，作为消融。
- **当某任务数据极小（<1K 张）**：切到 PS（α=0.5），避免过拟合。

### 3.3 Sampler 伪代码

```python
# datasets/cross_dataset.py
class CrossDatasetSampler:
    """每次产出 (task_id, batch)。"""
    def __init__(self, loaders: Dict[str, DataLoader], mode: str):
        self.iters  = {t: iter(loader) for t, loader in loaders.items()}
        self.sizes  = {t: len(loader) for t, loader in loaders.items()}
        self.mode   = mode  # 'rr' | 'ps' | 'hm'
        self.order  = self._build_order()
        self.cursor = 0

    def _build_order(self):
        if self.mode == 'rr':
            # 4 任务轮转，按 max(sizes) 长度展开
            n = max(self.sizes.values())
            return [(t, _) for _ in range(n) for t in self.iters]   # outer-step major
        if self.mode == 'ps':
            n_total = sum(self.sizes.values())
            ...   # 加权抽样
        if self.mode == 'hm':
            n = max(self.sizes.values())
            return [tuple(self.iters.keys()) for _ in range(n)]  # 每 step 给所有 task

    def __next__(self):
        slot = self.order[self.cursor]; self.cursor += 1
        if isinstance(slot, tuple):           # HM
            return {t: self._take(t) for t in slot}
        t, _ = slot                            # RR / PS
        return {t: self._take(t)}

    def _take(self, t):
        try: return next(self.iters[t])
        except StopIteration:
            self.iters[t] = iter(self.loaders[t])
            return next(self.iters[t])
```

### 3.4 Epoch / Step 语义

- "1 epoch" 定义为最大数据集（通常是 cls，~万级）跑完一次。
- 优化器 step 次数：RR 模式下 = `max_size × #tasks // accum`；HM 模式下 = `max_size`。
- LR scheduler：以 optimizer step 计 cosine annealing。

---

## 4. 接口冻结 (Frozen Interfaces)

> 以下接口在 W12 冻结。任何修改需 PR + A review。

### 4.1 任务键
```python
# datasets/registry.py
TASKS = ('seg', 'det', 'cnt', 'cls')   # 顺序固定，不可改
```

### 4.2 Backbone

```python
# models/backbone/swin.py
class SwinBackbone(nn.Module):
    out_channels: List[int]  # 长度 4，对应 stage 1..4 的通道数
    out_strides:  List[int]  # 例如 [4, 8, 16, 32]

    def forward(self, x: Tensor, task: Optional[str] = None) -> Dict[str, Tensor]:
        """
        x: [B, 3, 384, 384]
        return: {'s1': [B,C1,96,96], 's2': [B,C2,48,48], 's3': [B,C3,24,24], 's4': [B,C4,12,12]}
        """
```

> `task` 参数对 vanilla Swin 无用；MTL 方法（TADFormer/TaskPrompter/PGT 等）在子类里使用它来路由 task-specific 参数。

### 4.3 InputAdapter

```python
# datasets/input_adapter.py
class InputAdapter(nn.Module):
    def __init__(self, task: str, train: bool): ...
    def forward(self, sample: Dict) -> Dict:
        """
        sample: 原始 dataset 返回的 dict
        return: {'image': [B,3,384,384], 'targets': <task-specific>}
        """
```

### 4.4 TaskHead

```python
# models/heads/base.py
class BaseTaskHead(nn.Module):
    task: str       # 'seg' | 'det' | 'cnt' | 'cls'
    in_channels: List[int]    # 期望 backbone out_channels（4 元组），缺位用 None

    def forward(self, feats: Dict[str, Tensor], targets=None) -> Dict[str, Any]:
        """
        train: return {'loss': scalar, 'loss_items': dict, 'pred': ...}
        eval : return {'pred': ..., 'metric': dict}
        """
```

### 4.5 MTLModel

```python
# models/mtl/base.py
class MTLModel(nn.Module):
    backbone: SwinBackbone
    heads: nn.ModuleDict  # {'seg': ..., 'det': ..., 'cnt': ..., 'cls': ...}
    # 子类按需添加: lora_modules / dyn_modulators / prompts / phi_t / ...

    def forward(self, batch: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        batch: {'seg': {'image', 'targets'}, ...}  仅包含本 step 涉及的 task
        return: {task: head_output}
        """
```

### 4.6 Metric / Logging 协议

- 每个 head 训练时必须在 `loss_items` 里输出至少一个标量子项，命名规约：`<task>/<name>`，如 `seg/ce`、`det/focal`。
- 评估指标统一格式：`{'<task>/<metric>': float}`，例如 `seg/mIoU`、`det/AP50`、`cnt/MAE`、`cls/mAP`。
- 训练日志：TensorBoard + `logs/results.csv`（D 维护）。

---

## 5. 损失加权

支持三种策略，由 `configs/*.yaml: loss.weighting` 切换：

| 策略 | 实现 | 参数 |
|---|---|---|
| `uniform` | `L = Σ_t L_t / |T|` | 无 |
| `dwa` | Dynamic Weight Averaging[a] | 温度 T=2 |
| `uncertainty` | Kendall et al. learnable σ_t | 每 task 一个可学习 `log σ²` |

**默认**：`uncertainty`（论文实证更稳定）。**回退**：`uniform`（梯度不稳时用作 sanity baseline）。

> 梯度累积模式下，先 per-task 算 `L_t`，累加未归一化梯度；HM 模式下，先 weighted sum 再 backward。

---

## 6. 5 种方法的统一适配规范

| 方法 | 主责 | 注入位置 | 训练参数 | 是否冻结 backbone | task 路由 |
|---|---|---|---|---|---|
| **vanilla** (baseline) | A | 无 | backbone + heads | 否 | 无 |
| **MTLoRA** | A | Swin attn/FFN LoRA | LoRA + heads + InputAdapter | 是 | TS-LoRA 按 `task` 选 |
| **TADFormer** | B | 每 block 动态调制 (γ, β) | 调制器 + heads（+ 可选 LoRA） | 视实验 | task embedding 路由 |
| **DiTASK** | C | backbone 输出后 `Φ_t` | Φ_t + heads | 是 | task → Φ_t |
| **TaskPrompter** | D | 每 block prompt token | prompt + heads | 是 | per-task prompt |
| **PGT** | E | 每 block prompt + cross-attn | prompt + cross-attn + heads | 是 | per-task prompt |

**适配规则（强约束）**：
1. 保持原方法**网络结构**不变；只允许补 InputAdapter 和 4 task heads。
2. backbone 一律用 `swin_tiny`，预训练权重 = `timm/swin_tiny_patch4_window7_224.ms_in1k`，再扩到 384 输入（位置编码插值由 timm 处理）。
3. 显存超出 24GB 的方法降到 `swin_tiny` + 256 输入做调试，主表实验环节再升回 384。

---

## 7. 训练流程伪代码

```python
# train.py（简化）
model = build_mtl_model(cfg)
loaders = {t: build_loader(t, cfg) for t in TASKS}
sampler = CrossDatasetSampler(loaders, mode=cfg.data.scheduler)
loss_agg = build_loss_aggregator(cfg.loss.weighting, tasks=TASKS)

optimizer = build_optimizer(model)
scheduler = build_scheduler(optimizer)

accum = cfg.train.accum_steps  # RR 默认 = #TASKS
for step, batch_dict in enumerate(sampler):
    out = model(batch_dict)
    per_task_loss = {t: out[t]['loss'] for t in batch_dict}
    loss = loss_agg(per_task_loss) / accum
    loss.backward()

    if (step + 1) % accum == 0:
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); optimizer.zero_grad(); scheduler.step()
        log_step(step, per_task_loss)

    if step % cfg.train.eval_every == 0:
        evaluate(model, loaders, logger)
```

---

## 8. 评测协议

- 每任务用其**官方 val 划分**；若数据集无标准 split，B 在 dataloader 中固定 random_seed=42 划 8:2。
- 评测频率：每 2000 optimizer step 一次（约半 epoch），写入 TensorBoard + `logs/results.csv`。
- 主表：训练结束后 best-of-val 模型在 val 上的 final 指标。
- 单任务 baseline（reference）由各 head 主责者在 W13 跑通，作为 MTL 上限/下限对照。

---

## 9. 工程约束 (W12 之后冻结)

| 项 | 值 |
|---|---|
| 框架 | PyTorch ≥ 2.0，timm 最新，einops |
| 输入分辨率 | 384×384（主表）/ 256×256（调试 & 显存敏感方法消融） |
| Backbone | `swin_tiny`（主表），`swin_small`（消融） |
| Optimizer | AdamW，lr=1e-4，wd=0.05，warmup 1000 step |
| Scheduler | Cosine annealing 到 lr×0.01 |
| Batch | 8 per task（RR/PS）或 4 per task（HM） |
| Epoch | 50（主表）/ 20（调试） |
| 梯度裁剪 | grad-norm 1.0 |
| 随机种子 | 42（默认）；主表跑 1 seed，关键消融跑 3 seed 报均值 |

---

## 10. 实验计划（与 README §4 对齐）

主表（W16）：5 方法 × 4 任务 = 20 cell + 4 single-task reference = 24 cell。

消融（W17，至少 5 项）：
1. 调度：RR vs PS vs HM
2. Loss 加权：uniform vs DWA vs uncertainty
3. Backbone：swin_tiny vs swin_small
4. 输入分辨率：256 vs 384
5. 任务子集：3 任务（去 cls）vs 4 任务，分析"分类是否帮助 dense 任务"

可视化（W17）：分割 mask 叠加、检测框、计数密度图、分类 Grad-CAM。Demo：`scripts/demo.py <image>` 一脚本出 4 任务结果。

---

## 11. 风险与缓解（更新自 README）

| 风险 | 监测信号 | 缓解 |
|---|---|---|
| 数据集到位晚 | W11 末仍未拿全 | 用公开 GWHD + PlantVillage 子集占位跑 pipeline |
| HM 模式 OOM | OOM 报错 | 自动回退到 RR + accum=4 |
| 某 task loss 主导 | TensorBoard 某 task 曲线碾压 | 切 uncertainty 加权；必要时手工 clip 该任务的 loss scale |
| 五方法适配进度不齐 | W15 末仍有方法不能训 | 该成员降级目标到"vanilla baseline + 该方法仅一项消融"，组长 (A) 补位 |
| 评测脚本不一致 | 同一 method 多组数差异大 | D 在 W14 锁死 `utils/metrics.py`，所有人禁止本地 fork |

---

## 12. 待办与负责人（W12 收尾）

- [ ] A：合并 vanilla baseline 框架到 main（依赖 §4 接口）
- [ ] A：完成 `models/mtl/mtlora.py` skeleton（带 LoRA 注入）
- [ ] B：把 4 个数据集的 dataloader + InputAdapter 落到 `datasets/`
- [ ] C：实现 `utils/visualize.py` 雏形（仅占位）
- [ ] D：实现 `utils/metrics.py`（4 任务指标）
- [ ] E：写 `scripts/demo.py` 框架（暂跑 vanilla）
- [ ] 全员：自查本方法是否完全契合 §4 接口；不契合则在 PR 描述里说明，A 评估是否扩接口

---

## 注

- [a] DWA: Liu, S., Johns, E., Davison, A.J. *End-to-end multi-task learning with attention.* CVPR 2019.
- 其余引用见 `docs/literature_review.md` 与 `README.md`。
