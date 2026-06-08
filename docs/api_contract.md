# 接口契约 (API Contract) — W12 冻结版

> 本文件是 design.md §4 的可执行精简版，作为各成员开发时的"硬契约"。
> **冻结日期**：W12 末。任何修改必须发 PR，描述影响面，由 A review 后才能合入 main。

---

## 0. 任务键 & 目录

```python
TASKS = ('seg', 'det', 'cnt', 'cls')          # 不可变顺序

# 仓库内文件归属
datasets/
  __init__.py              # B
  registry.py              # A: 注册 dataset + dataloader 工厂
  input_adapter.py         # B
  cross_dataset.py         # A: CrossDatasetSampler
  wheat_seg.py             # C: Wheat Organ Segmentation
  wheat_det.py             # B: Wheat Head Detection
  wheat_cnt.py             # D: Wheat Leaf Counting
  wheat_cls.py             # E: Wheat Disease Classification

models/
  backbone/swin.py         # A
  heads/base.py            # A: BaseTaskHead
  heads/seg_dpt.py         # C
  heads/det_fcos.py        # B
  heads/cnt_pet.py         # D
  heads/cls_mlp.py         # E
  mtl/base.py              # A: MTLModel
  mtl/vanilla.py           # A: 基线（无任何 MTL trick）
  mtl/mtlora.py            # A
  mtl/tadformer.py         # B
  mtl/ditask.py            # C
  mtl/taskprompter.py      # D
  mtl/pgt.py               # E

utils/
  metrics.py               # D
  losses.py                # A: loss aggregator (uniform/dwa/uncertainty)
  collate.py               # B: task_collate(batch, task)
  visualize.py             # C
  logger.py                # E
```

---

## 1. Backbone

```python
# models/backbone/swin.py
class SwinBackbone(nn.Module):
    out_channels: List[int]   # [C1, C2, C3, C4]，Swin-T 默认 [96, 192, 384, 768]
    out_strides:  List[int]   # [4, 8, 16, 32]

    def forward(self, x: Tensor, task: Optional[str] = None) -> Dict[str, Tensor]:
        ...
```

**强约束**：
- 输入：`x.shape == [B, 3, 384, 384]`，dtype=float32，已归一化到 ImageNet mean/std。
- 输出：`{'s1': [B, C1, 96, 96], 's2': [B, C2, 48, 48], 's3': [B, C3, 24, 24], 's4': [B, C4, 12, 12]}`。
- `task` 参数：vanilla 子类忽略；MTL 子类用 `task ∈ TASKS` 来路由 task-specific 参数。
- 不可在 backbone 内部访问 `targets`；任何 task 监督只能进入 head。

---

## 2. InputAdapter

```python
# datasets/input_adapter.py
class InputAdapter:
    def __init__(self, task: str, train: bool, input_size: int = 384): ...
    def __call__(self, sample: Dict) -> Dict:
        ...
```

**输入 (`sample`)**：
- 各 task dataset `__getitem__` 的原始返回。约定为 dict：
  - 公共字段：`image_path: str`、`image_pil: PIL.Image`
  - 任务字段：
    - seg → `mask: np.ndarray [H, W] uint8`
    - det → `boxes: np.ndarray [N, 4] (x1,y1,x2,y2) 原图像素坐标`、`labels: np.ndarray [N] int64`（0=background, ≥1=foreground）
    - cnt → `points: np.ndarray [N, 2] 原图像素坐标`、`count: int`
    - cls → `label: int`

**输出**：
```python
{
    'image':   Tensor[3, 384, 384],          # 已归一化 (ImageNet mean/std)
    'targets': <task-specific tensor dict>,
    'meta':    {'image_path': str, 'orig_size': (H, W), 'task': str}
}
```

**坐标空间约定**：
- det 输出 `targets['boxes']` 已经过 keep-ratio resize + pad，取值 `[0, input_size]` 像素；`targets['labels']` 长度与 boxes 同步。
- cnt 输出 `targets['points']` 已缩到 `[0, input_size)` 像素；`targets['count']` 是标量 float tensor。
- seg 输出 `targets['mask']` 是 `[input_size, input_size] long` (NEAREST resize)。
- cls 输出 `targets['label']` 是标量 long tensor。

**collate**：dataloader 用 `utils.collate.task_collate(batch, task)` 把上面 dict 合 batch。变长目标 (det boxes/labels, cnt points) 保留为 list，其余 stack。`meta` 透传为 list-of-dict。

---

## 3. TaskHead

```python
# models/heads/base.py
class BaseTaskHead(nn.Module, ABC):
    task: str                        # 'seg' | 'det' | 'cnt' | 'cls'
    in_channels: List[Optional[int]] # 长度 4，None 表示该 stage 不消费

    @abstractmethod
    def forward(self, feats: Dict[str, Tensor], targets=None) -> Dict[str, Any]:
        """
        feats: 与 Backbone 输出一致，{'s1': ..., 's4': ...}
        targets: train 时给，eval 时可省

        train  : return {
            'loss': Tensor scalar,
            'loss_items': Dict[str, float],   # 子项已 .item()
            'pred': <仅用于可视化>
        }
        eval   : return {
            'pred': <task-specific>,
            'metric': Dict[str, float]        # e.g. {'seg/mIoU': 0.42}
        }
        """
```

**子类输出 (`pred`) 格式约定**：

| Head | pred 字段 |
|---|---|
| seg_dpt | `logits: [B, C, 384, 384]` |
| det_fcos | `boxes: [B, N, 4]`, `scores: [B, N]`, `labels: [B, N]` (after NMS) |
| cnt_pet | **W13**: `density: [B, 1, H', W']`, `count: [B]` (head 内部从 `targets['points']` 现算 GT density, 不暴露在 dataset/adapter 接口); **W14** 切到 PET point-query 时升级为 `points: [B, N, 2]`, `count: [B]` |
| cls_mlp | `logits: [B, C]` |

---

## 4. MTLModel

```python
# models/mtl/base.py
class MTLModel(nn.Module, ABC):
    backbone: SwinBackbone
    heads: nn.ModuleDict      # {task: BaseTaskHead}

    def forward(self, batch: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        batch: 由 sampler 产出，{'seg': {'image': ..., 'targets': ...}, ...}
               * RR/PS 模式下 len(batch) == 1
               * HM 模式下     len(batch) == #TASKS
        return: {task: head_output}     # head_output 见 §3
        """
```

**适配规则**（所有 5 个 MTL 子类必须满足）：
1. 父类 `__init__` 接受 `cfg: dict`，子类可加自有字段但不得改父类签名。
2. 子类若引入 task-specific 参数，按 `self.task_modules[task]` 命名存放，便于 checkpoint 检查。
3. 子类必须能在 `len(batch) == 1` 与 `== 4` 两种情况下都正确 forward。
4. backbone 冻结 / LoRA 注入等通过 `self._freeze_backbone()` / `self._inject_lora()` 等显式方法完成，禁止在外部 train.py 里改 grad flag。

---

## 5. Sampler

```python
# datasets/cross_dataset.py
class CrossDatasetSampler:
    mode: str   # 'rr' | 'ps' | 'hm'
    def __iter__(self) -> Iterator[Dict[str, Dict]]: ...
```

每次产出一个"batch dict"。整数 epoch 后自动 re-init 所有内部 iter。

`__len__` 返回 outer-step 数：
- RR：`max(|D_t|) // batch_per_task`
- PS：同上（按数据量上界估）
- HM：同上

---

## 6. Loss Aggregator

```python
# utils/losses.py
class LossAggregator(nn.Module):
    mode: str    # 'uniform' | 'dwa' | 'uncertainty'
    def forward(self, per_task_loss: Dict[str, Tensor]) -> Tensor:
        """返回标量。uncertainty 模式下，模块内部维护 log_sigma²。"""
```

**注册到优化器**：uncertainty 模式下，`LossAggregator` 自己的参数加入 main optimizer。

---

## 7. Metrics

```python
# utils/metrics.py

def compute_seg_metric(pred_logits, mask) -> Dict[str, float]:
    """returns {'seg/mIoU': ..., 'seg/mAcc': ...}"""

def compute_det_metric(preds, targets) -> Dict[str, float]:
    """returns {'det/AP': ..., 'det/AP50': ...}"""

def compute_cnt_metric(pred_count, gt_count) -> Dict[str, float]:
    """returns {'cnt/MAE': ..., 'cnt/RMSE': ..., 'cnt/R2': ...}"""

def compute_cls_metric(logits, labels) -> Dict[str, float]:
    """returns {'cls/mAP': ..., 'cls/BA': ...}"""
```

**评测 entry**：`utils/metrics.evaluate_model(model, loaders) -> Dict[str, float]`，把上面 4 个汇总到一个 flat dict，行尾追加 `logs/results.csv`。

---

## 8. Config (yaml) 顶层 schema

```yaml
exp_name: <str>            # 例：mtlora_swint_384_rr_uncertainty
method:   <str>            # vanilla | mtlora | tadformer | ditask | taskprompter | pgt
seed:     42

model:
  backbone: swin_tiny       # swin_tiny | swin_small
  pretrained: timm/swin_tiny_patch4_window7_224.ms_in1k
  input_size: 384
  freeze_backbone: true     # 5 种 MTL 方法默认 true；vanilla false

  # method-specific（仅对应方法读取，其它方法忽略）
  mtlora: { rank: 16, alpha: 16, target_modules: ['qkv','proj','fc1','fc2'] }
  tadformer: { task_dim: 64 }
  ditask: { phi_type: svd_rotation, rank: 8 }
  taskprompter: { n_spatial: 4, n_channel: 4 }
  pgt: { n_prompt: 8 }

data:
  scheduler: rr             # rr | ps | hm
  batch_per_task: 8
  num_workers: 8
  proportional_alpha: 0.5   # 仅 ps 用

tasks:
  seg:
    enabled: true
    data_root: datasets/raw/segmentation_dataset
    num_classes: 4               # Background, Head, Stem, Leaf
    label_subdir: class_id
    classes: [Background, Head, Stem, Leaf]
  det:
    enabled: true
    data_root: datasets/raw/detect_dataset
    num_classes: 1               # 麦穗单类 (label index = 1; 0 留给 background)
    fmt: yolo
  cnt:
    enabled: true
    data_root: datasets/raw/count_dataset
    anno_fmt: voc_xml            # 叶尖计数, bbox 中心作为点标注
    target_repr: point_from_bbox_center
    semantics: leaf_tip
  cls:
    enabled: true
    data_root: datasets/raw/classification_dataset
    num_classes: 6               # 生育期 6 类 (与原指导书的"病害分类 8 类"不同, 已确认)
    semantics: growth_stage
    classes: [1_Tillering, 2_Jointing, 3_BH, 4_Flowering, 5_Filling, 6_Ripening]

loss:
  weighting: uncertainty    # uniform | dwa | uncertainty
  dwa_temp: 2.0

train:
  epochs: 50
  optimizer: adamw
  lr: 1.0e-4
  weight_decay: 0.05
  warmup_steps: 1000
  grad_clip: 1.0
  accum_steps: 4            # rr 默认 = #TASKS；hm 设 1
  eval_every: 2000
  ckpt_every: 5000

paths:
  out_dir: checkpoints/${exp_name}
  log_dir: logs/${exp_name}
```

---

## 9. Checkpoint 格式

```python
{
    'step': int,
    'epoch': int,
    'model_state': dict,            # MTLModel.state_dict()
    'optimizer_state': dict,
    'scheduler_state': dict,
    'loss_agg_state': dict,         # uncertainty 的 log_sigma²
    'best_metric': {                # 单个标量，用于 best-checkpoint 选择
        'seg/mIoU': ...,
        'det/AP50': ...,
        'cnt/MAE': ...,
        'cls/mAP': ...,
        'aggregate': ...,           # 简单平均（去 MAE 取负后），由 evaluate 写
    },
    'cfg': dict,                    # 原始 yaml 解析后
}
```

文件名：`<out_dir>/ckpt_step{step}.pt` + `<out_dir>/best.pt`（软链/拷贝）。

---

## 10. PR 模板（A 强制执行）

每个 PR 必须在描述里列：
1. 影响的契约项（如有）
2. 单测：本次新增/修改的 dataset / head / mtl 模块，至少跑通 forward + backward + 一次 train step
3. 截图：TensorBoard 第一个 1k step 的 loss 曲线（粗证不发散）
4. checklist：
   - [ ] 通过 `python -m scripts.check_contract`（A 提供脚本）
   - [ ] 没有改父类签名 / 全局 TASKS / yaml schema 顶层键
   - [ ] 文档：如改了任何 §1–§9，同步更新本文件

