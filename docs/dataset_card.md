# 数据集卡片 (Dataset Card) — W12 末

> 真实数据已就位，统计与示例标注采样自 `datasets/raw/`。本文件是后续 dataloader / InputAdapter 实现的依据，B/C/D/E 在 W13 开始按此对接。

---

## 1. 概览

| 任务 | 目录 | train | val | test | 标注格式 | 分辨率 |
|---|---|---|---|---|---|---|
| **cls 生育期分类** | `datasets/raw/classification_dataset/` | 61 489 | 7 702 | 7 701 | ImageFolder（6 子目录） | 400×400 |
| **det 麦穗检测** | `datasets/raw/detect_dataset/` | 3 607 | 1 448 | 1 382 | YOLO `.txt`（cls cx cy w h, 归一化） | 1024×1024 |
| **cnt 叶尖计数** | `datasets/raw/count_dataset/` | 1 508 | 379 | — | Pascal VOC XML，`<object><name>tip</name><bndbox>...</bndbox>` | 1024×1024 |
| **seg 器官分割** | `datasets/raw/segmentation_dataset/` | 789 | 99 | 99 | 单通道 mask PNG，class_id ∈ {0,1,2,3} = {Background, Head, Stem, Leaf} | 512×512 |

---

## 2. cls：生育期分类（**与原指导书的"病害分类"不同，已确认**）

### 类别（6 类，目录名即类名）

| index | label folder | 中文 | Zadoks 大致区间 |
|---|---|---|---|
| 0 | `1_Tillering` | 分蘖期 | Z20–Z29 |
| 1 | `2_Jointing` | 拔节期 | Z30–Z39 |
| 2 | `3_BH` | 孕穗+抽穗 (Booting + Heading) | Z40–Z59 |
| 3 | `4_Flowering` | 开花期 | Z60–Z69 |
| 4 | `5_Filling` | 灌浆期 | Z70–Z79 |
| 5 | `6_Ripening` | 成熟期 | Z80–Z99 |

### 样本计数（每类）

| split / class | Tillering | Jointing | BH | Flowering | Filling | Ripening |
|---|---|---|---|---|---|---|
| train | 10 583 | 10 922 | 9 072 | 10 921 | 10 804 | 9 187 |
| val | 1 325 | 1 368 | 1 137 | 1 368 | 1 353 | 1 151 |
| test | 1 325 | 1 367 | 1 137 | 1 368 | 1 353 | 1 151 |

类间近似均衡（min/max ≈ 0.83），可用 plain CE，无需 class-balanced loss。

### 文件名样式
`WGSP-CN-Data__210203A101.png` / `..._c.png` / `..._cd.png` — 同一拍摄可能有变体（疑似多视角/裁剪），E 在 W13 需确认是否将"同主名"放入同一 split 以避免泄漏。

---

## 3. det：麦穗检测

### 结构
```
detect_dataset/
  images/
    train/  3607  (1024×1024 .jpg)
    val/    1448
    test/   1382
    train_val_nobox/   # ⚠ 含义未确认；疑似无目标负样本
  labels/
    train/  3607  (.txt, YOLO)
    val/    1448
    test/   1382
```

### 标注示例
`labels/train/<hash>.txt`：每行一个 box
```
0  0.934082  0.041992  0.040039  0.076172
0  0.971191  0.055664  0.055664  0.111328
0  0.712891  0.042969  0.126953  0.068359
...
```
格式：`class cx cy w h`，归一化到 [0, 1]。**仅 class=0**（麦穗）。

### B 在 W13 需澄清
- [ ] `train_val_nobox/` 用途（负样本？训练集补充？）→ 联系老师
- [ ] 是否需要做 keep-ratio resize → 1024 直接缩 384 会丢小穗，建议保 keep-ratio 并在 InputAdapter 里 pad 到 384×384
- [ ] FCOS 主表分辨率：384 跑通后，消融时升到 512/768 看 AP50 改善

---

## 4. cnt：叶尖计数 (Wheat Leaf Tip Counting)

> 已与老师确认：原 `count_dataset (.mat)` 已被 **叶尖计数** 数据集替换，标注是 *点标注*（bbox 仅作为承载叶尖位置的容器，取中心即可）。

### 结构
```
count_dataset/
  images/{train,val}/        *.jpg  1024×1024
  annotations/{train,val}/   *.xml  Pascal VOC 格式 (filename 一一对应)
```

> 仅 `train/val` 两个 split，没有 test。后续评测用 val。

### 标注示例
```xml
<object>
  <name>tip</name>
  <bndbox>
    <xmin>24</xmin> <ymin>85</ymin>
    <xmax>41</xmax> <ymax>100</ymax>
  </bndbox>
</object>
```
- `name` 恒为 `tip`（叶尖单类）
- `bndbox` 是叶尖周围的一个小框，**实际作为点标注使用**：取 `(cx, cy) = ((xmin+xmax)/2, (ymin+ymax)/2)`

### Loader 行为（[`datasets/wheat_cnt.py`](../datasets/wheat_cnt.py)）
1. `_parse_voc_xml` 把每个 object 的 bbox 中心转点
2. `_gaussian_density` 在 stride=8 的低分辨密度图（48×48）上撒高斯（σ=2）
3. 返回 `{image: [3,384,384], density: [48,48], count: scalar}`

---

## 5. seg：器官分割

### 结构
```
segmentation_dataset/
  {train,val,test}/
    images/      *.png   512×512 RGB
    class_id/    *.png   512×512 L (uint8), 像素值 ∈ {0,1,2,3}
```

文件名一一对应，无需额外 mapping。

### 类别（4 类，含背景，**已确认**）

| pixel value | 语义 |
|---|---|
| 0 | Background |
| 1 | Head |
| 2 | Stem |
| 3 | Leaf |

> 30 张采样像素值集合恒为 {0,1,2,3}。

---

## 6. 与 base.yaml 的对应

```yaml
tasks:
  seg: { num_classes: 4, label_subdir: class_id,
         classes: [Background, Head, Stem, Leaf] }
  det: { num_classes: 1, fmt: yolo }
  cnt: { anno_fmt: voc_xml, target_repr: point_from_bbox_center, semantics: leaf_tip }
  cls: { num_classes: 6, semantics: growth_stage,
         classes: [1_Tillering, 2_Jointing, 3_BH, 4_Flowering, 5_Filling, 6_Ripening] }
```

---

## 7. 已澄清问题汇总（老师 W12 末回复）

| # | 任务 | 问题 | 回答 |
|---|---|---|---|
| Q1 | det | `train_val_nobox/` 用途 | "不用管它" — Loader 已忽略 |
| Q2 | cnt | `sub_bnd_box` 是否为点标注 | 已换数据集 → **叶尖计数 VOC XML**，bbox 作为点标注承载 |
| Q3 | cnt | 空 .mat 是 "count=0" 还是 "未标注" | 不再适用（旧数据集已被替换） |
| Q4 | seg | 4 个 class_id 的具体器官语义 | **Background / Head / Stem / Leaf** |

### 仍待澄清（低优先级）

| # | 任务 | 问题 |
|---|---|---|
| Q5 | cls | "_c"/"_cd" 后缀的含义；同主名是否同 split |
| Q6 | 全局 | 是否允许使用 ImageNet 预训练权重（默认允许，提一下即可） |
