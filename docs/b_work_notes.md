# B 工作说明

## 已完成模块

- `datasets/wheat_det.py`: 读取检测数据集，支持 `images/<split>` + `labels/<split>` 的 YOLO txt 标注。
- `datasets/input_adapter.py`: 统一把各任务原始样本变成 `image / targets / meta`；检测任务使用 keep-ratio resize + padding 到 384。
- `utils/collate.py`: 提供 `task_collate(batch, task)`，检测框和计数点这类变长目标会保留为 list。
- `datasets/registry.py`: 注册 `TASKS`，目前检测任务可以直接 build dataset/loader，其它任务等待对应成员实现。
- `models/heads/det_fcos.py`: 轻量 FCOS-style 检测头，可 forward/backward。
- `models/mtl/tadformer.py`: TADFormer 第一版，使用 task embedding 生成每个 stage 的 `gamma/beta` 调制参数。

## 数据格式确认

检测数据 `detect_dataset.zip` 为 YOLO 格式：

```text
class cx cy w h
```

其中 `cx/cy/w/h` 是相对图像宽高的归一化坐标，`WheatDetectionDataset` 会转换为绝对 `xyxy`。

计数数据 `count_dataset.zip` 已更新为 XML 标注：

```text
count_dataset/
  images/train/*.jpg
  annotations/train/*.xml
```

XML 内每个 `<object>` 表示一个 `tip`，可由 bbox 中心转换为计数点；这部分正式实现由 D 接手时按新格式写。

## Smoke Test

只测代码闭环：

```powershell
.\.venv\Scripts\python.exe scripts\smoke_b.py
```

如果检测数据已解压到 `data/detect_dataset`：

```powershell
.\.venv\Scripts\python.exe scripts\smoke_b.py --data-root data\detect_dataset
```
