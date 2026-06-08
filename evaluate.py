"""评测入口: 按 config 构建 model + val loaders, 跑 4 个任务指标。

用法:
  python evaluate.py --config configs/method/vanilla.yaml
  python evaluate.py --config configs/method/vanilla.yaml --ckpt checkpoints/xxx.pt
  python evaluate.py --config configs/method/vanilla.yaml --tag vanilla_5ep_ps
"""
from __future__ import annotations
import argparse
import csv
import os
import time
from datetime import datetime
import torch

from utils import load_config, evaluate_model
from datasets import build_all_loaders
from models import build_mtl_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--ckpt', default=None, help='可选: 模型 .pt 权重路径')
    p.add_argument('--tag', default=None, help='结果行的 exp 标签 (默认用 cfg.exp_name)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--csv', default='logs/results.csv', help='结果追加到此 csv')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)

    print(f'[cfg] method={cfg.method} backbone={cfg.model.backbone} exp={cfg.exp_name}')
    model = build_mtl_model(cfg).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f'[model] trainable params: {n_train:.2f} M')

    if args.ckpt:
        sd = torch.load(args.ckpt, map_location=device)
        sd = sd.get('model_state', sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f'[ckpt] loaded {args.ckpt}; missing={len(missing)} unexpected={len(unexpected)}')

    val_loaders = build_all_loaders(cfg, split='val')
    enabled = list(val_loaders.keys())
    print(f'[data] val: ' + ', '.join(f'{t}={len(l.dataset)}' for t, l in val_loaders.items()))

    t0 = time.time()
    flat = evaluate_model(model, val_loaders, device)
    dt = time.time() - t0
    print(f'[eval] done in {dt:.1f}s')
    print('-' * 60)
    for k, v in flat.items():
        print(f'  {k:24s}  {v:.4f}')
    print('-' * 60)

    # 追加到 csv
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    tag = args.tag or cfg.exp_name
    row = {
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'tag':  tag,
        'method': cfg.method,
        'backbone': cfg.model.backbone,
        'scheduler': cfg.data.scheduler,
        'weighting': cfg.loss.weighting,
        'ckpt': args.ckpt or '-',
        **flat,
    }
    new_file = not os.path.isfile(args.csv)
    with open(args.csv, 'a', newline='', encoding='utf-8') as f:
        # 列名: 取 union, 用本行的 keys
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)
    print(f'[csv] appended -> {args.csv} (tag={tag})')


if __name__ == '__main__':
    main()
