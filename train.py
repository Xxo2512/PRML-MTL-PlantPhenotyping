"""W13 训练入口: 跑通 vanilla baseline 的 smoke test 与短训。

用法:
  python train.py --config configs/method/vanilla.yaml --steps 100      # smoke
  python train.py --config configs/method/vanilla.yaml --epochs 5       # 短训
"""
from __future__ import annotations
import argparse
import time
import torch
from torch.optim import AdamW

from utils import load_config, LossAggregator
from datasets import build_all_loaders, CrossDatasetSampler
from models import build_mtl_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--steps', type=int, default=None,
                   help='smoke test: 跑指定 step 数后退出 (覆盖 epochs)')
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--log_every', type=int, default=10)
    return p.parse_args()


def to_device(batch, device):
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [to_device(x, device) for x in batch]
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    return batch


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)

    print(f'[cfg] method={cfg.method} backbone={cfg.model.backbone} '
          f'sched={cfg.data.scheduler} lossw={cfg.loss.weighting}')

    # ---- model
    model = build_mtl_model(cfg).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f'[model] trainable params: {n_train:.2f} M')

    # ---- data
    train_loaders = build_all_loaders(cfg, split='train')
    enabled = list(train_loaders.keys())
    print(f'[data] enabled tasks: {enabled}')
    print('[data] sizes: ' + ', '.join(f'{t}={len(l)}' for t, l in train_loaders.items()))
    alpha = cfg.data.get('proportional_alpha', 0.5) if hasattr(cfg.data, 'get') else 0.5
    sampler = CrossDatasetSampler(train_loaders, mode=cfg.data.scheduler, alpha=alpha)

    # ---- loss aggregator
    loss_mode = cfg.loss.weighting if cfg.loss.weighting in ('uniform', 'uncertainty') else 'uniform'
    loss_agg = LossAggregator(mode=loss_mode, tasks=tuple(enabled)).to(device)

    # ---- optim
    params = [p for p in model.parameters() if p.requires_grad] + list(loss_agg.parameters())
    optim = AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    # ---- loop
    steps_per_epoch = len(sampler)
    steps_target = args.steps if args.steps else (args.epochs or 1) * steps_per_epoch
    print(f'[run] steps_per_epoch={steps_per_epoch} steps_target={steps_target}')
    step = 0
    t0 = time.time()
    model.train()
    while step < steps_target:
        for batch in sampler:
            if step >= steps_target:
                break
            batch = to_device(batch, device)
            out = model(batch)
            per_task_loss = {t: o['loss'] for t, o in out.items() if 'loss' in o}
            if not per_task_loss:
                step += 1
                continue
            loss = loss_agg(per_task_loss)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
            optim.step()

            if step % args.log_every == 0:
                items_str = ' '.join(f'{k}={v.item():.3f}' for k, v in per_task_loss.items())
                rate = (step + 1) / (time.time() - t0 + 1e-8)
                print(f'[step {step:5d}] L={loss.item():.3f} {items_str} ({rate:.2f} it/s)')
            step += 1
    print('[done]')


if __name__ == '__main__':
    main()
