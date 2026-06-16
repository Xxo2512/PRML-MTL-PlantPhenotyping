"""训练入口: smoke / 短训 / single-task baseline / 全量训练通用。

用法:
  # smoke (5 step)
  python train.py --config configs/method/vanilla.yaml --steps 5

  # 短训 1 epoch + 末态自动落盘
  python train.py --config configs/method/vanilla.yaml --epochs 1

  # MTLoRA 全量
  python train.py --config configs/method/mtlora.yaml --epochs 50

  # 单任务 baseline (只在某 task 数据上训, 其它 task head 不实例化)
  python train.py --config configs/method/vanilla.yaml --epochs 5 --single_task cls

  # 自定义 tag (会用于 ckpt / log 子目录)
  python train.py --config configs/method/mtlora.yaml --epochs 50 --tag mtlora_50ep_ps
"""
from __future__ import annotations
import argparse
import os
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
                   help='smoke: 跑指定 step 数后退出 (覆盖 epochs)')
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--log_every', type=int, default=20)
    p.add_argument('--save_every', type=int, default=0,
                   help='中途按 step 数保存 (0=只保存末态)')
    p.add_argument('--tag', default=None,
                   help='覆盖 cfg.exp_name; 用于 ckpt 路径与日志目录')
    p.add_argument('--single_task', default=None, choices=[None, 'seg', 'det', 'cnt', 'cls'],
                   help='只训该单任务 (其它 task disable)')
    p.add_argument('--no_save', action='store_true', help='不保存 ckpt (smoke 时用)')
    p.add_argument('--num_workers', type=int, default=None,
                   help='覆盖 cfg.data.num_workers (Windows smoke 建议设 0)')
    p.add_argument('--batch_per_task', type=int, default=None,
                   help='覆盖 cfg.data.batch_per_task')
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
    if args.tag:
        cfg.exp_name = args.tag
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers
    if args.batch_per_task is not None:
        cfg.data.batch_per_task = args.batch_per_task
    if args.single_task:
        for t in ('seg', 'det', 'cnt', 'cls'):
            cfg.tasks[t].enabled = (t == args.single_task)
        cfg.exp_name = f'single_{args.single_task}_{cfg.exp_name}'

    device = torch.device(args.device)
    print(f'[cfg] exp={cfg.exp_name} method={cfg.method} backbone={cfg.model.backbone} '
          f'sched={cfg.data.scheduler} lossw={cfg.loss.weighting}')

    # ---- model
    model = build_mtl_model(cfg).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    n_total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'[model] trainable: {n_train:.2f}M / total: {n_total:.2f}M')

    # ---- data
    train_loaders = build_all_loaders(cfg, split='train')
    enabled = list(train_loaders.keys())
    print(f'[data] enabled tasks: {enabled}')
    print('[data] sizes (batches): ' + ', '.join(f'{t}={len(l)}' for t, l in train_loaders.items()))
    alpha = cfg.data.get('proportional_alpha', 0.5) if hasattr(cfg.data, 'get') else 0.5
    sampler = CrossDatasetSampler(train_loaders, mode=cfg.data.scheduler, alpha=alpha)

    # ---- loss aggregator
    loss_mode = cfg.loss.weighting if cfg.loss.weighting in ('uniform', 'uncertainty') else 'uniform'
    loss_agg = LossAggregator(mode=loss_mode, tasks=tuple(enabled)).to(device)

    # ---- optim
    params = [p for p in model.parameters() if p.requires_grad] + list(loss_agg.parameters())
    optim = AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    # ---- ckpt 路径
    out_dir = os.path.join('checkpoints', cfg.exp_name)
    os.makedirs(out_dir, exist_ok=True)

    def save_ckpt(step: int, name: str):
        if args.no_save:
            return
        path = os.path.join(out_dir, f'{name}.pt')
        torch.save({
            'step': step,
            'model_state': model.state_dict(),
            'loss_agg_state': loss_agg.state_dict(),
            'cfg': dict(cfg),
        }, path)
        print(f'[ckpt] saved -> {path}')

    # ---- loop
    steps_per_epoch = len(sampler)
    steps_target = args.steps if args.steps else (args.epochs or 1) * steps_per_epoch
    print(f'[run] steps_per_epoch={steps_per_epoch} steps_target={steps_target}')

    step = 0
    t0 = time.time()
    model.train()
    try:
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

                if args.save_every > 0 and step > 0 and step % args.save_every == 0:
                    save_ckpt(step, f'step{step}')
                step += 1
    except KeyboardInterrupt:
        print('\n[interrupted] saving last ckpt before exit')
        save_ckpt(step, 'last')
        raise

    # 末态保存
    save_ckpt(step, 'last')
    print('[done]')


if __name__ == '__main__':
    main()
