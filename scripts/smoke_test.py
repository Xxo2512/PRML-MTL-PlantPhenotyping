"""快速 smoke test: 不依赖 GPU, 只跑 5 step CPU vanilla forward+backward, 验证接口对齐。

用法:
  python scripts/smoke_test.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils import load_config, LossAggregator
from datasets import build_all_loaders, CrossDatasetSampler
from models import build_mtl_model


def main():
    cfg = load_config('configs/method/vanilla.yaml')
    # CPU smoke: bs 极小, 只装 4 个任务每个 1 batch
    cfg.data.batch_per_task = 1
    cfg.data.num_workers = 0
    cfg.train.lr = 1e-4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[smoke] device = {device}')

    model = build_mtl_model(cfg).to(device)
    print(f'[smoke] model OK, params={sum(p.numel() for p in model.parameters())/1e6:.1f}M')

    loaders = build_all_loaders(cfg, split='val')
    print(f'[smoke] loaders OK: ' + ', '.join(f'{t}={len(l)}' for t, l in loaders.items()))
    sampler = CrossDatasetSampler(loaders, mode='ps', alpha=0.5, length=8)

    loss_agg = LossAggregator('uncertainty', tasks=tuple(loaders.keys())).to(device)
    params = [p for p in model.parameters() if p.requires_grad] + list(loss_agg.parameters())
    optim = torch.optim.AdamW(params, lr=1e-4)

    model.train()
    for i, batch in enumerate(sampler):
        # to device
        def mv(x):
            if isinstance(x, dict): return {k: mv(v) for k, v in x.items()}
            if isinstance(x, list): return [mv(y) for y in x]
            if torch.is_tensor(x): return x.to(device)
            return x
        batch = mv(batch)
        out = model(batch)
        per_task_loss = {t: o['loss'] for t, o in out.items() if 'loss' in o}
        L = loss_agg(per_task_loss)
        optim.zero_grad(); L.backward(); optim.step()
        items = ' '.join(f'{t}={v.item():.3f}' for t, v in per_task_loss.items())
        print(f'[smoke step {i}] L={L.item():.3f}  {items}')
    print('[smoke] PASSED')


if __name__ == '__main__':
    main()
