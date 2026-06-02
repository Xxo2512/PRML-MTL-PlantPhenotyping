"""极简 yaml 配置加载, 支持 _base_ 继承; 支持属性访问与下标访问。"""
from __future__ import annotations
import os
import yaml
from typing import Any


class Cfg(dict):
    """既能 cfg.train.lr 也能 cfg['train']['lr']; 写入用 dict 操作。"""

    def __getattr__(self, k):
        if k.startswith('__') and k.endswith('__'):
            raise AttributeError(k)
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = v


def _wrap(d: Any) -> Any:
    if isinstance(d, dict):
        return Cfg({k: _wrap(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_wrap(x) for x in d]
    return d


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str) -> Cfg:
    path = os.path.abspath(path)
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    base = data.pop('_base_', None)
    if base:
        base_path = os.path.normpath(os.path.join(os.path.dirname(path), base))
        base_data = load_config(base_path)
        # base_data 已被 _wrap 成 Cfg, 这里转回 dict 再合
        from copy import deepcopy
        data = _deep_merge(deepcopy(dict(base_data)), data)
    # 后处理 backbone 完整名
    bb = data.setdefault('model', {}).get('backbone', 'swin_tiny')
    if bb == 'swin_tiny':
        data['model']['backbone_full_name'] = 'swin_tiny_patch4_window7_224'
    elif bb == 'swin_small':
        data['model']['backbone_full_name'] = 'swin_small_patch4_window7_224'
    else:
        data['model']['backbone_full_name'] = bb
    data['model']['pretrained_flag'] = bool(data['model'].get('pretrained'))
    return _wrap(data)
