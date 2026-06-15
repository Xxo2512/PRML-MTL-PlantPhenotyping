"""TensorBoard visualization helpers for the four project tasks."""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

DEFAULT_PALETTE = torch.tensor(
    [
        [0.00, 0.00, 0.00],
        [0.90, 0.18, 0.13],
        [0.13, 0.55, 0.22],
        [0.12, 0.36, 0.95],
        [0.95, 0.70, 0.12],
        [0.65, 0.25, 0.90],
        [0.00, 0.75, 0.75],
        [0.95, 0.45, 0.70],
    ],
    dtype=torch.float32,
)


def _as_batch(images: torch.Tensor) -> torch.Tensor:
    if images.dim() == 3:
        images = images.unsqueeze(0)
    if images.dim() != 4:
        raise ValueError(f"expected image tensor [B,C,H,W] or [C,H,W], got {tuple(images.shape)}")
    return images


def denormalize_images(images: torch.Tensor) -> torch.Tensor:
    images = _as_batch(images).detach().float().cpu()
    mean = IMAGENET_MEAN.to(images)
    std = IMAGENET_STD.to(images)
    return (images * std + mean).clamp(0.0, 1.0)


def _palette(num_classes: int, device: torch.device) -> torch.Tensor:
    if num_classes <= DEFAULT_PALETTE.shape[0]:
        return DEFAULT_PALETTE[:num_classes].to(device)
    repeats = (num_classes + DEFAULT_PALETTE.shape[0] - 1) // DEFAULT_PALETTE.shape[0]
    return DEFAULT_PALETTE.repeat(repeats, 1)[:num_classes].to(device)


def _logits_from_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        pred = output.get("pred", output)
        if isinstance(pred, torch.Tensor):
            return pred
        if isinstance(pred, dict) and "logits" in pred:
            return pred["logits"]
        if "logits" in output:
            return output["logits"]
    raise ValueError("could not find segmentation/classification logits in output")


def make_seg_overlay(
    images: torch.Tensor,
    logits: torch.Tensor,
    masks: Optional[torch.Tensor] = None,
    classes: Optional[list[str]] = None,
    alpha: float = 0.45,
) -> torch.Tensor:
    """Return RGB overlays for segmentation predictions.

    ``classes`` is accepted for API stability; class names are consumed by
    callers when they build TensorBoard captions.
    """
    del classes
    base = denormalize_images(images)
    pred = logits.detach().cpu().argmax(dim=1)
    if pred.shape[-2:] != base.shape[-2:]:
        pred = F.interpolate(pred.unsqueeze(1).float(), size=base.shape[-2:], mode="nearest").squeeze(1).long()

    colors = _palette(max(int(pred.max().item()) + 1, logits.shape[1]), base.device)
    color_mask = colors[pred.clamp_min(0)].permute(0, 3, 1, 2)
    overlay = base * (1.0 - alpha) + color_mask * alpha

    if masks is not None:
        gt = masks.detach().cpu().long()
        if gt.dim() == 2:
            gt = gt.unsqueeze(0)
        if gt.shape[-2:] != base.shape[-2:]:
            gt = F.interpolate(gt.unsqueeze(1).float(), size=base.shape[-2:], mode="nearest").squeeze(1).long()
        gt_color = colors[gt.clamp(min=0, max=colors.shape[0] - 1)].permute(0, 3, 1, 2)
        overlay = torch.cat([overlay.clamp(0, 1), (base * 0.55 + gt_color * 0.45).clamp(0, 1)], dim=0)
    return overlay.clamp(0.0, 1.0)


def make_det_overlay(
    images: torch.Tensor,
    boxes: torch.Tensor,
    scores: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    score_thresh: float = 0.05,
) -> torch.Tensor:
    base = denormalize_images(images)
    drawn = []
    for idx, image in enumerate(base):
        canvas = image.clone()
        cur_boxes = boxes[idx].detach().cpu() if boxes is not None and idx < boxes.shape[0] else torch.empty(0, 4)
        cur_scores = scores[idx].detach().cpu() if scores is not None and idx < scores.shape[0] else None
        cur_labels = labels[idx].detach().cpu() if labels is not None and idx < labels.shape[0] else None
        del cur_labels
        if cur_scores is not None:
            keep = cur_scores >= score_thresh
            cur_boxes = cur_boxes[keep]
            cur_scores = cur_scores[keep]
        height, width = image.shape[-2:]
        cur_boxes = cur_boxes.clamp(min=0, max=max(height, width)).float()
        if cur_boxes.numel() > 0:
            for box in cur_boxes:
                x1, y1, x2, y2 = box.round().long().tolist()
                x1, x2 = sorted((max(0, min(width - 1, x1)), max(0, min(width - 1, x2))))
                y1, y2 = sorted((max(0, min(height - 1, y1)), max(0, min(height - 1, y2))))
                if x2 <= x1 or y2 <= y1:
                    continue
                canvas[:, y1 : min(y1 + 2, height), x1 : x2 + 1] = torch.tensor([0.1, 1.0, 0.1]).view(3, 1, 1)
                canvas[:, max(y2 - 1, 0) : y2 + 1, x1 : x2 + 1] = torch.tensor([0.1, 1.0, 0.1]).view(3, 1, 1)
                canvas[:, y1 : y2 + 1, x1 : min(x1 + 2, width)] = torch.tensor([0.1, 1.0, 0.1]).view(3, 1, 1)
                canvas[:, y1 : y2 + 1, max(x2 - 1, 0) : x2 + 1] = torch.tensor([0.1, 1.0, 0.1]).view(3, 1, 1)
        drawn.append(canvas.clamp(0.0, 1.0))
    return torch.stack(drawn, dim=0)


def _colorize_heatmap(heatmap: torch.Tensor) -> torch.Tensor:
    heat = heatmap.detach().float().cpu()
    if heat.dim() == 4:
        heat = heat[:, 0]
    elif heat.dim() == 2:
        heat = heat.unsqueeze(0)
    heat = heat - heat.amin(dim=(-2, -1), keepdim=True)
    heat = heat / heat.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    red = heat
    green = (1.0 - (heat - 0.5).abs() * 2.0).clamp(0.0, 1.0)
    blue = 1.0 - heat
    return torch.stack([red, green, blue], dim=1)


def make_cnt_heatmap(images: torch.Tensor, density: torch.Tensor, alpha: float = 0.45) -> torch.Tensor:
    base = denormalize_images(images)
    heat = _colorize_heatmap(density)
    if heat.shape[-2:] != base.shape[-2:]:
        heat = F.interpolate(heat, size=base.shape[-2:], mode="bilinear", align_corners=False)
    return (base * (1.0 - alpha) + heat * alpha).clamp(0.0, 1.0)


def make_cls_gradcam(model: torch.nn.Module, batch: Dict[str, Any], task: str = "cls") -> torch.Tensor:
    """Create a Grad-CAM overlay for classification batches.

    The hook captures the shared backbone's ``s4`` feature map. If gradients
    are unavailable, the function returns denormalized images so logging keeps
    running during smoke tests.
    """
    task_batch = batch.get(task, batch)
    images = task_batch["image"]
    was_training = model.training
    model.eval()
    activations: Dict[str, torch.Tensor] = {}

    def _hook(_module, _inputs, output):
        if isinstance(output, dict) and "s4" in output:
            act = output["s4"]
            act.retain_grad()
            activations["s4"] = act

    handle = model.backbone.register_forward_hook(_hook)
    try:
        model.zero_grad(set_to_none=True)
        task_batch = dict(task_batch)
        task_batch["image"] = images.detach().clone().requires_grad_(True)
        out = model({task: task_batch})[task]
        logits = _logits_from_output(out)
        score = logits.max(dim=-1).values.sum()
        score.backward()
        act = activations.get("s4")
        if act is None or act.grad is None:
            return denormalize_images(images)
        weights = act.grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * act).sum(dim=1, keepdim=True).relu()
        cam = cam - cam.amin(dim=(-2, -1), keepdim=True)
        cam = cam / cam.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        heat = _colorize_heatmap(cam)
        base = denormalize_images(images)
        heat = F.interpolate(heat, size=base.shape[-2:], mode="bilinear", align_corners=False)
        return (base * 0.55 + heat * 0.45).clamp(0.0, 1.0)
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
        model.train(was_training)


def log_task_visuals(
    writer,
    task: str,
    batch: Dict[str, Any],
    output: Dict[str, Any],
    step: int,
    max_images: int = 4,
    model: Optional[torch.nn.Module] = None,
) -> None:
    images = batch["image"][:max_images]
    targets = batch.get("targets", {})

    if task == "seg":
        logits = _logits_from_output(output)[:max_images]
        masks = targets.get("mask")
        visuals = make_seg_overlay(images, logits, masks[:max_images] if masks is not None else None)
        writer.add_images("vis/seg_overlay", visuals, step)
    elif task == "det":
        pred = output.get("pred", output)
        visuals = make_det_overlay(
            images,
            pred.get("boxes", torch.empty(0, 0, 4))[:max_images],
            pred.get("scores", None)[:max_images] if pred.get("scores", None) is not None else None,
            pred.get("labels", None)[:max_images] if pred.get("labels", None) is not None else None,
        )
        writer.add_images("vis/det_boxes", visuals, step)
    elif task == "cnt":
        pred = output.get("pred", output)
        density = pred["density"] if isinstance(pred, dict) and "density" in pred else pred
        writer.add_images("vis/cnt_density", make_cnt_heatmap(images, density[:max_images]), step)
    elif task == "cls":
        if model is not None:
            visuals = make_cls_gradcam(model, {"cls": batch}, task="cls")[:max_images]
        else:
            visuals = denormalize_images(images)
        writer.add_images("vis/cls_gradcam", visuals, step)
    else:
        raise ValueError(f"unknown task {task!r}")
