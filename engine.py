"""
Training and evaluation utilities for MSMamba.
"""

from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from timm.utils import ModelEma

import utils


def compute_batch_iou(logits, targets, threshold=0.5):
    """
    Compute mean IoU over a training batch.

    Args:
        logits: Raw model predictions with shape (B, 1, H, W).
        targets: Binary ground-truth masks with the same shape as logits.
        threshold: Binarization threshold after sigmoid.

    Returns:
        Mean IoU over the batch.
    """

    if logits.shape != targets.shape:
        raise ValueError(
            f"logits and targets must have the same shape, "
            f"but got {logits.shape} and {targets.shape}."
        )

    targets = targets.float()
    if targets.max() > 1.0:
        targets = targets / 255.0
    targets = (targets > 0.5)

    preds = torch.sigmoid(logits) >= threshold

    preds = preds.flatten(1)
    targets = targets.flatten(1)

    intersection = (preds & targets).sum(dim=1).float()
    union = (preds | targets).sum(dim=1).float()

    return (intersection / (union + 1e-6)).mean()


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    amp_autocast,
    max_norm: Optional[float] = None,
    model_ema: Optional[ModelEma] = None,
    set_training_mode: bool = True,
    args=None,
):
    """
    Train MSMamba for one epoch.

    The model is expected to return:
        pred, target, loss
    during training.
    """

    model.train(set_training_mode)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr",
        utils.SmoothedValue(window_size=1, fmt="{value:.6f}"),
    )

    header = f"Epoch: [{epoch}]"
    print_freq = 20

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        images = batch["query_img"].to(device, non_blocking=True)
        masks = batch["query_mask"].to(device, non_blocking=True).float()
        sentences = batch["sentence"]

        with amp_autocast():
            pred, target, loss = model(images, sentences, masks, epoch=epoch)

        loss_value = float(loss.item())
        train_iou = compute_batch_iou(pred, target)

        optimizer.zero_grad(set_to_none=True)

        if loss_scaler != "none":
            is_second_order = (
                hasattr(optimizer, "is_second_order")
                and optimizer.is_second_order
            )
            loss_scaler(
                loss,
                optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=is_second_order,
            )
        else:
            loss.backward()
            if max_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(train_loss=loss_value)
        metric_logger.update(train_iou=train_iou)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {
        key: meter.global_avg
        for key, meter in metric_logger.meters.items()
    }


def _normalize_sentences(sentences):
    """
    Normalize the sentence field from the dataloader.

    Some datasets return a list of expressions per image, while others return
    one expression per image. This helper converts both formats into a nested
    list: List[List[str]].
    """

    normalized = []

    for item in sentences:
        if isinstance(item, (list, tuple)):
            normalized.append([str(sent) for sent in item])
        else:
            normalized.append([str(item)])

    return normalized


def _prepare_eval_batch(batch, device):
    """
    Expand image-level batches into expression-level batches.

    For datasets where each image has multiple referring expressions, each
    expression is evaluated separately while sharing the same image.
    """

    images = batch["query_img"].to(device, non_blocking=True)
    masks = batch["query_mask"].to(device, non_blocking=True).float()
    sentences = _normalize_sentences(batch["sentence"])

    expanded_images = []
    expanded_masks = []
    expanded_sentences = []
    original_gt_masks = []

    for batch_idx, expression_list in enumerate(sentences):
        num_expr = len(expression_list)

        expanded_images.append(
            images[batch_idx].unsqueeze(0).repeat(num_expr, 1, 1, 1)
        )
        expanded_masks.append(
            masks[batch_idx].unsqueeze(0).repeat(num_expr, 1, 1, 1)
        )
        expanded_sentences.extend(expression_list)

        original_gt = batch.get("org_gt", None)
        if original_gt is not None:
            gt_mask = original_gt[batch_idx]
        else:
            gt_mask = masks[batch_idx].detach().cpu().numpy()

        for _ in range(num_expr):
            original_gt_masks.append(gt_mask)

    expanded_images = torch.cat(expanded_images, dim=0)
    expanded_masks = torch.cat(expanded_masks, dim=0)

    return expanded_images, expanded_masks, expanded_sentences, original_gt_masks


def _compute_sample_metrics(pred, gt_mask, pr_thresholds):
    """
    Compute IoU, intersection, union, and Pr@X for one prediction.
    """

    target_size = gt_mask.shape[-2:] if hasattr(gt_mask, "shape") and len(gt_mask.shape) >= 2 else gt_mask.shape

    pred = F.interpolate(
        pred.unsqueeze(0),
        size=target_size,
        mode="bilinear",
        align_corners=True,
    )[0]

    pred_mask = pred.sigmoid().detach().cpu().numpy() > 0.5
    gt_mask = np.asarray(gt_mask)

    if gt_mask.max() > 1:
        gt_mask = gt_mask / 255.0
    gt_mask = gt_mask > 0.5

    pred_mask = np.squeeze(pred_mask)
    gt_mask = np.squeeze(gt_mask)

    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = intersection / (union + 1e-6)

    pr_results = {
        f"pr@{int(threshold * 100)}": 1.0 if iou >= threshold else 0.0
        for threshold in pr_thresholds
    }

    return iou, intersection, union, pr_results


@torch.no_grad()
def evaluate(data_loader, model, device, amp_autocast, log_every=10):
    """
    Evaluate MSMamba.

    Returns:
        A dictionary containing:
            - iou: mean IoU;
            - oiou: overall IoU;
            - pr@50, pr@60, pr@70, pr@80, pr@90.
    """

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    pr_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    pr_names = [f"pr@{int(threshold * 100)}" for threshold in pr_thresholds]

    for name in pr_names:
        metric_logger.add_meter(
            name,
            utils.SmoothedValue(window_size=1, fmt="{value:.4f}"),
        )

    model.eval()

    for batch in metric_logger.log_every(data_loader, log_every, header):
        images, masks, sentences, gt_masks = _prepare_eval_batch(batch, device)

        with amp_autocast():
            preds = model(images, sentences, masks)

        for pred, gt_mask in zip(preds, gt_masks):
            iou, intersection, union, pr_results = _compute_sample_metrics(
                pred,
                gt_mask,
                pr_thresholds,
            )

            metric_logger.meters["inter"].update(float(intersection))
            metric_logger.meters["union"].update(float(union))
            metric_logger.meters["iou"].update(float(iou))

            for name, value in pr_results.items():
                metric_logger.meters[name].update(value)

    metric_logger.synchronize_between_processes()

    mean_iou = metric_logger.iou.global_avg
    overall_iou = metric_logger.inter.global_avg / (
        metric_logger.union.global_avg + 1e-6
    )

    pr_results = {
        name: metric_logger.meters[name].global_avg
        for name in pr_names
    }

    print(f"* mIoU {mean_iou:.4f}  oIoU {overall_iou:.4f}")
    print("* " + "  ".join(f"{name} {value:.4f}" for name, value in pr_results.items()))

    return {
        "iou": mean_iou,
        "oiou": overall_iou,
        **pr_results,
    }
