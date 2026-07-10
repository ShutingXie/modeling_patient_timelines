"""Validation evaluation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.transformer_encoder import PatientTimelineTransformer
from src.training.metrics import compute_multilabel_metrics


@torch.no_grad()
def evaluate_model(
    model: PatientTimelineTransformer,
    dataloader: DataLoader,
    device: torch.device,
    target_codes: list[str],
    loss_fn: torch.nn.Module | None = None,
    use_amp: bool = False,
) -> dict[str, Any]:
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        modality_ids = batch["modality_ids"].to(device)
        time_bucket_ids = batch["time_bucket_ids"].to(device)
        age_bucket_ids = batch["age_bucket_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda"):
                logits = model(
                    input_ids, modality_ids, time_bucket_ids, age_bucket_ids, attention_mask, task="cls"
                )
                if loss_fn is not None:
                    total_loss += loss_fn(logits, labels).item()
        else:
            logits = model(
                input_ids, modality_ids, time_bucket_ids, age_bucket_ids, attention_mask, task="cls"
            )
            if loss_fn is not None:
                total_loss += loss_fn(logits, labels).item()

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())
        n_batches += 1

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    metrics = compute_multilabel_metrics(y_true, y_prob, target_codes)
    if loss_fn is not None and n_batches > 0:
        metrics["loss"] = total_loss / n_batches
    return metrics
