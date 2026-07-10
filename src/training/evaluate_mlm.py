"""MLM evaluation helpers."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from src.data.mlm_utils import MLM_IGNORE_INDEX
from src.models.transformer_encoder import PatientTimelineTransformer


@torch.no_grad()
def evaluate_mlm(
    model: PatientTimelineTransformer,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: torch.nn.Module,
    use_amp: bool = False,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_masked = 0
    n_batches = 0

    for batch in dataloader:
        mlm_input_ids = batch["mlm_input_ids"].to(device)
        modality_ids = batch["modality_ids"].to(device)
        time_bucket_ids = batch["time_bucket_ids"].to(device)
        age_bucket_ids = batch["age_bucket_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        mlm_labels = batch["mlm_labels"].to(device)

        if use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda"):
                logits = model(
                    mlm_input_ids,
                    modality_ids,
                    time_bucket_ids,
                    age_bucket_ids,
                    attention_mask,
                    task="mlm",
                )
                loss = loss_fn(logits.view(-1, logits.size(-1)), mlm_labels.view(-1))
        else:
            logits = model(
                mlm_input_ids,
                modality_ids,
                time_bucket_ids,
                age_bucket_ids,
                attention_mask,
                task="mlm",
            )
            loss = loss_fn(logits.view(-1, logits.size(-1)), mlm_labels.view(-1))

        total_loss += loss.item()
        n_batches += 1

        mask = mlm_labels != MLM_IGNORE_INDEX
        if mask.any():
            preds = logits.argmax(dim=-1)
            total_correct += (preds[mask] == mlm_labels[mask]).sum().item()
            total_masked += mask.sum().item()

    metrics: dict[str, Any] = {
        "mlm_loss": total_loss / max(n_batches, 1),
        "mlm_accuracy": total_correct / max(total_masked, 1),
        "n_masked_tokens": total_masked,
    }
    return metrics
