"""Checkpoint loading utilities for MEM pretrain -> fine-tune."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def remap_legacy_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map old flat keys to encoder.* / classifier.* structure."""
    remapped: dict[str, torch.Tensor] = {}
    embedding_prefixes = (
        "token_emb.",
        "modality_emb.",
        "time_bucket_emb.",
        "age_bucket_emb.",
        "position_emb.",
    )
    for key, value in state_dict.items():
        if key.startswith("mlm_head."):
            remapped[key] = value
        elif any(key.startswith(prefix) for prefix in embedding_prefixes):
            remapped[f"encoder.{key}"] = value
        elif key.startswith("encoder."):
            remapped[f"encoder.encoder.{key[len('encoder.'):]}"] = value
        elif key.startswith("classifier."):
            remapped[f"classifier.classifier.{key[len('classifier.'):]}"] = value
        else:
            remapped[key] = value
    return remapped


def _filter_encoder_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v for k, v in state_dict.items() if k.startswith("encoder.")}


def load_pretrained_encoder(
    model: nn.Module,
    pretrain_ckpt_path: str | Path,
    device: str | torch.device = "cpu",
) -> list[str]:
    """Load encoder weights from a pretrain checkpoint into a fine-tune model."""
    ckpt = torch.load(pretrain_ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    if not any(k.startswith("encoder.") for k in state_dict):
        state_dict = remap_legacy_state_dict(state_dict)

    encoder_state = _filter_encoder_keys(state_dict)
    if not encoder_state:
        raise ValueError(f"No encoder.* keys found in checkpoint: {pretrain_ckpt_path}")

    missing, unexpected = model.load_state_dict(encoder_state, strict=False)
    encoder_missing = [k for k in missing if k.startswith("encoder.")]
    if encoder_missing:
        raise RuntimeError(
            f"Failed to load encoder weights. Missing keys: {encoder_missing[:5]}"
        )

    loaded_keys = sorted(encoder_state.keys())
    print(f"Loaded {len(loaded_keys)} encoder keys from {pretrain_ckpt_path}")
    if unexpected:
        print(f"Unexpected keys ignored: {unexpected[:5]}")
    return loaded_keys


def filter_finetune_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Keep only encoder + classifier weights for fine-tune checkpoints."""
    return {
        k: v
        for k, v in state_dict.items()
        if k.startswith("encoder.") or k.startswith("classifier.")
    }
