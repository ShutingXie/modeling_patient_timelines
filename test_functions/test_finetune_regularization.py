"""Tests for fine-tune regularization helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.transformer_encoder import build_model_from_config
from src.training.train_transformer import (
    build_finetune_optimizer,
    set_encoder_trainable,
)


def _tiny_config(**training_overrides) -> dict:
    training = {
        "learning_rate": 0.0003,
        "weight_decay": 0.01,
        "use_param_groups": False,
        **training_overrides,
    }
    return {
        "model": {"d_model": 32, "num_layers": 1, "num_heads": 4, "dim_feedforward": 64, "dropout": 0.1},
        "data": {"max_seq_len": 32},
        "training": training,
    }


def _build_model(config: dict) -> torch.nn.Module:
    return build_model_from_config(
        config,
        vocab_size=20,
        num_labels=3,
        num_modalities=5,
        num_time_buckets=4,
        num_age_buckets=4,
        include_mlm_head=False,
        include_classifier=True,
    )


def test_set_encoder_trainable_freezes_encoder_only() -> None:
    model = _build_model(_tiny_config())
    set_encoder_trainable(model, False)

    for name, param in model.named_parameters():
        if name.startswith("encoder."):
            assert not param.requires_grad, name
        elif name.startswith("classifier."):
            assert param.requires_grad, name


def test_set_encoder_trainable_unfreezes_encoder() -> None:
    model = _build_model(_tiny_config())
    set_encoder_trainable(model, False)
    set_encoder_trainable(model, True)

    for name, param in model.named_parameters():
        assert param.requires_grad, name


def test_build_finetune_optimizer_single_group() -> None:
    model = _build_model(_tiny_config())
    optimizer = build_finetune_optimizer(model, _tiny_config()["training"])

    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == 0.0003


def test_build_finetune_optimizer_param_groups() -> None:
    train_cfg = {
        "use_param_groups": True,
        "encoder_lr": 0.0001,
        "classifier_lr": 0.0003,
        "weight_decay": 0.05,
    }
    model = _build_model(_tiny_config(**train_cfg))
    optimizer = build_finetune_optimizer(model, train_cfg)

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["lr"] == 0.0001
    assert optimizer.param_groups[1]["lr"] == 0.0003
    assert optimizer.param_groups[0]["weight_decay"] == 0.05


def test_build_model_from_config_classifier_dropout_override() -> None:
    config = _tiny_config(classifier_dropout=0.25, encoder_dropout=0.15)
    model = build_model_from_config(
        config,
        vocab_size=20,
        num_labels=3,
        num_modalities=5,
        num_time_buckets=4,
        num_age_buckets=4,
        include_classifier=True,
    )
    classifier_dropout = model.classifier.classifier[2].p
    encoder_dropout = model.encoder.encoder.layers[0].dropout.p
    assert classifier_dropout == 0.25
    assert encoder_dropout == 0.15


if __name__ == "__main__":
    test_set_encoder_trainable_freezes_encoder_only()
    test_set_encoder_trainable_unfreezes_encoder()
    test_build_finetune_optimizer_single_group()
    test_build_finetune_optimizer_param_groups()
    test_build_model_from_config_classifier_dropout_override()
    print("All fine-tune regularization tests passed.")
