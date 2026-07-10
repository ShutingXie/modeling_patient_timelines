"""Transformer encoder with CLS pooling for multi-label condition prediction."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


class PatientTimelineTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_modalities: int,
        num_time_buckets: int,
        num_age_buckets: int,
        num_labels: int,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.modality_emb = nn.Embedding(num_modalities, d_model)
        self.time_bucket_emb = nn.Embedding(num_time_buckets, d_model)
        self.age_bucket_emb = nn.Embedding(num_age_buckets, d_model)
        self.position_emb = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_labels),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        modality_ids: torch.Tensor,
        time_bucket_ids: torch.Tensor,
        age_bucket_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        x = (
            self.token_emb(input_ids)
            + self.modality_emb(modality_ids)
            + self.time_bucket_emb(time_bucket_ids)
            + self.age_bucket_emb(age_bucket_ids)
            + self.position_emb(positions)
        )
        x = x * math.sqrt(self.d_model)

        # True = ignore position in PyTorch transformer
        key_padding_mask = ~attention_mask
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        cls_hidden = x[:, 0, :]
        return self.classifier(cls_hidden)


def build_model_from_config(config: dict[str, Any], vocab_size: int, num_labels: int, num_modalities: int, num_time_buckets: int, num_age_buckets: int) -> PatientTimelineTransformer:
    model_cfg = config.get("model", config)
    data_cfg = config.get("data", config)
    return PatientTimelineTransformer(
        vocab_size=vocab_size,
        num_modalities=num_modalities,
        num_time_buckets=num_time_buckets,
        num_age_buckets=num_age_buckets,
        num_labels=num_labels,
        d_model=model_cfg.get("d_model", 256),
        num_layers=model_cfg.get("num_layers", 4),
        num_heads=model_cfg.get("num_heads", 8),
        dim_feedforward=model_cfg.get("dim_feedforward", 1024),
        dropout=model_cfg.get("dropout", 0.1),
        max_seq_len=data_cfg.get("max_seq_len", 1024),
    )
