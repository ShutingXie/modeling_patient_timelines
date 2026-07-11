#!/usr/bin/env bash
# Run tier A then tier B/C fine-tune regularization experiments.
# Requires a pretrain checkpoint trained with the same model config (e.g. from Colab).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PRETRAIN_CKPT="${PRETRAIN_CKPT:-outputs/checkpoints/pretrain_best.pt}"

echo "=== reg_a (config-only regularization) ==="
python scripts/train_transformer.py \
  --config configs/transformer_finetune_reg_a.yaml \
  --pretrained-checkpoint "$PRETRAIN_CKPT"

echo "=== reg_bc (full regularization) ==="
python scripts/train_transformer.py \
  --config configs/transformer_finetune_reg.yaml \
  --pretrained-checkpoint "$PRETRAIN_CKPT"

echo "=== comparison ==="
python scripts/compare_reg_experiments.py
