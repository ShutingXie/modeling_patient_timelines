# EHR Timeline Transformer

Masked Event Modeling (MEM) pretrain → fine-tune → test predictions on synthetic patient timelines.

## Requirements

- Python 3.10+
- CUDA GPU strongly recommended for pretrain and fine-tune (CPU works but is slow)
- Raw data is **not** in git (`data/` and `outputs/` are gitignored)

## Setup

```bash
git clone https://github.com/ShutingXie/modeling_patient_timelines.git
cd modeling_patient_timelines

python -m venv .venv
source .venv/bin/activate 
pip install -r requirements.txt
wandb login   # optional; use --no-wandb on training scripts to skip
```

Default W&B project: `ehr-timeline-transformer` ([configs/transformer.yaml](configs/transformer.yaml)).

## Data layout

Place the official Synthea splits under `data/`:

```
data/
├── patient_splits.csv
├── target_conditions.csv
├── test_anchors.csv
├── train_val/          # full CSV tables for train + val patients
└── test/               # test tables truncated at each patient's
```

Sanity check: `test -f data/patient_splits.csv && test -d data/train_val && test -d data/test`

If data lives elsewhere, pass `--data-dir /path/to/data` to `prepare_data.py`, `train_pretrain.py`, and `train_transformer.py`.

## Pipeline

Run from the repo root:

```bash
# 1. Prepare processed artifacts (vocab, events, labels)
python scripts/prepare_data.py --config configs/transformer.yaml

# 2. MEM pretrain (official train split only)
python scripts/train_pretrain.py \
    --config configs/transformer.yaml \
    --mask-prob 0.20 \
    --pretrain-epochs 50

# 3. Fine-tune (train + val for model selection)
python scripts/train_transformer.py \
    --config configs/transformer.yaml \
    --pretrained-checkpoint outputs/checkpoints/pretrain_best.pt

# 4. Generate test-set probabilities
python scripts/make_predictions.py \
    --checkpoint outputs/checkpoints/best_model.pt \
    --output outputs/predictions.csv
```


| Step      | Main output                                                             |
| --------- | ----------------------------------------------------------------------- |
| Prepare   | `outputs/processed/` (includes `[MASK]` in `vocab.json`)                |
| Pretrain  | `outputs/checkpoints/pretrain_best.pt`                                  |
| Fine-tune | `outputs/checkpoints/best_model.pt`, `outputs/metrics/val_metrics.json` |
| Predict   | `outputs/predictions.csv`                                               |


Pretrain uses official **train** only (internal `pretrain_val` in `outputs/processed/pretrain_split.json`); fine-tune uses **train + val**; test labels are withheld. Use `--no-wandb` to skip W&B; omit `--pretrained-checkpoint` on fine-tune for a no-pretrain baseline. Hyperparameters: [configs/transformer.yaml](configs/transformer.yaml).

## Quick smoke test

Verify the install without full training ([configs/debug.yaml](configs/debug.yaml): 1 epoch, small model, W&B disabled):

```bash
python scripts/prepare_data.py --config configs/debug.yaml
python scripts/train_pretrain.py --config configs/debug.yaml --pretrain-epochs 1 --no-wandb
python scripts/train_transformer.py --config configs/debug.yaml --pretrained-checkpoint outputs/checkpoints/pretrain_best.pt --no-wandb
python scripts/make_predictions.py --checkpoint outputs/checkpoints/best_model.pt
```

## Repo layout

```
configs/                # YAML hyperparameters
scripts/                # CLI entry points
src/                    # library code
outputs/                # created at runtime (processed, checkpoints, metrics, predictions)
```

