"""Train the patient-timeline transformer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import load_json_mapping, load_processed_dataset
from src.data.vocab import Vocabulary
from src.models.checkpoint_utils import filter_finetune_state_dict, load_pretrained_encoder
from src.models.transformer_encoder import build_model_from_config
from src.training.evaluate import evaluate_model
from src.utils.seed import set_seed


def compute_pos_weights(
    labels_df: pd.DataFrame,
    target_codes: list[str],
    clip: float = 20.0,
) -> torch.Tensor:
    y = labels_df[target_codes].values
    n_pos = y.sum(axis=0)
    n_neg = len(y) - n_pos
    weights = np.where(n_pos > 0, n_neg / n_pos, 1.0)
    weights = np.clip(weights, 1.0, clip)
    return torch.tensor(weights, dtype=torch.float32)


def train_transformer(
    config: dict[str, Any],
    processed_dir: str | Path,
    checkpoint_dir: str | Path = "outputs/checkpoints",
    metrics_dir: str | Path = "outputs/metrics",
    plots_dir: str | Path = "outputs/plots",
    use_wandb: bool = True,
    pretrained_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    set_seed(config.get("seed", 42))
    processed_dir = Path(processed_dir)
    checkpoint_dir = Path(checkpoint_dir)
    metrics_dir = Path(metrics_dir)
    plots_dir = Path(plots_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = config.get("training", config)
    data_cfg = config.get("data", config)
    wandb_cfg = config.get("wandb", {})

    train_ds = load_processed_dataset(processed_dir, "train", config, include_labels=True)
    val_ds = load_processed_dataset(processed_dir, "val", config, include_labels=True)
    target_codes = load_json_mapping(processed_dir / "target_codes.json")

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.get("batch_size", 32),
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.get("batch_size", 32),
        shuffle=False,
        num_workers=0,
    )

    vocab = Vocabulary.load(processed_dir / "vocab.json")
    modality_to_id = load_json_mapping(processed_dir / "modality_to_id.json")
    time_bucket_to_id = load_json_mapping(processed_dir / "time_bucket_to_id.json")
    age_bucket_to_id = load_json_mapping(processed_dir / "age_bucket_to_id.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(
        config,
        vocab_size=len(vocab),
        num_labels=len(target_codes),
        num_modalities=len(modality_to_id),
        num_time_buckets=len(time_bucket_to_id),
        num_age_buckets=len(age_bucket_to_id),
        include_mlm_head=False,
        include_classifier=True,
    ).to(device)

    if pretrained_checkpoint is not None:
        load_pretrained_encoder(model, pretrained_checkpoint, device=device)

    train_labels = pd.read_parquet(processed_dir / "train_labels.parquet")
    pos_weight = compute_pos_weights(
        train_labels, target_codes, clip=train_cfg.get("pos_weight_clip", 20.0)
    ).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("learning_rate", 3e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    use_amp = train_cfg.get("use_amp", True) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    run = None
    if use_wandb and wandb_cfg.get("mode") != "disabled":
        import wandb

        run = wandb.init(
            project=wandb_cfg.get("project", "ehr-timeline-transformer"),
            entity=wandb_cfg.get("entity"),
            name=wandb_cfg.get("run_name"),
            config=config,
        )

    best_map = -1.0
    best_metrics: dict[str, Any] = {}
    patience = train_cfg.get("early_stopping_patience", 5)
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_macro_auroc": [], "val_map": []}

    epochs = train_cfg.get("epochs", 50)
    grad_clip = train_cfg.get("gradient_clip_norm", 1.0)
    monitor = train_cfg.get("monitor_metric", "val_map")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device)
            modality_ids = batch["modality_ids"].to(device)
            time_bucket_ids = batch["time_bucket_ids"].to(device)
            age_bucket_ids = batch["age_bucket_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if use_amp:
                with torch.autocast(device_type="cuda"):
                    logits = model(
                        input_ids,
                        modality_ids,
                        time_bucket_ids,
                        age_bucket_ids,
                        attention_mask,
                        task="cls",
                    )
                    loss = loss_fn(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(
                    input_ids,
                    modality_ids,
                    time_bucket_ids,
                    age_bucket_ids,
                    attention_mask,
                    task="cls",
                )
                loss = loss_fn(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / max(n_batches, 1)
        val_metrics = evaluate_model(
            model, val_loader, device, target_codes, loss_fn=loss_fn, use_amp=use_amp
        )
        val_loss = val_metrics.get("loss", 0.0)
        val_map = val_metrics.get("mAP") or 0.0
        val_auroc = val_metrics.get("macro_auroc") or 0.0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_auroc"].append(val_auroc)
        history["val_map"].append(val_map)

        scheduler.step(val_map)
        log_dict = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_auroc": val_auroc,
            "val_map": val_map,
            "lr": optimizer.param_groups[0]["lr"],
        }
        print(
            f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_auroc={val_auroc:.4f} val_map={val_map:.4f}"
        )
        if run is not None:
            import wandb

            wandb.log(log_dict)

        if val_map > best_map:
            best_map = val_map
            best_metrics = val_metrics
            patience_counter = 0
            ckpt = {
                "model_state_dict": filter_finetune_state_dict(model.state_dict()),
                "config": config,
                "target_codes": target_codes,
                "vocab_path": str(processed_dir / "vocab.json"),
                "lab_binner_path": str(processed_dir / "lab_binner.json"),
                "best_val_metrics": best_metrics,
                "epoch": epoch,
                "pretrained_from": str(pretrained_checkpoint) if pretrained_checkpoint else None,
            }
            torch.save(ckpt, checkpoint_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    with open(metrics_dir / "val_metrics.json", "w") as f:
        json.dump(best_metrics, f, indent=2)

    per_cond = pd.DataFrame(
        {
            "code": target_codes,
            "auroc": [best_metrics.get("per_condition_auroc", {}).get(c) for c in target_codes],
            "ap": [best_metrics.get("per_condition_ap", {}).get(c) for c in target_codes],
        }
    )
    per_cond.to_csv(metrics_dir / "per_condition_metrics.csv", index=False)

    _save_training_curves(history, plots_dir / "training_curves.png")
    _save_per_condition_ap(best_metrics, target_codes, plots_dir / "per_condition_ap.png")

    if run is not None:
        import wandb

        wandb.finish()

    return {"best_val_metrics": best_metrics, "history": history}


def _save_training_curves(history: dict, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history["train_loss"], label="train")
        axes[0].plot(history["val_loss"], label="val")
        axes[0].set_title("Loss")
        axes[0].legend()
        axes[1].plot(history["val_macro_auroc"], label="macro AUROC")
        axes[1].plot(history["val_map"], label="mAP")
        axes[1].set_title("Validation metrics")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"Could not save training curves: {e}")


def _save_per_condition_ap(metrics: dict, target_codes: list[str], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt

        aps = [metrics.get("per_condition_ap", {}).get(c) or 0 for c in target_codes]
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.bar(range(len(aps)), aps)
        ax.set_title("Per-condition AP (validation)")
        ax.set_xlabel("Condition index")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"Could not save per-condition plot: {e}")
