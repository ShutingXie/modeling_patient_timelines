"""Train Masked Event Modeling pretraining on official train split only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import load_json_mapping, load_processed_dataset, split_pretrain_patients
from src.data.mlm_utils import MLM_IGNORE_INDEX
from src.data.vocab import Vocabulary
from src.models.transformer_encoder import build_model_from_config
from src.training.evaluate_mlm import evaluate_mlm
from src.utils.io import save_json
from src.utils.seed import set_seed


def train_pretrain(
    config: dict[str, Any],
    processed_dir: str | Path,
    checkpoint_dir: str | Path = "outputs/checkpoints",
    metrics_dir: str | Path = "outputs/metrics",
    plots_dir: str | Path = "outputs/plots",
    use_wandb: bool = True,
) -> dict[str, Any]:
    set_seed(config.get("seed", 42))
    processed_dir = Path(processed_dir)
    checkpoint_dir = Path(checkpoint_dir)
    metrics_dir = Path(metrics_dir)
    plots_dir = Path(plots_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    pretrain_cfg = config.get("pretrain", {})
    wandb_cfg = config.get("wandb", {})

    train_events = pd.read_parquet(processed_dir / "train_events.parquet")
    all_train_ids = train_events["patient_id"].unique().tolist()
    pretrain_train_ids, pretrain_val_ids = split_pretrain_patients(
        all_train_ids,
        val_frac=pretrain_cfg.get("split_frac_val", 0.1),
        seed=pretrain_cfg.get("split_seed", config.get("seed", 42)),
    )
    save_json(
        {
            "pretrain_train_ids": sorted(pretrain_train_ids),
            "pretrain_val_ids": sorted(pretrain_val_ids),
            "split_frac_val": pretrain_cfg.get("split_frac_val", 0.1),
            "split_seed": pretrain_cfg.get("split_seed", config.get("seed", 42)),
        },
        processed_dir / "pretrain_split.json",
    )

    train_ds = load_processed_dataset(
        processed_dir,
        "train",
        config,
        include_labels=False,
        mlm=True,
        patient_ids=pretrain_train_ids,
    )
    val_ds = load_processed_dataset(
        processed_dir,
        "train",
        config,
        include_labels=False,
        mlm=True,
        patient_ids=pretrain_val_ids,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=pretrain_cfg.get("batch_size", 32),
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=pretrain_cfg.get("batch_size", 32),
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
        num_labels=1,
        num_modalities=len(modality_to_id),
        num_time_buckets=len(time_bucket_to_id),
        num_age_buckets=len(age_bucket_to_id),
        include_mlm_head=True,
        include_classifier=False,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=MLM_IGNORE_INDEX)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=pretrain_cfg.get("learning_rate", 3e-4),
        weight_decay=pretrain_cfg.get("weight_decay", 0.01),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    use_amp = pretrain_cfg.get("use_amp", True) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    run = None
    if use_wandb and wandb_cfg.get("mode") != "disabled":
        import wandb

        run = wandb.init(
            project=wandb_cfg.get("project", "ehr-timeline-transformer"),
            entity=wandb_cfg.get("entity"),
            name=wandb_cfg.get("run_name", "mem_pretrain"),
            config=config,
            tags=["pretrain", "mlm"],
        )

    best_val_loss = float("inf")
    best_metrics: dict[str, Any] = {}
    patience = pretrain_cfg.get("early_stopping_patience", 3)
    patience_counter = 0
    history = {
        "train_mlm_loss": [],
        "pretrain_val_mlm_loss": [],
        "pretrain_val_mlm_accuracy": [],
        "pretrain_val_mlm_top5_accuracy": [],
    }

    epochs = pretrain_cfg.get("epochs", 20)
    grad_clip = pretrain_cfg.get("gradient_clip_norm", 1.0)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            mlm_input_ids = batch["mlm_input_ids"].to(device)
            modality_ids = batch["modality_ids"].to(device)
            time_bucket_ids = batch["time_bucket_ids"].to(device)
            age_bucket_ids = batch["age_bucket_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            mlm_labels = batch["mlm_labels"].to(device)

            if use_amp:
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
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
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
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / max(n_batches, 1)
        val_metrics = evaluate_mlm(model, val_loader, device, loss_fn, use_amp=use_amp)
        val_loss = val_metrics["mlm_loss"]
        val_acc = val_metrics["mlm_accuracy"]
        val_top5_acc = val_metrics["mlm_top5_accuracy"]

        history["train_mlm_loss"].append(train_loss)
        history["pretrain_val_mlm_loss"].append(val_loss)
        history["pretrain_val_mlm_accuracy"].append(val_acc)
        history["pretrain_val_mlm_top5_accuracy"].append(val_top5_acc)

        scheduler.step(val_loss)
        log_dict = {
            "epoch": epoch,
            "train_mlm_loss": train_loss,
            "pretrain_val_mlm_loss": val_loss,
            "pretrain_val_mlm_accuracy": val_acc,
            "pretrain_val_mlm_top5_accuracy": val_top5_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        print(
            f"Pretrain epoch {epoch}/{epochs} | train_mlm_loss={train_loss:.4f} "
            f"pretrain_val_mlm_loss={val_loss:.4f} pretrain_val_mlm_acc={val_acc:.4f} "
            f"pretrain_val_mlm_top5_acc={val_top5_acc:.4f}"
        )
        if run is not None:
            import wandb

            wandb.log(log_dict)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = val_metrics
            patience_counter = 0
            ckpt = {
                "model_state_dict": model.state_dict(),
                "config": config,
                "vocab_path": str(processed_dir / "vocab.json"),
                "best_val_metrics": best_metrics,
                "epoch": epoch,
                "pretrain_task": "mlm",
            }
            torch.save(ckpt, checkpoint_dir / "pretrain_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Pretrain early stopping at epoch {epoch}")
                break

    with open(metrics_dir / "pretrain_metrics.json", "w") as f:
        json.dump(best_metrics, f, indent=2)

    _save_pretrain_curves(history, plots_dir / "pretrain_curves.png")

    if run is not None:
        import wandb

        wandb.finish()

    return {"best_val_metrics": best_metrics, "history": history}


def _save_pretrain_curves(history: dict, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history["train_mlm_loss"], label="train")
        axes[0].plot(history["pretrain_val_mlm_loss"], label="pretrain_val")
        axes[0].set_title("MLM Loss")
        axes[0].legend()
        axes[1].plot(history["pretrain_val_mlm_accuracy"], label="top-1")
        axes[1].plot(history["pretrain_val_mlm_top5_accuracy"], label="top-5")
        axes[1].set_title("MLM Accuracy")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"Could not save pretrain curves: {e}")
