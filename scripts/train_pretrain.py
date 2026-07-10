#!/usr/bin/env python3
"""Train MEM (Masked Event Modeling) pretraining."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.train_pretrain import train_pretrain
from src.utils.io import load_config, resolve_data_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/transformer.yaml")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override config data.data_dir (absolute path or relative to repo root)",
    )
    parser.add_argument(
        "--mask-prob",
        type=float,
        default=None,
        help="Override config pretrain.mask_prob (default: use yaml value)",
    )
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=None,
        help="Override config pretrain.epochs (default: use yaml value)",
    )
    args = parser.parse_args()

    if args.mask_prob is not None and not (0.0 < args.mask_prob <= 1.0):
        parser.error("--mask-prob must be in (0, 1]")
    if args.pretrain_epochs is not None and args.pretrain_epochs < 1:
        parser.error("--pretrain-epochs must be >= 1")

    config = load_config(ROOT / args.config)
    if args.data_dir is not None:
        config["data"]["data_dir"] = args.data_dir
    pretrain_cfg = config.setdefault("pretrain", {})
    if args.mask_prob is not None:
        pretrain_cfg["mask_prob"] = args.mask_prob
    if args.pretrain_epochs is not None:
        pretrain_cfg["epochs"] = args.pretrain_epochs
    resolve_data_dir(config, ROOT, data_dir_override=args.data_dir)
    processed_dir = ROOT / config["data"]["processed_dir"]

    results = train_pretrain(
        config=config,
        processed_dir=processed_dir,
        use_wandb=not args.no_wandb,
    )
    best = results["best_val_metrics"]
    print(
        f"\nBest pretrain validation: mlm_loss={best.get('mlm_loss')} "
        f"mlm_accuracy={best.get('mlm_accuracy')} "
        f"mlm_top5_accuracy={best.get('mlm_top5_accuracy')}"
    )


if __name__ == "__main__":
    main()
