#!/usr/bin/env python3
"""Train the transformer model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.train_transformer import train_transformer
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
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    if args.data_dir is not None:
        config["data"]["data_dir"] = args.data_dir
    resolve_data_dir(config, ROOT, data_dir_override=args.data_dir)
    processed_dir = ROOT / config["data"]["processed_dir"]

    results = train_transformer(
        config=config,
        processed_dir=processed_dir,
        use_wandb=not args.no_wandb,
    )
    best = results["best_val_metrics"]
    print(
        f"\nBest validation: macro_auroc={best.get('macro_auroc')} "
        f"mAP={best.get('mAP')}"
    )


if __name__ == "__main__":
    main()
