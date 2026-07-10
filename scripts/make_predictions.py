#!/usr/bin/env python3
"""Generate test predictions from a trained checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference.predict_test import predict_test
from src.utils.io import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best_model.pt")
    parser.add_argument("--output", default="outputs/predictions.csv")
    parser.add_argument("--config", default=None, help="Optional; read from checkpoint if omitted")
    args = parser.parse_args()

    checkpoint = ROOT / args.checkpoint
    output = ROOT / args.output

    import torch

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    processed_dir = ROOT / ckpt["config"]["data"]["processed_dir"]

    pred_df = predict_test(
        checkpoint_path=checkpoint,
        processed_dir=processed_dir,
        output_path=output,
    )
    print(f"Wrote {len(pred_df)} predictions to {output}")


if __name__ == "__main__":
    main()
