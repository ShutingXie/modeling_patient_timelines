#!/usr/bin/env python3
"""Compare baseline vs regularized fine-tune validation metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics-dir",
        default="outputs/metrics",
        help="Directory containing val_metrics*.json files",
    )
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    runs = [
        ("baseline", metrics_dir / "val_metrics.json"),
        ("reg_a", metrics_dir / "val_metrics_reg_a.json"),
        ("reg_bc", metrics_dir / "val_metrics_reg.json"),
    ]

    print("Fine-tune regularization comparison")
    print("-" * 72)
    print(f"{'run':<10} {'macro_auroc':>14} {'mAP':>10} {'val_loss':>10} {'status':>12}")
    print("-" * 72)

    best_map = -1.0
    best_run = None
    for name, path in runs:
        metrics = _load_metrics(path)
        if metrics is None:
            print(f"{name:<10} {'n/a':>14} {'n/a':>10} {'n/a':>10} {'missing':>12}")
            continue
        macro_auroc = metrics.get("macro_auroc")
        map_score = metrics.get("mAP")
        val_loss = metrics.get("loss")
        print(
            f"{name:<10} {_fmt(macro_auroc):>14} {_fmt(map_score):>10} "
            f"{_fmt(val_loss):>10} {'ok':>12}"
        )
        if map_score is not None and map_score > best_map:
            best_map = map_score
            best_run = name

    print("-" * 72)
    if best_run is not None:
        print(f"Recommended checkpoint family: {best_run} (highest val mAP)")
        if best_run == "baseline":
            print("Use outputs/checkpoints/best_model.pt")
        elif best_run == "reg_a":
            print("Use outputs/checkpoints/best_model_reg_a.pt")
        else:
            print("Use outputs/checkpoints/best_model_reg.pt")
    else:
        print("No metrics files found. Run fine-tune experiments first.")


if __name__ == "__main__":
    main()
