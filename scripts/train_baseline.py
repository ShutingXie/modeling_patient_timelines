#!/usr/bin/env python3
"""Train and evaluate baseline models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.models.baselines import evaluate_baselines
from src.utils.io import ensure_data_dir, load_config, load_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    data_cfg = config["data"]
    ensure_data_dir(ROOT / data_cfg["data_dir"], repo_root=ROOT)
    processed_dir = ROOT / data_cfg["processed_dir"]
    metrics_dir = ROOT / "outputs/metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    target_codes = load_json(processed_dir / "target_codes.json")
    train_labels = pd.read_parquet(processed_dir / "train_labels.parquet")
    val_labels = pd.read_parquet(processed_dir / "val_labels.parquet")
    train_events = pd.read_parquet(processed_dir / "train_events.parquet")
    val_events = pd.read_parquet(processed_dir / "val_events.parquet")
    train_anchors = pd.read_parquet(processed_dir / "train_anchors.parquet")
    val_anchors = pd.read_parquet(processed_dir / "val_anchors.parquet")
    train_patients = pd.read_parquet(processed_dir / "train_patients.parquet")
    val_patients = pd.read_parquet(processed_dir / "val_patients.parquet")

    train_cfg = config.get("training", config)
    results = evaluate_baselines(
        train_labels=train_labels,
        val_labels=val_labels,
        train_events=train_events,
        val_events=val_events,
        train_anchors=train_anchors,
        val_anchors=val_anchors,
        train_patients=train_patients,
        val_patients=val_patients,
        target_codes=target_codes,
        max_iter=train_cfg.get("max_iter", 1000),
        output_path=metrics_dir / "baseline_metrics.json",
    )

    print("Baseline results (validation):")
    for name, metrics in results.items():
        if isinstance(metrics, dict) and "macro_auroc" in metrics:
            print(
                f"  {name}: macro_auroc={metrics.get('macro_auroc')} "
                f"mAP={metrics.get('mAP')}"
            )


if __name__ == "__main__":
    main()
