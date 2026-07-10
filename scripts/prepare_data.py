#!/usr/bin/env python3
"""Prepare processed datasets: anchors, labels, events, vocab, and leakage checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.anchors import build_train_val_anchors, load_test_anchors_df
from src.data.dataset import (
    build_age_bucket_map,
    build_modality_to_id,
    build_time_bucket_map,
)
from src.data.events import build_events_for_split, finalize_events_parquet
from src.data.lab_binning import LabBinner
from src.data.labels import build_labels
from src.data.load_data import (
    load_patient_splits,
    load_patients,
    load_table,
    load_target_conditions,
    load_test_anchors,
    validate_splits,
)
from src.data.vocab import Vocabulary
from src.utils.checks import print_check_summary, run_preprocessing_checks
from src.utils.io import ensure_data_dir, load_config, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/transformer.yaml")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    data_cfg = config["data"]
    data_dir = ensure_data_dir(ROOT / data_cfg["data_dir"], repo_root=ROOT)
    processed_dir = ROOT / data_cfg["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("Loading metadata...")
    splits = load_patient_splits(data_dir)
    targets = load_target_conditions(data_dir)
    target_codes = targets["code"].tolist()
    test_anchors_raw = load_test_anchors(data_dir)
    validate_splits(splits, test_anchors_raw)

    train_ids = set(splits.loc[splits["split"] == "train", "patient_id"])
    val_ids = set(splits.loc[splits["split"] == "val", "patient_id"])
    test_ids = set(splits.loc[splits["split"] == "test", "patient_id"])

    print("Building anchors...")
    encounters = load_table(data_dir, "train_val", "encounters")
    train_val_anchors = build_train_val_anchors(encounters, train_ids | val_ids)
    train_anchors = train_val_anchors[train_val_anchors["patient_id"].isin(train_ids)]
    val_anchors = train_val_anchors[train_val_anchors["patient_id"].isin(val_ids)]
    test_anchors = load_test_anchors_df(test_anchors_raw)

    train_anchors.to_parquet(processed_dir / "train_anchors.parquet", index=False)
    val_anchors.to_parquet(processed_dir / "val_anchors.parquet", index=False)
    test_anchors.to_parquet(processed_dir / "test_anchors.parquet", index=False)

    print("Building labels...")
    conditions = load_table(data_dir, "train_val", "conditions")
    train_labels = build_labels(conditions, train_anchors, target_codes)
    val_labels = build_labels(conditions, val_anchors, target_codes)
    train_labels.to_parquet(processed_dir / "train_labels.parquet", index=False)
    val_labels.to_parquet(processed_dir / "val_labels.parquet", index=False)
    save_json(target_codes, processed_dir / "target_codes.json")

    print("Fitting lab binner on train pre-anchor observations...")
    obs = load_table(data_dir, "train_val", "observations")
    obs = obs.merge(train_anchors[["patient_id", "anchor_date"]], on="patient_id")
    obs_train = obs[(obs["patient_id"].isin(train_ids)) & (obs["DATE"] < obs["anchor_date"])]
    lab_binner = LabBinner(num_bins=data_cfg.get("lab_num_bins", 5)).fit(obs_train)
    lab_binner.save(processed_dir / "lab_binner.json")

    event_config = {
        "use_tables": data_cfg.get("use_tables"),
    }

    print("Building event tables...")
    for split_name, ids, split_dir, anchor_df, out_name in [
        ("train", train_ids, "train_val", train_anchors, "train_events.parquet"),
        ("val", val_ids, "train_val", val_anchors, "val_events.parquet"),
        ("test", test_ids, "test", test_anchors, "test_events.parquet"),
    ]:
        out_path = processed_dir / out_name
        if out_path.exists():
            out_path.unlink()
        build_events_for_split(
            data_dir=data_dir,
            split_dir=split_dir,
            patient_ids=ids,
            anchors_df=anchor_df,
            lab_binner=lab_binner,
            config=event_config,
            output_path=out_path,
        )
        finalize_events_parquet(out_path)
        print(f"  {split_name}: {len(pd.read_parquet(out_path)):,} events")

    train_events = pd.read_parquet(processed_dir / "train_events.parquet")

    print("Fitting vocabulary on train events...")
    vocab = Vocabulary().fit(
        train_events["event_token"],
        min_freq=data_cfg.get("min_token_freq", 2),
    )
    vocab.save(processed_dir / "vocab.json")

    modality_to_id = build_modality_to_id()
    save_json(modality_to_id, processed_dir / "modality_to_id.json")

    time_bucket_to_id = build_time_bucket_map(data_cfg.get("time_buckets_days", []))
    save_json(time_bucket_to_id, processed_dir / "time_bucket_to_id.json")

    print("Building age bucket map...")
    train_patients = load_patients(data_dir, "train_val")
    train_patients = train_patients[train_patients["patient_id"].isin(train_ids)]
    age_bucket_to_id = build_age_bucket_map(
        train_events, train_patients, data_cfg.get("age_bucket_years", 5)
    )
    save_json(age_bucket_to_id, processed_dir / "age_bucket_to_id.json")

    # Save patient subsets for dataset loading
    val_patients = load_patients(data_dir, "train_val")
    val_patients = val_patients[val_patients["patient_id"].isin(val_ids)]
    test_patients = load_patients(data_dir, "test")
    test_patients = test_patients[test_patients["patient_id"].isin(test_ids)]
    train_patients.to_parquet(processed_dir / "train_patients.parquet", index=False)
    val_patients.to_parquet(processed_dir / "val_patients.parquet", index=False)
    test_patients.to_parquet(processed_dir / "test_patients.parquet", index=False)

    val_events = pd.read_parquet(processed_dir / "val_events.parquet")
    test_events = pd.read_parquet(processed_dir / "test_events.parquet")

    print("Running leakage checks...")
    results = run_preprocessing_checks(
        splits=splits,
        target_codes=target_codes,
        train_anchors=train_anchors,
        val_anchors=val_anchors,
        test_anchors=test_anchors,
        train_events=train_events,
        val_events=val_events,
        test_events=test_events,
        train_labels=train_labels,
        val_labels=val_labels,
        conditions_df=conditions,
    )
    all_ok = print_check_summary(results)
    if not all_ok:
        raise SystemExit("Preprocessing checks failed.")

    print(f"\nDone. Processed data saved to {processed_dir}")


if __name__ == "__main__":
    main()
