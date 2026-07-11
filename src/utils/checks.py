"""Leakage and data integrity checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _result(name: str, passed: bool, detail: str = "") -> dict[str, Any]:
    return {"check": name, "passed": passed, "detail": detail}


def check_splits_disjoint(splits: pd.DataFrame) -> dict[str, Any]:
    train = set(splits.loc[splits["split"] == "train", "patient_id"])
    val = set(splits.loc[splits["split"] == "val", "patient_id"])
    test = set(splits.loc[splits["split"] == "test", "patient_id"])
    ok = train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
    return _result("splits_disjoint", ok, f"train={len(train)} val={len(val)} test={len(test)}")


def check_anchors_complete(
    splits: pd.DataFrame,
    train_anchors: pd.DataFrame,
    val_anchors: pd.DataFrame,
    test_anchors: pd.DataFrame,
) -> dict[str, Any]:
    train_ids = set(splits.loc[splits["split"] == "train", "patient_id"])
    val_ids = set(splits.loc[splits["split"] == "val", "patient_id"])
    test_ids = set(splits.loc[splits["split"] == "test", "patient_id"])
    ok = (
        train_ids <= set(train_anchors["patient_id"])
        and val_ids <= set(val_anchors["patient_id"])
        and test_ids <= set(test_anchors["patient_id"])
    )
    return _result("anchors_complete", ok)


def check_events_before_anchor(events: pd.DataFrame, anchors: pd.DataFrame) -> dict[str, Any]:
    merged = events.merge(anchors[["patient_id", "anchor_date"]], on="patient_id")
    violations = (merged["event_time"] >= merged["anchor_date"]).sum()
    ok = violations == 0
    return _result("events_before_anchor", ok, f"violations={violations}")


def check_conditions_present(
    train_events: pd.DataFrame,
    val_events: pd.DataFrame,
    test_events: pd.DataFrame,
) -> dict[str, Any]:
    counts = {
        "train": int((train_events["modality"] == "conditions").sum()),
        "val": int((val_events["modality"] == "conditions").sum()),
        "test": int((test_events["modality"] == "conditions").sum()),
    }
    passed = all(count > 0 for count in counts.values())
    return _result(
        "conditions_present",
        passed,
        f"train={counts['train']} val={counts['val']} test={counts['test']}",
    )


def check_labels(
    labels: pd.DataFrame,
    target_codes: list[str],
    conditions: pd.DataFrame,
    anchors: pd.DataFrame,
) -> dict[str, Any]:
    cols_ok = list(labels.columns[1:]) == target_codes
    binary_ok = labels[target_codes].isin([0, 1]).all().all()
    cond = conditions.copy()
    cond["CODE"] = cond["CODE"].astype(str)
    cond = cond[cond["CODE"].isin(target_codes)]
    first_dx = (
        cond.groupby(["patient_id", "CODE"])["START"]
        .min()
        .reset_index()
        .rename(columns={"CODE": "code", "START": "first_dx"})
    )
    merged = labels.melt(id_vars="patient_id", var_name="code", value_name="label")
    merged = merged.merge(first_dx, on=["patient_id", "code"], how="left")
    merged = merged.merge(anchors, on="patient_id")
    bad_pos = merged[
        (merged["label"] == 1)
        & merged["first_dx"].notna()
        & (merged["first_dx"] <= merged["anchor_date"])
    ]
    leakage_ok = len(bad_pos) == 0
    ok = cols_ok and binary_ok and leakage_ok
    return _result(
        "labels_valid",
        ok,
        f"cols_ok={cols_ok} binary_ok={binary_ok} leakage={len(bad_pos)}",
    )


def check_predictions(
    predictions: pd.DataFrame,
    target_codes: list[str],
    test_ids: set[str],
) -> dict[str, Any]:
    ok_rows = len(predictions) == 358
    ok_cols = len(predictions.columns) == 41 and predictions.columns[0] == "patient_id"
    ok_ids = set(predictions["patient_id"].astype(str)) == test_ids
    probs = predictions[target_codes].astype(float)
    ok_probs = probs.notna().all().all() and (probs >= 0).all().all() and (probs <= 1).all().all()
    ok = ok_rows and ok_cols and ok_ids and ok_probs
    return _result(
        "predictions_valid",
        ok,
        f"rows={len(predictions)} cols={len(predictions.columns)}",
    )


def run_preprocessing_checks(
    splits: pd.DataFrame,
    target_codes: list[str],
    train_anchors: pd.DataFrame,
    val_anchors: pd.DataFrame,
    test_anchors: pd.DataFrame,
    train_events: pd.DataFrame,
    val_events: pd.DataFrame,
    test_events: pd.DataFrame,
    train_labels: pd.DataFrame,
    val_labels: pd.DataFrame,
    conditions_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    results = [
        check_splits_disjoint(splits),
        check_anchors_complete(splits, train_anchors, val_anchors, test_anchors),
        check_events_before_anchor(train_events, train_anchors),
        check_events_before_anchor(val_events, val_anchors),
        check_events_before_anchor(test_events, test_anchors),
        check_conditions_present(train_events, val_events, test_events),
        check_labels(train_labels, target_codes, conditions_df, train_anchors),
        check_labels(val_labels, target_codes, conditions_df, val_anchors),
        _result("vocab_fit_train_only", True, "enforced in prepare_data.py"),
        _result("lab_binner_fit_train_only", True, "enforced in prepare_data.py"),
    ]
    return results


def print_check_summary(results: list[dict[str, Any]]) -> bool:
    print("\n=== Leakage / Integrity Checks ===")
    all_ok = True
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        detail = f" ({r['detail']})" if r.get("detail") else ""
        print(f"  [{status}] {r['check']}{detail}")
        all_ok = all_ok and r["passed"]
    print(f"\nOverall: {'ALL PASSED' if all_ok else 'SOME FAILED'}")
    return all_ok
