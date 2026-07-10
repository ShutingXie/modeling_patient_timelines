"""Baseline models: prevalence and bag-of-events logistic regression."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder

from src.training.metrics import compute_multilabel_metrics


def prevalence_predictions(train_labels: pd.DataFrame, target_codes: list[str]) -> np.ndarray:
    means = train_labels[target_codes].mean().values
    return means


def build_baseline_features(
    events_df: pd.DataFrame,
    anchors_df: pd.DataFrame,
    patients_df: pd.DataFrame,
    patient_ids: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    anchors = anchors_df.set_index("patient_id")["anchor_date"]
    patients = patients_df.set_index("patient_id")
    rows = []
    feature_names: list[str] = []

    for pid in patient_ids:
        row: dict[str, float] = {}
        pat_events = events_df[events_df["patient_id"] == pid]
        counts = pat_events["event_token"].value_counts()
        for token, cnt in counts.items():
            col = f"tok_{token}"
            row[col] = float(cnt)
            if col not in feature_names:
                feature_names.append(col)
        anchor = anchors.get(pid)
        birth = patients.loc[pid, "BIRTHDATE"] if pid in patients.index else pd.NaT
        if pd.notna(anchor) and pd.notna(birth):
            row["age_at_anchor"] = (anchor - birth).days / 365.25
        else:
            row["age_at_anchor"] = 0.0
        gender = patients.loc[pid, "GENDER"] if pid in patients.index else "U"
        row[f"gender_{gender}"] = 1.0
        rows.append({"patient_id": pid, **row})

    if "age_at_anchor" not in feature_names:
        feature_names.append("age_at_anchor")
    gender_cols = sorted({k for r in rows for k in r if k.startswith("gender_")})
    feature_names.extend(gender_cols)

    wide = pd.DataFrame(rows).fillna(0.0)
    for col in feature_names:
        if col not in wide.columns:
            wide[col] = 0.0
    X = wide[["patient_id"] + feature_names]
    return X, feature_names


def train_logistic_baseline(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    feature_names: list[str],
    max_iter: int = 1000,
) -> OneVsRestClassifier:
    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=max_iter, solver="lbfgs", class_weight="balanced")
    )
    clf.fit(X_train[feature_names].values, y_train)
    return clf


def evaluate_baselines(
    train_labels: pd.DataFrame,
    val_labels: pd.DataFrame,
    train_events: pd.DataFrame,
    val_events: pd.DataFrame,
    train_anchors: pd.DataFrame,
    val_anchors: pd.DataFrame,
    train_patients: pd.DataFrame,
    val_patients: pd.DataFrame,
    target_codes: list[str],
    max_iter: int = 1000,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    train_ids = train_labels["patient_id"].tolist()
    val_ids = val_labels["patient_id"].tolist()
    y_train = train_labels[target_codes].values
    y_val = val_labels[target_codes].values

    prev_probs = prevalence_predictions(train_labels, target_codes)
    prev_pred = np.tile(prev_probs, (len(val_ids), 1))
    prev_metrics = compute_multilabel_metrics(y_val, prev_pred, target_codes)

    X_train_df, feature_names = build_baseline_features(
        train_events, train_anchors, train_patients, train_ids
    )
    X_val_df, _ = build_baseline_features(
        val_events, val_anchors, val_patients, val_ids
    )
    for col in feature_names:
        if col not in X_val_df.columns:
            X_val_df[col] = 0.0

    clf = train_logistic_baseline(X_train_df, y_train, feature_names, max_iter=max_iter)
    lr_probs = clf.predict_proba(X_val_df[feature_names].values)
    lr_metrics = compute_multilabel_metrics(y_val, lr_probs, target_codes)

    results = {
        "prevalence": prev_metrics,
        "logistic_regression": lr_metrics,
        "feature_count": len(feature_names),
    }
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
    return results
