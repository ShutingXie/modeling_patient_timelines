"""Load raw CSV tables and metadata."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TABLE_TIME_COLUMNS = {
    "conditions": "START",
    "medications": "START",
    "procedures": "DATE",
    "observations": "DATE",
    "encounters": "START",
    "immunizations": "DATE",
    "careplans": "START",
    "allergies": "START",
    "devices": "START",
}


def _to_patient_id(df: pd.DataFrame) -> pd.DataFrame:
    if "PATIENT" in df.columns:
        df = df.rename(columns={"PATIENT": "patient_id"})
    elif "Id" in df.columns:
        df = df.rename(columns={"Id": "patient_id"})
    df["patient_id"] = df["patient_id"].astype(str)
    return df


def load_patient_splits(data_dir: str | Path) -> pd.DataFrame:
    df = pd.read_csv(Path(data_dir) / "patient_splits.csv")
    df = _to_patient_id(df)
    assert set(df["split"]) == {"train", "val", "test"}
    return df[["patient_id", "split"]]


def load_target_conditions(data_dir: str | Path) -> pd.DataFrame:
    df = pd.read_csv(Path(data_dir) / "target_conditions.csv")
    df["code"] = df["CODE"].astype(str)
    assert len(df) == 40
    return df[["code", "DESCRIPTION"]]


def load_test_anchors(data_dir: str | Path) -> pd.DataFrame:
    df = pd.read_csv(Path(data_dir) / "test_anchors.csv")
    df = _to_patient_id(df)
    df["anchor_date"] = pd.to_datetime(df["anchor_date"], utc=True).dt.normalize()
    return df[["patient_id", "anchor_date"]]


def load_patients(data_dir: str | Path, split_dir: str) -> pd.DataFrame:
    df = pd.read_csv(Path(data_dir) / split_dir / "patients.csv")
    df = _to_patient_id(df)
    df["BIRTHDATE"] = pd.to_datetime(df["BIRTHDATE"], utc=True, errors="coerce")
    return df


def load_table(data_dir: str | Path, split_dir: str, table_name: str) -> pd.DataFrame:
    path = Path(data_dir) / split_dir / f"{table_name}.csv"
    df = pd.read_csv(path, low_memory=False)
    df = _to_patient_id(df)
    time_col = TABLE_TIME_COLUMNS.get(table_name)
    if time_col and time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    if table_name == "encounters":
        df["STOP"] = pd.to_datetime(df["STOP"], utc=True, errors="coerce")
    return df


def validate_splits(splits: pd.DataFrame, test_anchors: pd.DataFrame) -> None:
    train_ids = set(splits.loc[splits["split"] == "train", "patient_id"])
    val_ids = set(splits.loc[splits["split"] == "val", "patient_id"])
    test_ids = set(splits.loc[splits["split"] == "test", "patient_id"])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert len(test_ids) == 358
    assert set(test_anchors["patient_id"]) == test_ids
