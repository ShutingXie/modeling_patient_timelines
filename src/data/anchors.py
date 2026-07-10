"""Anchor date computation for train/val and test patients."""

from __future__ import annotations

import pandas as pd


def build_train_val_anchors(
    encounters_df: pd.DataFrame,
    train_val_ids: set[str] | list[str],
) -> pd.DataFrame:
    ids = set(train_val_ids)
    enc = encounters_df[encounters_df["patient_id"].isin(ids)].copy()
    enc["END_DT"] = enc["STOP"].fillna(enc["START"])
    last = enc.groupby("patient_id", as_index=False)["END_DT"].max()
    last = last.rename(columns={"END_DT": "last_encounter_date"})
    last["anchor_date"] = last["last_encounter_date"] - pd.DateOffset(years=5)
    assert len(last) == len(ids), "Missing anchors for some train/val patients"
    assert (last["anchor_date"] < last["last_encounter_date"]).all()
    return last[["patient_id", "last_encounter_date", "anchor_date"]]


def load_test_anchors_df(test_anchors: pd.DataFrame) -> pd.DataFrame:
    df = test_anchors.copy()
    df["anchor_date"] = pd.to_datetime(df["anchor_date"], utc=True).dt.normalize()
    return df[["patient_id", "anchor_date"]]
