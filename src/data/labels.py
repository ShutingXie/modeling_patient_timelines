"""Multi-label target construction from post-anchor first diagnoses."""

from __future__ import annotations

import pandas as pd


def build_labels(
    conditions_df: pd.DataFrame,
    anchors_df: pd.DataFrame,
    target_codes: list[str],
) -> pd.DataFrame:
    cond = conditions_df.copy()
    cond["CODE"] = cond["CODE"].astype(str)
    cond = cond[cond["CODE"].isin(target_codes)]
    cond["first_dx"] = cond.groupby(["patient_id", "CODE"])["START"].transform("min")

    first_dx = cond.drop_duplicates(["patient_id", "CODE"])[
        ["patient_id", "CODE", "first_dx"]
    ]
    anchors = anchors_df[["patient_id", "anchor_date"]].copy()
    merged = first_dx.merge(anchors, on="patient_id", how="right")
    window_end = merged["anchor_date"] + pd.DateOffset(years=5)
    merged["label"] = (
        (merged["first_dx"] > merged["anchor_date"])
        & (merged["first_dx"] <= window_end)
    ).astype(int)
    merged["label"] = merged["label"].fillna(0).astype(int)

    wide = merged.pivot(index="patient_id", columns="CODE", values="label")
    for code in target_codes:
        if code not in wide.columns:
            wide[code] = 0
    wide = wide[target_codes].fillna(0).astype(int).reset_index()
    return wide
