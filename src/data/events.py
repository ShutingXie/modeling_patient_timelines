"""Build timestamped pre-anchor event sequences from structured EHR tables."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.lab_binning import LabBinner, _sanitize_text_value
from src.data.load_data import TABLE_TIME_COLUMNS, load_table

MODALITY_ORDER = [
    "conditions",
    "medications",
    "procedures",
    "observations",
    "encounters",
    "immunizations",
    "careplans",
    "allergies",
    "devices",
]
MODALITY_TO_PREFIX = {
    "conditions": "COND",
    "medications": "MED",
    "procedures": "PROC",
    "observations": "OBS",
    "encounters": "ENC",
    "immunizations": "IMM",
    "careplans": "CARE",
    "allergies": "ALLERGY",
    "devices": "DEVICE",
}


def _unknown_token(modality: str) -> str:
    return f"{MODALITY_TO_PREFIX[modality]}_UNKNOWN"


def _code_str(val: Any) -> str | None:
    if pd.isna(val):
        return None
    return str(val)


def _process_conditions(df: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(anchors, on="patient_id")
    df = df[df["START"] < df["anchor_date"]]
    code = df["CODE"].map(_code_str)
    token = code.where(code.notna(), _unknown_token("conditions"))
    token = "COND_" + token.astype(str)
    return pd.DataFrame(
        {
            "patient_id": df["patient_id"],
            "event_time": df["START"],
            "event_token": token,
            "modality": "conditions",
            "raw_code": code.fillna("UNKNOWN"),
            "value": None,
        }
    )


def _process_medications(df: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(anchors, on="patient_id")
    df = df[df["START"] < df["anchor_date"]]
    code = df["CODE"].map(_code_str)
    token = "MED_" + code.where(code.notna(), "UNKNOWN").astype(str)
    return pd.DataFrame(
        {
            "patient_id": df["patient_id"],
            "event_time": df["START"],
            "event_token": token,
            "modality": "medications",
            "raw_code": code.fillna("UNKNOWN"),
            "value": None,
        }
    )


def _process_procedures(df: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(anchors, on="patient_id")
    df = df[df["DATE"] < df["anchor_date"]]
    code = df["CODE"].map(_code_str)
    token = "PROC_" + code.where(code.notna(), "UNKNOWN").astype(str)
    return pd.DataFrame(
        {
            "patient_id": df["patient_id"],
            "event_time": df["DATE"],
            "event_token": token,
            "modality": "procedures",
            "raw_code": code.fillna("UNKNOWN"),
            "value": None,
        }
    )


def _process_observations(
    df: pd.DataFrame,
    anchors: pd.DataFrame,
    lab_binner: LabBinner | None,
) -> pd.DataFrame:
    df = df.merge(anchors, on="patient_id")
    df = df[df["DATE"] < df["anchor_date"]].copy()
    if df.empty:
        return pd.DataFrame(
            columns=["patient_id", "event_time", "event_token", "modality", "raw_code", "value"]
        )

    code = df["CODE"].map(_code_str).fillna("UNKNOWN").astype(str)
    is_numeric = df["TYPE"] == "numeric"
    suffix = pd.Series("BIN_0", index=df.index, dtype=str)

    if is_numeric.any():
        vals = pd.to_numeric(df.loc[is_numeric, "VALUE"], errors="coerce")
        if lab_binner is not None:
            num_suffix = [
                lab_binner.transform_numeric(c, float(v)) if pd.notna(v) else "BIN_0"
                for c, v in zip(code[is_numeric], vals)
            ]
        else:
            num_suffix = ["BIN_0"] * is_numeric.sum()
        suffix.loc[is_numeric] = num_suffix

    if (~is_numeric).any():
        text_df = df.loc[~is_numeric]
        text_code = code[~is_numeric]
        if lab_binner is not None:
            text_suffix = [
                lab_binner.transform_text(c, v) for c, v in zip(text_code, text_df["VALUE"])
            ]
        else:
            text_suffix = text_df["VALUE"].map(_sanitize_text_value).tolist()
        suffix.loc[~is_numeric] = text_suffix

    token = "OBS_" + code + "_" + suffix
    return pd.DataFrame(
        {
            "patient_id": df["patient_id"].values,
            "event_time": df["DATE"].values,
            "event_token": token.values,
            "modality": "observations",
            "raw_code": code.values,
            "value": df["VALUE"].astype(str).where(df["VALUE"].notna(), None),
        }
    )


def _process_encounters(df: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(anchors, on="patient_id")
    df = df[df["START"] < df["anchor_date"]]
    code = df["CODE"].map(_code_str)
    enc_class = df["ENCOUNTERCLASS"].astype(str)
    use_class = code.isna()
    token = "ENC_" + code.where(~use_class, enc_class).fillna("UNKNOWN").astype(str)
    token = token.where(~use_class, "ENC_CLASS_" + enc_class)
    return pd.DataFrame(
        {
            "patient_id": df["patient_id"],
            "event_time": df["START"],
            "event_token": token,
            "modality": "encounters",
            "raw_code": code.fillna(enc_class).fillna("UNKNOWN"),
            "value": None,
        }
    )


def _process_simple(
    df: pd.DataFrame,
    anchors: pd.DataFrame,
    modality: str,
    time_col: str,
) -> pd.DataFrame:
    prefix = MODALITY_TO_PREFIX[modality]
    df = df.merge(anchors, on="patient_id")
    df = df[df[time_col] < df["anchor_date"]]
    code = df["CODE"].map(_code_str)
    token = prefix + "_" + code.where(code.notna(), "UNKNOWN").astype(str)
    return pd.DataFrame(
        {
            "patient_id": df["patient_id"],
            "event_time": df[time_col],
            "event_token": token,
            "modality": modality,
            "raw_code": code.fillna("UNKNOWN"),
            "value": None,
        }
    )


TABLE_PROCESSORS = {
    "conditions": lambda df, anchors, lb: _process_conditions(df, anchors),
    "medications": lambda df, anchors, lb: _process_medications(df, anchors),
    "procedures": lambda df, anchors, lb: _process_procedures(df, anchors),
    "observations": lambda df, anchors, lb: _process_observations(df, anchors, lb),
    "encounters": lambda df, anchors, lb: _process_encounters(df, anchors),
    "immunizations": lambda df, anchors, lb: _process_simple(
        df, anchors, "immunizations", "DATE"
    ),
    "careplans": lambda df, anchors, lb: _process_simple(df, anchors, "careplans", "START"),
    "allergies": lambda df, anchors, lb: _process_simple(df, anchors, "allergies", "START"),
    "devices": lambda df, anchors, lb: _process_simple(df, anchors, "devices", "START"),
}


EVENT_COLUMNS = ["patient_id", "event_time", "event_token", "modality", "raw_code", "value"]


def _normalize_events_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["patient_id"] = out["patient_id"].astype(str)
    out["event_time"] = pd.to_datetime(out["event_time"], utc=True)
    out["event_token"] = out["event_token"].astype(str)
    out["modality"] = out["modality"].astype(str)
    out["raw_code"] = out["raw_code"].astype(str)
    out["value"] = out["value"].where(out["value"].notna(), None).astype("string")
    return out[EVENT_COLUMNS]


def _append_parquet(df: pd.DataFrame, path: Path, writer: pq.ParquetWriter | None) -> pq.ParquetWriter:
    df = _normalize_events_schema(df)
    table = pa.Table.from_pandas(df, preserve_index=False)
    if writer is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(path, table.schema)

    writer.write_table(table)
    return writer


def build_events_for_split(
    data_dir: str | Path,
    split_dir: str,
    patient_ids: set[str] | list[str],
    anchors_df: pd.DataFrame,
    lab_binner: LabBinner | None = None,
    config: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    config = config or {}
    use_tables = config.get("use_tables", list(MODALITY_ORDER))
    ids = set(patient_ids)
    anchors = anchors_df[anchors_df["patient_id"].isin(ids)][
        ["patient_id", "anchor_date"]
    ].copy()

    writer = None
    out_path = Path(output_path) if output_path else None
    chunks: list[pd.DataFrame] = []

    for table_name in use_tables:
        if table_name not in TABLE_PROCESSORS:
            continue
        raw = load_table(data_dir, split_dir, table_name)
        raw = raw[raw["patient_id"].isin(ids)]
        events = TABLE_PROCESSORS[table_name](raw, anchors, lab_binner)
        if events.empty:
            continue
        if out_path is not None:
            writer = _append_parquet(events, out_path, writer)
        else:
            chunks.append(events)

    if writer is not None:
        writer.close()
        return pd.read_parquet(out_path)

    if not chunks:
        return pd.DataFrame(
            columns=[
                "patient_id",
                "event_time",
                "event_token",
                "modality",
                "raw_code",
                "value",
            ]
        )

    events = pd.concat(chunks, ignore_index=True)
    mod_order = {m: i for i, m in enumerate(MODALITY_ORDER)}
    events["modality_order"] = events["modality"].map(mod_order)
    events = events.sort_values(
        ["patient_id", "event_time", "modality_order", "event_token"]
    )
    return events.drop(columns=["modality_order"])


def finalize_events_parquet(path: str | Path) -> pd.DataFrame:
    """Sort a parquet event file written table-by-table."""
    events = pd.read_parquet(path)
    mod_order = {m: i for i, m in enumerate(MODALITY_ORDER)}
    events["modality_order"] = events["modality"].map(mod_order)
    events = events.sort_values(
        ["patient_id", "event_time", "modality_order", "event_token"]
    )
    events = events.drop(columns=["modality_order"])
    events.to_parquet(path, index=False)
    return events
