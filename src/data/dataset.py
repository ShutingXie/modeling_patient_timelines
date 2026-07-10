"""PyTorch dataset for patient timeline sequences."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.mlm_utils import apply_mlm_mask
from src.data.vocab import CLS_TOKEN, PAD_TOKEN, Vocabulary


def build_time_bucket_map(bucket_edges_days: list[int]) -> dict[str, int]:
    labels = [f"TB_{i}" for i in range(len(bucket_edges_days) + 1)]
    return {label: i for i, label in enumerate(labels)}


def days_to_time_bucket(days: float, bucket_edges_days: list[int]) -> int:
    if days < 0:
        return 0
    for i, edge in enumerate(bucket_edges_days):
        if days <= edge:
            return i
    return len(bucket_edges_days)


def age_to_bucket(
    event_time: pd.Timestamp,
    birthdate: pd.Timestamp | None,
    bucket_years: int = 5,
) -> str:
    if birthdate is None or pd.isna(birthdate):
        return "AGE_UNKNOWN"
    age_years = (event_time - birthdate).days / 365.25
    if age_years < 0:
        return "AGE_UNKNOWN"
    bucket = int(age_years // bucket_years) * bucket_years
    return f"AGE_{bucket}"


def build_age_bucket_map(events: pd.DataFrame, patients: pd.DataFrame, bucket_years: int) -> dict[str, int]:
    birth = patients.set_index("patient_id")["BIRTHDATE"].to_dict()
    buckets = {"AGE_UNKNOWN": 0}
    for _, row in events.iterrows():
        b = age_to_bucket(row["event_time"], birth.get(row["patient_id"]), bucket_years)
        if b not in buckets:
            buckets[b] = len(buckets)
    return buckets


def split_pretrain_patients(
    patient_ids: list[str],
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[set[str], set[str]]:
    """Split official train patients into pretrain_train / pretrain_val."""
    ids = sorted(set(patient_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_val = max(1, int(round(val_frac * len(ids))))
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])
    return train_ids, val_ids


class EHRTimelineDataset(Dataset):
    def __init__(
        self,
        events_df: pd.DataFrame,
        anchors_df: pd.DataFrame,
        vocab: Vocabulary,
        modality_to_id: dict[str, int],
        time_bucket_to_id: dict[str, int],
        age_bucket_to_id: dict[str, int],
        patients_df: pd.DataFrame,
        target_codes: list[str] | None = None,
        labels_df: pd.DataFrame | None = None,
        max_seq_len: int = 1024,
        time_bucket_edges: list[int] | None = None,
        age_bucket_years: int = 5,
        mlm: bool = False,
        mask_prob: float = 0.15,
        mask_token_prob: float = 0.8,
        random_token_prob: float = 0.1,
    ):
        self.vocab = vocab
        self.modality_to_id = modality_to_id
        self.time_bucket_to_id = time_bucket_to_id
        self.age_bucket_to_id = age_bucket_to_id
        self.max_seq_len = max_seq_len
        self.time_bucket_edges = time_bucket_edges or [30, 90, 180, 365, 730, 1825, 3650]
        self.age_bucket_years = age_bucket_years
        self.target_codes = target_codes or []
        self.birthdates = patients_df.set_index("patient_id")["BIRTHDATE"].to_dict()
        self.mlm = mlm
        self.mask_prob = mask_prob
        self.mask_token_prob = mask_token_prob
        self.random_token_prob = random_token_prob

        anchors = anchors_df.set_index("patient_id")["anchor_date"].to_dict()
        labels_map: dict[str, np.ndarray] = {}
        if labels_df is not None and target_codes:
            for _, row in labels_df.iterrows():
                labels_map[row["patient_id"]] = row[target_codes].values.astype(np.float32)

        self.samples: list[dict[str, Any]] = []
        grouped = events_df.groupby("patient_id")
        for patient_id, grp in grouped:
            grp = grp.sort_values("event_time")
            anchor = anchors[patient_id]
            birth = self.birthdates.get(patient_id)
            tokens = grp["event_token"].tolist()
            modalities = grp["modality"].tolist()
            times = grp["event_time"].tolist()

            max_events = max_seq_len - 1
            if len(tokens) > max_events:
                tokens = tokens[-max_events:]
                modalities = modalities[-max_events:]
                times = times[-max_events:]

            input_ids = [vocab.token_id(CLS_TOKEN)] + vocab.encode(tokens)
            modality_ids = [modality_to_id.get("cls", 0)] + [
                modality_to_id.get(m, modality_to_id.get("unknown", 0)) for m in modalities
            ]
            time_bucket_ids = [time_bucket_to_id.get("TB_0", 0)]
            age_bucket_ids = [age_bucket_to_id.get("AGE_UNKNOWN", 0)]
            for t in times:
                days = (anchor - t).total_seconds() / 86400.0
                tb = days_to_time_bucket(days, self.time_bucket_edges)
                tb_key = f"TB_{tb}"
                time_bucket_ids.append(time_bucket_to_id.get(tb_key, tb))
                age_key = age_to_bucket(t, birth, self.age_bucket_years)
                age_bucket_ids.append(age_bucket_to_id.get(age_key, 0))

            seq_len = len(input_ids)
            pad_len = max_seq_len - seq_len
            attention_mask = [1] * seq_len + [0] * pad_len
            input_ids = input_ids + [vocab.pad_id] * pad_len
            modality_ids = modality_ids + [modality_to_id.get("pad", 0)] * pad_len
            time_bucket_ids = time_bucket_ids + [time_bucket_to_id.get("TB_0", 0)] * pad_len
            age_bucket_ids = age_bucket_ids + [age_bucket_to_id.get("AGE_UNKNOWN", 0)] * pad_len

            sample: dict[str, Any] = {
                "patient_id": patient_id,
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "modality_ids": torch.tensor(modality_ids, dtype=torch.long),
                "time_bucket_ids": torch.tensor(time_bucket_ids, dtype=torch.long),
                "age_bucket_ids": torch.tensor(age_bucket_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            }
            if target_codes and labels_df is not None:
                sample["labels"] = torch.tensor(
                    labels_map.get(patient_id, np.zeros(len(target_codes), dtype=np.float32)),
                    dtype=torch.float32,
                )
            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        if not self.mlm:
            return sample

        out = dict(sample)
        mlm_input_ids, mlm_labels = apply_mlm_mask(
            sample["input_ids"].clone(),
            sample["attention_mask"],
            self.vocab,
            mask_prob=self.mask_prob,
            mask_token_prob=self.mask_token_prob,
            random_token_prob=self.random_token_prob,
        )
        out["mlm_input_ids"] = mlm_input_ids
        out["mlm_labels"] = mlm_labels
        return out


def build_modality_to_id() -> dict[str, int]:
    mapping = {"pad": 0, "cls": 1, "unknown": 2}
    modalities = [
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
    for i, m in enumerate(modalities, start=3):
        mapping[m] = i
    return mapping


def load_processed_dataset(
    processed_dir: str | Path,
    split: str,
    config: dict[str, Any],
    include_labels: bool = True,
    mlm: bool = False,
    patient_ids: set[str] | None = None,
) -> EHRTimelineDataset:
    processed_dir = Path(processed_dir)
    events = pd.read_parquet(processed_dir / f"{split}_events.parquet")
    anchors = pd.read_parquet(processed_dir / f"{split}_anchors.parquet")
    patients = pd.read_parquet(processed_dir / f"{split}_patients.parquet")
    if patient_ids is not None:
        events = events[events["patient_id"].isin(patient_ids)]
        anchors = anchors[anchors["patient_id"].isin(patient_ids)]
        patients = patients[patients["patient_id"].isin(patient_ids)]

    vocab = Vocabulary.load(processed_dir / "vocab.json")
    modality_to_id = load_json_mapping(processed_dir / "modality_to_id.json")
    time_bucket_to_id = load_json_mapping(processed_dir / "time_bucket_to_id.json")
    age_bucket_to_id = load_json_mapping(processed_dir / "age_bucket_to_id.json")
    target_codes = None
    labels_df = None
    if include_labels and split in ("train", "val"):
        target_codes = load_json_mapping(processed_dir / "target_codes.json")
        labels_df = pd.read_parquet(processed_dir / f"{split}_labels.parquet")
        if patient_ids is not None:
            labels_df = labels_df[labels_df["patient_id"].isin(patient_ids)]

    data_cfg = config.get("data", config)
    pretrain_cfg = config.get("pretrain", {})
    return EHRTimelineDataset(
        events_df=events,
        anchors_df=anchors,
        vocab=vocab,
        modality_to_id=modality_to_id,
        time_bucket_to_id=time_bucket_to_id,
        age_bucket_to_id=age_bucket_to_id,
        patients_df=patients,
        target_codes=target_codes,
        labels_df=labels_df,
        max_seq_len=data_cfg.get("max_seq_len", 1024),
        time_bucket_edges=data_cfg.get("time_buckets_days"),
        age_bucket_years=data_cfg.get("age_bucket_years", 5),
        mlm=mlm,
        mask_prob=pretrain_cfg.get("mask_prob", 0.15),
        mask_token_prob=pretrain_cfg.get("mask_token_prob", 0.8),
        random_token_prob=pretrain_cfg.get("random_token_prob", 0.1),
    )


def load_json_mapping(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)
