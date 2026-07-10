"""Generate test-set predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import EHRTimelineDataset, load_json_mapping
from src.data.lab_binning import LabBinner
from src.data.vocab import Vocabulary
from src.models.transformer_encoder import build_model_from_config
from src.utils.checks import check_predictions


@torch.no_grad()
def predict_test(
    checkpoint_path: str | Path,
    processed_dir: str | Path,
    output_path: str | Path,
    batch_size: int = 32,
) -> pd.DataFrame:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    target_codes = ckpt["target_codes"]
    processed_dir = Path(processed_dir)

    vocab = Vocabulary.load(processed_dir / "vocab.json")
    modality_to_id = load_json_mapping(processed_dir / "modality_to_id.json")
    time_bucket_to_id = load_json_mapping(processed_dir / "time_bucket_to_id.json")
    age_bucket_to_id = load_json_mapping(processed_dir / "age_bucket_to_id.json")

    events = pd.read_parquet(processed_dir / "test_events.parquet")
    anchors = pd.read_parquet(processed_dir / "test_anchors.parquet")
    patients = pd.read_parquet(processed_dir / "test_patients.parquet")
    data_cfg = config.get("data", config)

    dataset = EHRTimelineDataset(
        events_df=events,
        anchors_df=anchors,
        vocab=vocab,
        modality_to_id=modality_to_id,
        time_bucket_to_id=time_bucket_to_id,
        age_bucket_to_id=age_bucket_to_id,
        patients_df=patients,
        max_seq_len=data_cfg.get("max_seq_len", 1024),
        time_bucket_edges=data_cfg.get("time_buckets_days"),
        age_bucket_years=data_cfg.get("age_bucket_years", 5),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(
        config,
        vocab_size=len(vocab),
        num_labels=len(target_codes),
        num_modalities=len(modality_to_id),
        num_time_buckets=len(time_bucket_to_id),
        num_age_buckets=len(age_bucket_to_id),
        include_mlm_head=False,
        include_classifier=True,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    all_probs = []
    all_ids = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        modality_ids = batch["modality_ids"].to(device)
        time_bucket_ids = batch["time_bucket_ids"].to(device)
        age_bucket_ids = batch["age_bucket_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(
            input_ids, modality_ids, time_bucket_ids, age_bucket_ids, attention_mask, task="cls"
        )
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_ids.extend(batch["patient_id"])

    probs_arr = __import__("numpy").concatenate(all_probs, axis=0)
    pred_df = pd.DataFrame(probs_arr, columns=target_codes)
    pred_df.insert(0, "patient_id", all_ids)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_path, index=False)

    from src.data.load_data import load_patient_splits

    data_dir = Path(config["data"]["data_dir"])
    if not data_dir.is_absolute():
        data_dir = Path(__file__).resolve().parents[2] / data_dir
    splits = load_patient_splits(data_dir)
    test_ids = set(splits.loc[splits["split"] == "test", "patient_id"])
    result = check_predictions(pred_df, target_codes, test_ids)
    if not result["passed"]:
        raise ValueError(f"Prediction validation failed: {result['detail']}")

    return pred_df
