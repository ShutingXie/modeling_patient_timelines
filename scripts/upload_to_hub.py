#!/usr/bin/env python3
"""Upload trained model artifacts to Hugging Face Hub."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="HuggingFace repo id, e.g. user/ehr-timeline")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best_model.pt")
    parser.add_argument("--processed-dir", default="outputs/processed")
    args = parser.parse_args()

    import torch
    from huggingface_hub import upload_folder

    hf_dir = ROOT / "hf_model"
    if hf_dir.exists():
        shutil.rmtree(hf_dir)
    hf_dir.mkdir()

    ckpt_path = ROOT / args.checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    torch.save(ckpt["model_state_dict"], hf_dir / "pytorch_model.bin")

    with open(hf_dir / "config.json", "w") as f:
        json.dump(ckpt["config"], f, indent=2)

    processed = ROOT / args.processed_dir
    for name in [
        "vocab.json",
        "lab_binner.json",
        "target_codes.json",
        "modality_to_id.json",
        "time_bucket_to_id.json",
        "age_bucket_to_id.json",
    ]:
        shutil.copy(processed / name, hf_dir / name)

    readme = hf_dir / "README.md"
    readme.write_text(
        """---
language: en
tags:
- medical
- ehr
- transformer
---

# Patient Timeline Transformer

Encoder-only transformer trained **from scratch** on the provided synthetic Synthea EHR data.

- Structured CSV tables only (9 event modalities)
- 40 multi-label target conditions
- No external data or pretrained weights
- **Not for clinical use**

## Usage

Load `pytorch_model.bin` with the accompanying `config.json`, `vocab.json`, and preprocessing artifacts.
"""
    )

    upload_folder(
        repo_id=args.repo_id,
        folder_path=str(hf_dir),
        repo_type="model",
    )
    print(f"Uploaded to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
