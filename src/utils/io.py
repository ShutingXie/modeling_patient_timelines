"""I/O helpers for config and JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def ensure_data_dir(data_dir: str | Path, repo_root: str | Path | None = None) -> Path:
    """Ensure data_dir exists; symlink from repo root if raw files live there."""
    data_dir = Path(data_dir)
    if data_dir.exists() and (data_dir / "patient_splits.csv").exists():
        return data_dir

    root = Path(repo_root or Path(__file__).resolve().parents[2])
    required = [
        "patient_splits.csv",
        "target_conditions.csv",
        "test_anchors.csv",
        "train_val",
        "test",
    ]
    if not all((root / name).exists() for name in required):
        raise FileNotFoundError(
            f"Expected data in {data_dir} or repo root {root}. "
            "Copy or symlink train_val/, test/, and CSV metadata into data/."
        )

    data_dir.mkdir(parents=True, exist_ok=True)
    for name in required:
        src = root / name
        dst = data_dir / name
        if not dst.exists():
            dst.symlink_to(src.resolve())
    return data_dir
