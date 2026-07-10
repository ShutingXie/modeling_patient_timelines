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


REQUIRED_DATA_ARTIFACTS = (
    "patient_splits.csv",
    "target_conditions.csv",
    "test_anchors.csv",
    "train_val",
    "test",
)


def resolve_data_dir(
    config: dict[str, Any],
    repo_root: str | Path,
    data_dir_override: str | Path | None = None,
) -> Path:
    """Resolve config data_dir (relative or absolute) and ensure it is populated."""
    raw = data_dir_override if data_dir_override is not None else config["data"]["data_dir"]
    path = Path(raw)
    if not path.is_absolute():
        path = Path(repo_root) / path
    return ensure_data_dir(path, repo_root=repo_root)


def ensure_symlink(link: Path, target: str | Path) -> None:
    """Create or refresh a symlink at ``link`` pointing to ``target``."""
    target = Path(target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() == target:
            return
        link.unlink()
    elif link.exists():
        raise FileExistsError(
            f"Cannot symlink {link}: path exists and is not a symlink. "
            "Remove it manually or point config at the existing directory."
        )
    link.symlink_to(target, target_is_directory=True)


def ensure_data_dir(data_dir: str | Path, repo_root: str | Path | None = None) -> Path:
    """Ensure data_dir exists; symlink from repo root if raw files live there."""
    data_dir = Path(data_dir)
    if data_dir.exists() and (data_dir / "patient_splits.csv").exists():
        return data_dir.resolve()

    root = Path(repo_root or Path(__file__).resolve().parents[2])
    if not all((root / name).exists() for name in REQUIRED_DATA_ARTIFACTS):
        missing = [name for name in REQUIRED_DATA_ARTIFACTS if not (data_dir / name).exists()]
        raise FileNotFoundError(
            f"Expected data in {data_dir} or repo root {root}. "
            f"Missing: {', '.join(missing)}. "
            "Copy or symlink train_val/, test/, and CSV metadata into data/."
        )

    data_dir.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_DATA_ARTIFACTS:
        src = root / name
        dst = data_dir / name
        if not dst.exists():
            dst.symlink_to(src.resolve())
    return data_dir.resolve()
