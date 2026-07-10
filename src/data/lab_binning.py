"""Quantile binning for numeric labs and frequency caps for text observations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _sanitize_text_value(value: str, max_len: int = 40) -> str:
    import re

    s = str(value).lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len] if s else "other"


class LabBinner:
    def __init__(self, num_bins: int = 5, top_k_text: int = 50):
        self.num_bins = num_bins
        self.top_k_text = top_k_text
        self.numeric_edges: dict[str, list[float]] = {}
        self.text_top_values: dict[str, list[str]] = {}

    def fit(self, observations_df: pd.DataFrame) -> "LabBinner":
        obs = observations_df.copy()
        obs["CODE"] = obs["CODE"].astype(str)
        numeric = obs[obs["TYPE"] == "numeric"].dropna(subset=["VALUE"])
        numeric["VALUE"] = pd.to_numeric(numeric["VALUE"], errors="coerce")
        numeric = numeric.dropna(subset=["VALUE"])

        for code, grp in numeric.groupby("CODE"):
            values = grp["VALUE"].values
            n_unique = len(np.unique(values))
            n_bins = min(self.num_bins, max(1, n_unique))
            if n_bins == 1:
                self.numeric_edges[code] = [float(values.min()), float(values.max())]
            else:
                quantiles = np.linspace(0, 1, n_bins + 1)
                edges = np.unique(np.quantile(values, quantiles))
                if len(edges) < 2:
                    edges = np.array([values.min(), values.max()])
                self.numeric_edges[code] = edges.tolist()

        text = obs[obs["TYPE"] != "numeric"].dropna(subset=["VALUE"])
        for code, grp in text.groupby("CODE"):
            sanitized = grp["VALUE"].map(_sanitize_text_value)
            top = sanitized.value_counts().head(self.top_k_text).index.tolist()
            self.text_top_values[code] = top
        return self

    def transform_numeric(self, code: str, value: float) -> str:
        code = str(code)
        if code not in self.numeric_edges:
            return "BIN_0"
        edges = self.numeric_edges[code]
        if len(edges) <= 1:
            return "BIN_0"
        idx = int(np.digitize(value, edges[1:-1], right=True))
        idx = min(idx, len(edges) - 2)
        return f"BIN_{idx}"

    def transform_text(self, code: str, value: str) -> str:
        code = str(code)
        sanitized = _sanitize_text_value(value)
        allowed = self.text_top_values.get(code, [])
        if sanitized in allowed:
            return sanitized
        return "OTHER"

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_bins": self.num_bins,
            "top_k_text": self.top_k_text,
            "numeric_edges": self.numeric_edges,
            "text_top_values": self.text_top_values,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LabBinner":
        b = cls(num_bins=d.get("num_bins", 5), top_k_text=d.get("top_k_text", 50))
        b.numeric_edges = d.get("numeric_edges", {})
        b.text_top_values = d.get("text_top_values", {})
        return b

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "LabBinner":
        with open(path) as f:
            return cls.from_dict(json.load(f))
