"""Tests for streaming event parquet writes."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.events import _append_parquet


def _sample_events(
    event_token: str,
    modality: str,
    raw_code: str,
    event_time: str = "2020-01-01",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": ["patient-1"],
            "event_time": [pd.Timestamp(event_time, tz="UTC")],
            "event_token": [event_token],
            "modality": [modality],
            "raw_code": [raw_code],
            "value": [None],
        }
    )


class TestEventParquetWriting(unittest.TestCase):
    def test_append_parquet_writes_first_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "events.parquet"
            first_batch = _sample_events("COND_123", "conditions", "123")

            writer = _append_parquet(first_batch, output_path, writer=None)
            writer.close()

            saved = pd.read_parquet(output_path)

            self.assertEqual(len(saved), 1)
            self.assertEqual(saved.iloc[0]["event_token"], "COND_123")
            self.assertEqual(saved.iloc[0]["modality"], "conditions")

    def test_append_parquet_writes_all_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "events.parquet"
            conditions = _sample_events("COND_123", "conditions", "123")
            medications = _sample_events(
                "MED_456",
                "medications",
                "456",
                event_time="2020-02-01",
            )

            writer = _append_parquet(conditions, output_path, writer=None)
            writer = _append_parquet(medications, output_path, writer=writer)
            writer.close()

            saved = pd.read_parquet(output_path)

            self.assertEqual(len(saved), 2)
            self.assertEqual(set(saved["event_token"]), {"COND_123", "MED_456"})


if __name__ == "__main__":
    unittest.main()
