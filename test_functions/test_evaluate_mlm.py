import unittest
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.mlm_utils import MLM_IGNORE_INDEX
from src.training.evaluate_mlm import evaluate_mlm


class _FixedLogitsModel(nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.logits = logits

    def forward(self, mlm_input_ids: torch.Tensor, *args: Any, task: str = "mlm", **kwargs: Any) -> torch.Tensor:
        batch_size = mlm_input_ids.size(0)
        return self.logits.unsqueeze(0).expand(batch_size, -1, -1)


class TestEvaluateMlm(unittest.TestCase):
    def _run_eval(self, logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
        batch = {
            "mlm_input_ids": torch.zeros(labels.size(0), dtype=torch.long),
            "modality_ids": torch.zeros(labels.size(0), dtype=torch.long),
            "time_bucket_ids": torch.zeros(labels.size(0), dtype=torch.long),
            "age_bucket_ids": torch.zeros(labels.size(0), dtype=torch.long),
            "attention_mask": torch.ones(labels.size(0), dtype=torch.long),
            "mlm_labels": labels,
        }
        loader = DataLoader([batch], batch_size=1)
        model = _FixedLogitsModel(logits)
        loss_fn = nn.CrossEntropyLoss(ignore_index=MLM_IGNORE_INDEX)
        return evaluate_mlm(model, loader, torch.device("cpu"), loss_fn, use_amp=False)

    def test_top1_and_top5_perfect(self):
        logits = torch.tensor(
            [
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 10.0],
            ]
        )
        labels = torch.tensor([MLM_IGNORE_INDEX, 1, 2])
        metrics = self._run_eval(logits, labels)
        self.assertEqual(metrics["mlm_accuracy"], 1.0)
        self.assertEqual(metrics["mlm_top5_accuracy"], 1.0)
        self.assertEqual(metrics["n_masked_tokens"], 2)

    def test_top5_when_label_in_top5_not_top1(self):
        logits = torch.tensor(
            [
                [0.0, 3.0, 2.0, 1.0, 0.0, 0.0],
                [5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        labels = torch.tensor([4, MLM_IGNORE_INDEX])
        metrics = self._run_eval(logits, labels)
        self.assertEqual(metrics["mlm_accuracy"], 0.0)
        self.assertEqual(metrics["mlm_top5_accuracy"], 1.0)
        self.assertEqual(metrics["n_masked_tokens"], 1)

    def test_top5_miss(self):
        logits = torch.tensor([[10.0, 9.0, 8.0, 7.0, 6.0, 0.0]])
        labels = torch.tensor([5])
        metrics = self._run_eval(logits, labels)
        self.assertEqual(metrics["mlm_accuracy"], 0.0)
        self.assertEqual(metrics["mlm_top5_accuracy"], 0.0)

    def test_ignore_index_excluded(self):
        logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
        labels = torch.tensor([MLM_IGNORE_INDEX, 1])
        metrics = self._run_eval(logits, labels)
        self.assertEqual(metrics["n_masked_tokens"], 1)
        self.assertEqual(metrics["mlm_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
