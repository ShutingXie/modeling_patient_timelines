import unittest

import torch

from src.models.transformer_encoder import build_model_from_config


class TestModelForward(unittest.TestCase):
    def setUp(self):
        self.config = {
            "model": {
                "d_model": 32,
                "num_layers": 1,
                "num_heads": 4,
                "dim_feedforward": 64,
                "dropout": 0.0,
            },
            "data": {"max_seq_len": 16},
        }
        self.batch_size = 2
        self.seq_len = 8
        self.vocab_size = 20
        self.num_labels = 5

    def _batch(self):
        return {
            "input_ids": torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len)),
            "modality_ids": torch.zeros(self.batch_size, self.seq_len, dtype=torch.long),
            "time_bucket_ids": torch.zeros(self.batch_size, self.seq_len, dtype=torch.long),
            "age_bucket_ids": torch.zeros(self.batch_size, self.seq_len, dtype=torch.long),
            "attention_mask": torch.ones(self.batch_size, self.seq_len, dtype=torch.bool),
        }

    def test_mlm_head_output_shape(self):
        model = build_model_from_config(
            self.config,
            vocab_size=self.vocab_size,
            num_labels=self.num_labels,
            num_modalities=5,
            num_time_buckets=5,
            num_age_buckets=5,
            include_mlm_head=True,
            include_classifier=False,
        )
        batch = self._batch()
        out = model(
            batch["input_ids"],
            batch["modality_ids"],
            batch["time_bucket_ids"],
            batch["age_bucket_ids"],
            batch["attention_mask"],
            task="mlm",
        )
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.vocab_size))

    def test_cls_head_output_shape(self):
        model = build_model_from_config(
            self.config,
            vocab_size=self.vocab_size,
            num_labels=self.num_labels,
            num_modalities=5,
            num_time_buckets=5,
            num_age_buckets=5,
            include_mlm_head=False,
            include_classifier=True,
        )
        batch = self._batch()
        out = model(
            batch["input_ids"],
            batch["modality_ids"],
            batch["time_bucket_ids"],
            batch["age_bucket_ids"],
            batch["attention_mask"],
            task="cls",
        )
        self.assertEqual(out.shape, (self.batch_size, self.num_labels))

    def test_mlm_loss_computes(self):
        model = build_model_from_config(
            self.config,
            vocab_size=self.vocab_size,
            num_labels=self.num_labels,
            num_modalities=5,
            num_time_buckets=5,
            num_age_buckets=5,
            include_mlm_head=True,
            include_classifier=False,
        )
        batch = self._batch()
        logits = model(
            batch["input_ids"],
            batch["modality_ids"],
            batch["time_bucket_ids"],
            batch["age_bucket_ids"],
            batch["attention_mask"],
            task="mlm",
        )
        labels = torch.full((self.batch_size, self.seq_len), -100, dtype=torch.long)
        labels[:, 1:3] = batch["input_ids"][:, 1:3]
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, self.vocab_size), labels.view(-1), ignore_index=-100
        )
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
