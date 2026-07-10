import tempfile
import unittest
from pathlib import Path

import torch

from src.models.checkpoint_utils import load_pretrained_encoder, remap_legacy_state_dict
from src.models.transformer_encoder import build_model_from_config


class TestCheckpointUtils(unittest.TestCase):
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

    def _build_pretrain_model(self):
        return build_model_from_config(
            self.config,
            vocab_size=20,
            num_labels=1,
            num_modalities=5,
            num_time_buckets=5,
            num_age_buckets=5,
            include_mlm_head=True,
            include_classifier=False,
        )

    def _build_finetune_model(self):
        return build_model_from_config(
            self.config,
            vocab_size=20,
            num_labels=5,
            num_modalities=5,
            num_time_buckets=5,
            num_age_buckets=5,
            include_mlm_head=False,
            include_classifier=True,
        )

    def test_load_pretrained_encoder(self):
        pretrain_model = self._build_pretrain_model()
        finetune_model = self._build_finetune_model()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "pretrain.pt"
            torch.save({"model_state_dict": pretrain_model.state_dict()}, ckpt_path)
            loaded_keys = load_pretrained_encoder(finetune_model, ckpt_path)
            self.assertTrue(any(k.startswith("encoder.") for k in loaded_keys))

    def test_classifier_not_loaded_from_pretrain(self):
        pretrain_model = self._build_pretrain_model()
        finetune_model = self._build_finetune_model()
        classifier_before = {
            k: v.clone() for k, v in finetune_model.classifier.state_dict().items()
        }
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "pretrain.pt"
            torch.save({"model_state_dict": pretrain_model.state_dict()}, ckpt_path)
            load_pretrained_encoder(finetune_model, ckpt_path)
        classifier_after = finetune_model.classifier.state_dict()
        for key in classifier_before:
            self.assertTrue(torch.equal(classifier_before[key], classifier_after[key]))

    def test_remap_legacy_state_dict(self):
        legacy = {
            "token_emb.weight": torch.randn(20, 32),
            "encoder.layers.0.self_attn.in_proj_weight": torch.randn(96, 32),
            "classifier.0.weight": torch.randn(32, 32),
        }
        remapped = remap_legacy_state_dict(legacy)
        self.assertIn("encoder.token_emb.weight", remapped)
        self.assertIn("encoder.encoder.layers.0.self_attn.in_proj_weight", remapped)
        self.assertIn("classifier.classifier.0.weight", remapped)


if __name__ == "__main__":
    unittest.main()
