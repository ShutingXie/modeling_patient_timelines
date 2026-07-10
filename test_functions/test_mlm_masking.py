import unittest

import torch

from src.data.mlm_utils import MLM_IGNORE_INDEX, apply_mlm_mask
from src.data.vocab import Vocabulary


class TestMlmMasking(unittest.TestCase):
    def setUp(self):
        self.vocab = Vocabulary().fit(["a", "b", "c", "d"], min_freq=1)

    def test_skips_cls_and_pad(self):
        input_ids = torch.tensor(
            [self.vocab.cls_id, 4, 5, self.vocab.pad_id, self.vocab.pad_id],
            dtype=torch.long,
        )
        attention_mask = torch.tensor([1, 1, 1, 0, 0], dtype=torch.bool)
        _, labels = apply_mlm_mask(
            input_ids,
            attention_mask,
            self.vocab,
            mask_prob=1.0,
            rng=torch.Generator().manual_seed(0),
        )
        self.assertEqual(labels[0].item(), MLM_IGNORE_INDEX)
        self.assertEqual(labels[3].item(), MLM_IGNORE_INDEX)
        self.assertEqual(labels[4].item(), MLM_IGNORE_INDEX)

    def test_mask_rate_approximate(self):
        seq_len = 200
        input_ids = torch.full((seq_len,), self.vocab.pad_id, dtype=torch.long)
        input_ids[0] = self.vocab.cls_id
        input_ids[1:151] = 4
        attention_mask = torch.zeros(seq_len, dtype=torch.bool)
        attention_mask[:151] = True

        masked_total = 0
        for seed in range(20):
            _, labels = apply_mlm_mask(
                input_ids,
                attention_mask,
                self.vocab,
                mask_prob=0.15,
                rng=torch.Generator().manual_seed(seed),
            )
            masked_total += (labels != MLM_IGNORE_INDEX).sum().item()
        avg_rate = masked_total / (20 * 150)
        self.assertGreater(avg_rate, 0.08)
        self.assertLess(avg_rate, 0.22)

    def test_mlm_labels_match_original_ids(self):
        input_ids = torch.tensor([self.vocab.cls_id, 4, 5, 6], dtype=torch.long)
        attention_mask = torch.ones(4, dtype=torch.bool)
        _, labels = apply_mlm_mask(
            input_ids,
            attention_mask,
            self.vocab,
            mask_prob=1.0,
            rng=torch.Generator().manual_seed(1),
        )
        for i in range(1, 4):
            if labels[i].item() != MLM_IGNORE_INDEX:
                self.assertEqual(labels[i].item(), input_ids[i].item())


if __name__ == "__main__":
    unittest.main()
