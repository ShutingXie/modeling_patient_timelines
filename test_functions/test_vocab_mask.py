import unittest

from src.data.vocab import MASK_TOKEN, Vocabulary


class TestVocabMask(unittest.TestCase):
    def test_mask_token_registered(self):
        vocab = Vocabulary()
        self.assertEqual(vocab.mask_id, 3)
        self.assertIn(MASK_TOKEN, vocab.token_to_id)

    def test_mask_not_in_fit_counts(self):
        vocab = Vocabulary().fit(["event_a", "event_b", "event_a"], min_freq=1)
        self.assertIn(MASK_TOKEN, vocab.token_to_id)
        self.assertEqual(vocab.token_to_id[MASK_TOKEN], 3)

    def test_load_injects_mask_when_missing(self):
        vocab = Vocabulary().fit(["event_a"], min_freq=1)
        data = vocab.to_dict()
        del data["token_to_id"][MASK_TOKEN]
        loaded = Vocabulary.from_dict(data)
        self.assertIn(MASK_TOKEN, loaded.token_to_id)
        self.assertGreaterEqual(loaded.mask_id, 3)


if __name__ == "__main__":
    unittest.main()
