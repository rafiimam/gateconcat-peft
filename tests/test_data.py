"""Dataset tests with a fake tokenizer (no model download).

Skipped when torch/transformers are not installed (light environments);
runs in full in CI.
"""
import unittest

try:
    import torch
    from gateconcat.data import DocumentDatasetWithNoise
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class FakeEncoding(dict):
    def __init__(self, tensors, word_ids):
        super().__init__(tensors)
        self._word_ids = word_ids

    def word_ids(self, batch_index=0):
        return self._word_ids


class FakeTokenizer:
    """Counts calls — regression guard for the single-tokenization fix."""

    def __init__(self, max_len=8):
        self.max_len = max_len
        self.calls = 0

    def __call__(self, words, boxes=None, word_labels=None, **kwargs):
        self.calls += 1
        n = self.max_len
        word_ids = list(range(len(words))) + [None] * (n - len(words))
        tensors = {
            "input_ids": torch.zeros(1, n, dtype=torch.long),
            "attention_mask": torch.ones(1, n, dtype=torch.long),
            "bbox": torch.zeros(1, n, 4, dtype=torch.long),
            "labels": torch.zeros(1, n, dtype=torch.long),
            "offset_mapping": torch.zeros(1, n, 2, dtype=torch.long),
        }
        return FakeEncoding(tensors, word_ids)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestDocumentDataset(unittest.TestCase):
    def setUp(self):
        self.tok = FakeTokenizer(max_len=8)
        data = [{"tokens": ["total", "100"], "bboxes": [[0, 0, 5, 5], [6, 6, 9, 9]],
                 "ner_tags": [1, 2]}]
        self.ds = DocumentDatasetWithNoise(data, self.tok, p=0.0, max_len=8,
                                           noise_type="substitution", seed=42)

    def test_keys_and_single_tokenization(self):
        item = self.ds[0]
        self.assertEqual(set(item), {"input_ids", "attention_mask", "bbox",
                                     "labels", "token_quality_mask"})
        self.assertEqual(self.tok.calls, 1)

    def test_token_quality_clean(self):
        item = self.ds[0]
        q = item["token_quality_mask"]
        self.assertEqual(q[0].item(), 1.0)   # word token
        self.assertEqual(q[1].item(), 1.0)   # word token
        self.assertEqual(q[2].item(), 0.0)   # special/pad token (word_id None)


if __name__ == "__main__":
    unittest.main()
