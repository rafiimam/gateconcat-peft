"""Behavioural tests for the synthetic corruption transforms.

These pin the *published* corruption semantics: deterministic per
(seed, sample_idx), token-quality weights, and per-type effects.
"""
import unittest

from gateconcat.corruption import OCRNoiseTransform, ocr_confusion_sub, swap_two_boxes


class TestOCRConfusionSub(unittest.TestCase):
    def test_maps_first_matching_char(self):
        self.assertEqual(ocr_confusion_sub("hello"), "he1lo")  # l -> 1
        self.assertEqual(ocr_confusion_sub("O"), "0")
        self.assertEqual(ocr_confusion_sub("0"), "O")

    def test_no_match_unchanged(self):
        self.assertEqual(ocr_confusion_sub("xyz"), "xyz")


class TestTransform(unittest.TestCase):
    def setUp(self):
        self.words = ["hello", "world", "form"]
        self.boxes = [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]]

    def test_p_zero_is_identity_with_clean_weights(self):
        t = OCRNoiseTransform(p=0.0, noise_type="substitution", seed=42)
        words, q, boxes = t(self.words, self.boxes, sample_idx=0)
        self.assertEqual(words, self.words)
        self.assertEqual(q, [1.0] * 3)
        self.assertEqual(boxes, self.boxes)

    def test_deterministic_per_seed_and_sample(self):
        t = OCRNoiseTransform(p=1.0, noise_type="substitution", seed=42)
        a = t(self.words, self.boxes, sample_idx=3)
        b = t(self.words, self.boxes, sample_idx=3)
        self.assertEqual(a, b)

    def test_substitution_preserves_length_and_marks_corrupted(self):
        t = OCRNoiseTransform(p=1.0, noise_type="substitution", seed=42)
        words, q, _ = t(self.words, self.boxes, sample_idx=0)
        self.assertEqual([len(w) for w in words], [len(w) for w in self.words])
        self.assertEqual(q, [0.1] * 3)

    def test_insertion_grows_and_deletion_shrinks(self):
        ins = OCRNoiseTransform(p=1.0, noise_type="insertion", seed=1)
        words, _, _ = ins(["abcd"], [[0, 0, 1, 1]], sample_idx=0)
        self.assertEqual(len(words[0]), 5)
        dele = OCRNoiseTransform(p=1.0, noise_type="deletion", seed=1)
        words, _, _ = dele(["abcd"], [[0, 0, 1, 1]], sample_idx=0)
        self.assertEqual(len(words[0]), 3)
        words, _, _ = dele(["a"], [[0, 0, 1, 1]], sample_idx=0)
        self.assertEqual(words[0], "")

    def test_box_swap_swaps_and_uses_box_quality(self):
        t = OCRNoiseTransform(p=1.0, noise_type="box_swap", seed=42)
        _, q, boxes = t(self.words, self.boxes, sample_idx=0)
        self.assertNotEqual(boxes, self.boxes)
        self.assertEqual(sorted(map(tuple, boxes)), sorted(map(tuple, self.boxes)))
        self.assertEqual(q, [0.5] * 3)

    def test_unknown_noise_type_raises(self):
        t = OCRNoiseTransform(p=1.0, noise_type="nope", seed=42)
        with self.assertRaises(ValueError):
            t(self.words, self.boxes, sample_idx=0)

    def test_custom_weights_are_honoured(self):
        t = OCRNoiseTransform(p=1.0, noise_type="substitution", seed=42,
                              clean_weight=1.0, corrupted_weight=0.25)
        _, q, _ = t(self.words, self.boxes, sample_idx=0)
        self.assertEqual(q, [0.25] * 3)


class TestSwapTwoBoxes(unittest.TestCase):
    def test_too_few_boxes_unchanged(self):
        import numpy as np
        rng = np.random.RandomState(0)
        self.assertEqual(swap_two_boxes([[1, 2, 3, 4]], rng), [[1, 2, 3, 4]])


if __name__ == "__main__":
    unittest.main()
