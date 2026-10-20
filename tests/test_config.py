"""Guard the locked protocol: if these values change, the published
comparisons are no longer matched — the test should fail loudly."""
import unittest

from gateconcat.config import Config


class TestLockedProtocol(unittest.TestCase):
    def test_locked_values(self):
        self.assertEqual(Config.backbone_name, "microsoft/layoutlmv3-base")
        self.assertEqual((Config.lora_r, Config.lora_alpha, Config.lora_dropout), (16, 32, 0.1))
        self.assertEqual((Config.lr, Config.weight_decay), (2e-4, 0.01))
        self.assertEqual((Config.epochs, Config.batch_size), (15, 4))
        self.assertEqual(Config.seeds, [42, 123, 2024, 456, 789])
        self.assertEqual((Config.clean_token_weight, Config.corrupted_token_weight), (1.0, 0.1))
        self.assertEqual(Config.frozen_gate_value, 0.5)


if __name__ == "__main__":
    unittest.main()
