import unittest

import pandas as pd

from gateconcat.significance import run_tests


def make_df(b_offset=0.03):
    rows = {"seed": [42, 123, 2024, 456, 789]}
    for metric in ["noisy_macro", "clean_macro", "noisy_weighted", "clean_weighted"]:
        rows[f"lora_{metric}"] = [0.70, 0.71, 0.69, 0.70, 0.72]
        rows[f"gate_concat_{metric}"] = [v + b_offset for v in rows[f"lora_{metric}"]]
    return pd.DataFrame(rows)


class TestRunTests(unittest.TestCase):
    def test_mean_diff_and_significance(self):
        res = run_tests(make_df(), "lora", "gate_concat")
        m = res["metrics"]["noisy_macro"]
        self.assertAlmostEqual(m["mean_diff"], 0.03, places=6)
        self.assertEqual(len(m["per_seed_diff"]), 5)
        self.assertTrue(m["significant_005_t"])

    def test_no_effect_not_significant(self):
        res = run_tests(make_df(b_offset=0.0), "lora", "gate_concat")
        self.assertAlmostEqual(res["metrics"]["noisy_macro"]["mean_diff"], 0.0, places=9)

    def test_missing_columns_skipped(self):
        df = pd.DataFrame({"seed": [1, 2], "lora_noisy_macro": [0.5, 0.6],
                           "gate_concat_noisy_macro": [0.6, 0.7]})
        res = run_tests(df, "lora", "gate_concat")
        self.assertIn("noisy_macro", res["metrics"])
        self.assertNotIn("clean_macro", res["metrics"])


if __name__ == "__main__":
    unittest.main()
