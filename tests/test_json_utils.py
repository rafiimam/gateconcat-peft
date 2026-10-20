import io
import json
import unittest

from gateconcat.json_utils import dump_json, sanitize_for_json


def strict_loads(s):
    def reject(x):
        raise ValueError(f"non-strict JSON constant: {x}")
    return json.loads(s, parse_constant=reject)


class TestJsonUtils(unittest.TestCase):
    def test_sanitize_nested(self):
        out = sanitize_for_json({"a": float("nan"), "b": [float("inf"), 1.5],
                                 "c": {"d": float("-inf")}, "e": (1, 2)})
        self.assertEqual(out, {"a": None, "b": [None, 1.5], "c": {"d": None}, "e": [1, 2]})

    def test_dump_produces_strict_json(self):
        buf = io.StringIO()
        dump_json({"gate_auc_mean": float("nan"), "f1": 0.7542}, buf)
        parsed = strict_loads(buf.getvalue())
        self.assertIsNone(parsed["gate_auc_mean"])
        self.assertAlmostEqual(parsed["f1"], 0.7542)


if __name__ == "__main__":
    unittest.main()
