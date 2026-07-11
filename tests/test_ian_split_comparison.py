from __future__ import annotations

import unittest

import pandas as pd

from scripts.compare_ian_split_protocols import summarize_comparison


class IanSplitComparisonTests(unittest.TestCase):
    def test_seed_matched_differences_have_expected_sign(self) -> None:
        rows = []
        for protocol, offset in (("legacy", 0.0), ("repaired_spatial", -0.1)):
            for seed in (1, 2, 3):
                rows.append(
                    {
                        "protocol": protocol,
                        "mode": "crossview",
                        "seed": seed,
                        "accuracy": 0.8 + offset,
                        "macro_f1": 0.75 + offset,
                        "severe_recall": 0.9 + offset,
                    }
                )
        summary, differences = summarize_comparison(
            pd.DataFrame(rows), bootstrap_replicates=200, bootstrap_seed=7
        )
        self.assertEqual(len(summary), 2)
        self.assertTrue((differences["repaired_minus_legacy_mean"] < 0).all())
        self.assertTrue((differences["bootstrap_95_high"] < 0).all())


if __name__ == "__main__":
    unittest.main()
