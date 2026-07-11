from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.analyze_disagreement_anatomy import (
    add_anatomy_columns,
    cluster_bootstrap_ci,
    development_verdicts,
    positive_direction_share,
    probe_auroc,
    rank_biserial,
    visibility_median_difference,
)


def make_frame(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "protocol_role": "gate_fit",
        "spatial_block_id": "block_a",
        "building_ratio": 0.2,
        "center_building_ratio": 0.2,
        "center_minus_global": 0.0,
        "centroid_distance_norm": 0.5,
        "street_confidence": 0.6,
        "remote_confidence": 0.6,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


class AnatomyColumnTests(unittest.TestCase):
    def test_decidable_requires_exactly_one_correct_view(self) -> None:
        frame = add_anatomy_columns(
            make_frame(
                [
                    {"street_prediction": 0, "remote_prediction": 2, "target": 0},
                    {"street_prediction": 0, "remote_prediction": 2, "target": 1},
                    {"street_prediction": 1, "remote_prediction": 1, "target": 1},
                ]
            )
        )
        self.assertListEqual(frame["decidable"].tolist(), [True, False, False])
        self.assertListEqual(frame["signed_direction"].tolist(), [2, 2, 0])

    def test_positive_direction_share(self) -> None:
        frame = add_anatomy_columns(
            make_frame(
                [
                    {"street_prediction": 0, "remote_prediction": 2, "target": 0},
                    {"street_prediction": 2, "remote_prediction": 0, "target": 0},
                    {"street_prediction": 0, "remote_prediction": 1, "target": 0},
                    {"street_prediction": 1, "remote_prediction": 1, "target": 1},
                ]
            )
        )
        self.assertAlmostEqual(positive_direction_share(frame), 2.0 / 3.0)


class EffectTests(unittest.TestCase):
    def build_visibility_frame(self) -> pd.DataFrame:
        rows = []
        # Street-correct disagreements at high visibility, remote-correct at low.
        for idx in range(10):
            rows.append(
                {
                    "street_prediction": 0,
                    "remote_prediction": 2,
                    "target": 0,
                    "center_building_ratio": 0.8 + 0.01 * idx,
                    "spatial_block_id": f"block_{idx % 4}",
                }
            )
            rows.append(
                {
                    "street_prediction": 0,
                    "remote_prediction": 2,
                    "target": 2,
                    "center_building_ratio": 0.05 + 0.01 * idx,
                    "spatial_block_id": f"block_{idx % 4}",
                }
            )
        return add_anatomy_columns(make_frame(rows))

    def test_visibility_difference_and_rank_biserial_are_positive(self) -> None:
        frame = self.build_visibility_frame()
        self.assertGreater(visibility_median_difference(frame), 0.5)
        self.assertAlmostEqual(rank_biserial(frame), 1.0)

    def test_cluster_bootstrap_ci_brackets_point(self) -> None:
        frame = self.build_visibility_frame()
        rng = np.random.default_rng(7)
        point, low, high = cluster_bootstrap_ci(
            frame, visibility_median_difference, replicates=200, rng=rng
        )
        self.assertLessEqual(low, point)
        self.assertGreaterEqual(point, 0.5)
        self.assertGreaterEqual(high, point)

    def test_probe_auroc_learns_visibility_signal(self) -> None:
        frame = self.build_visibility_frame()
        auroc = probe_auroc(frame, frame, ["center_building_ratio"])
        self.assertGreater(auroc, 0.95)


class VerdictTests(unittest.TestCase):
    def test_development_verdicts_apply_seed_thresholds(self) -> None:
        per_seed = pd.DataFrame(
            {
                "h_a1_positive_direction_share": [0.7] * 5,
                "h_a1_positive_direction_share_ci_low": [0.55, 0.6, 0.58, 0.52, 0.45],
                "h_a1_positive_direction_share_ci_high": [0.8] * 5,
                "h_a2_visibility_median_diff_ci_low": [0.02, 0.01, 0.03, 0.05, -0.01],
                "h_a3_auroc_visibility_only_ci_low": [0.55, 0.52, 0.51, 0.49, 0.48],
            }
        )
        verdicts = development_verdicts(per_seed)
        self.assertTrue(verdicts["h_a1_directional_asymmetry"])
        self.assertTrue(verdicts["h_a2_visibility_conditions_correctness"])
        self.assertTrue(verdicts["h_a3_visibility_explains_held_out"])

    def test_verdicts_fail_below_threshold(self) -> None:
        per_seed = pd.DataFrame(
            {
                "h_a1_positive_direction_share": [0.7, 0.7, 0.3, 0.7, 0.7],
                "h_a1_positive_direction_share_ci_low": [0.55, 0.6, 0.1, 0.52, 0.51],
                "h_a1_positive_direction_share_ci_high": [0.8, 0.8, 0.45, 0.8, 0.8],
                "h_a2_visibility_median_diff_ci_low": [0.02, -0.01, -0.03, 0.05, -0.01],
                "h_a3_auroc_visibility_only_ci_low": [0.55, 0.49, 0.48, 0.47, 0.46],
            }
        )
        verdicts = development_verdicts(per_seed)
        self.assertFalse(verdicts["h_a1_directional_asymmetry"])
        self.assertFalse(verdicts["h_a2_visibility_conditions_correctness"])
        self.assertFalse(verdicts["h_a3_visibility_explains_held_out"])


if __name__ == "__main__":
    unittest.main()
