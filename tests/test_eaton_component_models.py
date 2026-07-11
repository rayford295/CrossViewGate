from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from scripts.run_eaton_component_models import (
    CLASS_ORDER,
    PROBABILITY_COLUMNS,
    _nll,
    build_seed_prediction_export,
    fit_temperature,
)


def _eval_frame(ids: list[str], logits: list[list[float]]) -> pd.DataFrame:
    values = np.asarray(logits, dtype=float)
    return pd.DataFrame(
        {
            "sample_id": ids,
            "target": [0, 1, 2][: len(ids)],
            "logit_0": values[:, 0],
            "logit_1": values[:, 1],
            "logit_2": values[:, 2],
        }
    )


class EatonComponentModelRunnerTest(unittest.TestCase):
    def test_temperature_fit_reduces_nll_for_overconfident_wrong_logits(self) -> None:
        logits = np.asarray([[8.0, 0.0, 0.0], [8.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
        targets = np.asarray([0, 1, 2])
        temperature = fit_temperature(logits, targets)
        self.assertGreater(temperature, 1.0)
        self.assertLessEqual(_nll(logits, targets, temperature), _nll(logits, targets, 1.0))

    def test_seed_export_is_complete_calibrated_and_role_locked(self) -> None:
        ids = ["a", "b", "c"]
        street = _eval_frame(ids, [[4, 1, 0], [0, 4, 1], [0, 1, 4]])
        overhead = _eval_frame(ids, [[3, 2, 0], [0, 3, 2], [0, 2, 3]])
        manifest = pd.DataFrame(
            {
                "pair_id": ids,
                "spatial_block_id": ["x", "y", "z"],
                "label": [0, 1, 2],
            }
        )
        export = build_seed_prediction_export(
            seed=42,
            street=street,
            overhead=overhead,
            manifest=manifest,
            street_temperature=1.5,
            overhead_temperature=2.0,
        )
        self.assertEqual(export["pair_id"].tolist(), ids)
        self.assertEqual(set(export["protocol_role"]), {"study_development"})
        self.assertEqual(json.loads(export.iloc[0]["class_order"]), list(CLASS_ORDER))
        self.assertTrue(
            np.allclose(export[list(PROBABILITY_COLUMNS["street"])].sum(axis=1), 1.0)
        )
        self.assertTrue(
            np.allclose(export[list(PROBABILITY_COLUMNS["overhead"])].sum(axis=1), 1.0)
        )

    def test_seed_export_refuses_silent_sample_loss_and_other_roles(self) -> None:
        street = _eval_frame(["a", "b"], [[4, 1, 0], [0, 4, 1]])
        overhead = _eval_frame(["a"], [[3, 2, 0]])
        manifest = pd.DataFrame(
            {"pair_id": ["a", "b"], "spatial_block_id": ["x", "y"], "label": [0, 1]}
        )
        with self.assertRaisesRegex(ValueError, "not identical"):
            build_seed_prediction_export(
                seed=42,
                street=street,
                overhead=overhead,
                manifest=manifest,
                street_temperature=1.0,
                overhead_temperature=1.0,
            )
        with self.assertRaisesRegex(ValueError, "study_development"):
            build_seed_prediction_export(
                seed=42,
                street=street,
                overhead=street,
                manifest=manifest,
                street_temperature=1.0,
                overhead_temperature=1.0,
                protocol_role="spatial_confirmation",
            )


if __name__ == "__main__":
    unittest.main()
