from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

from crossview_conflict.decision.risk_control import (
    assert_disjoint_protocol_splits,
    bounded_classification_loss,
    build_per_sample_decisions,
    calibrate_threshold,
    evaluate_locked_threshold,
    one_sided_risk_upper_bound,
    recall_at_review_budgets,
    risk_coverage_curve,
)
from scripts.eval_selective_triage import _audit_policy_identity, main


class BoundedLossTests(unittest.TestCase):
    def test_severe_miss_extreme_error_and_cost_weighted(self) -> None:
        targets = np.array([2, 2, 0, 0])
        predictions = np.array([1, 0, 2, 0])

        severe = bounded_classification_loss(
            targets,
            predictions,
            loss_name="severe_miss",
            severe_classes=[2],
        )
        extreme = bounded_classification_loss(
            targets,
            predictions,
            loss_name="extreme_error",
            extreme_distance=2,
        )
        cost = bounded_classification_loss(
            np.array([0, 1, 2]),
            np.array([2, 0, 1]),
            loss_name="cost_weighted",
            cost_matrix=[[0, 1, 3], [2, 0, 1], [4, 2, 0]],
            max_cost=4,
        )

        np.testing.assert_array_equal(severe, [1.0, 1.0, 0.0, 0.0])
        np.testing.assert_array_equal(extreme, [0.0, 1.0, 1.0, 0.0])
        np.testing.assert_allclose(cost, [0.75, 0.5, 0.5])
        self.assertTrue(((cost >= 0.0) & (cost <= 1.0)).all())

    def test_cost_bound_is_not_silently_clipped(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_cost"):
            bounded_classification_loss(
                [0],
                [1],
                loss_name="cost_weighted",
                cost_matrix=[[0, 2], [1, 0]],
                max_cost=1,
            )
        with self.assertRaisesRegex(ValueError, "complete and square"):
            bounded_classification_loss(
                [0],
                [0],
                loss_name="cost_weighted",
                cost_matrix={"0": {"0": 0, "1": 1}, "1": {"1": 0}},
            )


class SplitProtocolTests(unittest.TestCase):
    def test_gate_identity_audit_rejects_partial_missing_values(self) -> None:
        frames = {
            "gate_fit": pd.DataFrame({"gate_artifact_id": ["A", None]}),
            "risk_calibration": pd.DataFrame({"gate_artifact_id": ["A", "A"]}),
            "final_test": pd.DataFrame({"gate_artifact_id": ["A", "A"]}),
        }
        with self.assertRaisesRegex(ValueError, "missing or blank"):
            _audit_policy_identity(frames)

    def test_three_roles_must_be_disjoint(self) -> None:
        gate = pd.DataFrame({"dataset": ["ian"], "sample_id": ["shared"]})
        calibration = pd.DataFrame({"dataset": ["ian"], "sample_id": ["cal"]})
        test = pd.DataFrame({"dataset": ["ian"], "sample_id": ["shared"]})

        with self.assertRaisesRegex(ValueError, "overlap"):
            assert_disjoint_protocol_splits(gate, calibration, test)

        clean_test = pd.DataFrame({"dataset": ["ian"], "sample_id": ["test"]})
        audit = assert_disjoint_protocol_splits(gate, calibration, clean_test)
        self.assertTrue(audit["disjoint"])
        self.assertEqual(audit["id_columns"], ["dataset", "sample_id"])

    def test_role_column_cannot_fake_disjointness(self) -> None:
        frames = [pd.DataFrame({"role": [role]}) for role in ("fit", "cal", "test")]
        with self.assertRaisesRegex(ValueError, "cannot serve"):
            assert_disjoint_protocol_splits(*frames, id_columns=["role"])


class CalibrationAndMetricsTests(unittest.TestCase):
    def test_calibration_selects_maximum_certified_coverage(self) -> None:
        result = calibrate_threshold(
            risk_scores=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            losses=np.array([0.0, 0.0, 0.0, 0.0, 1.0]),
            threshold_grid=[0.2, 0.4, 0.5],
            alpha=0.6,
            delta=0.1,
            bound_method="auto",
            loss_is_binary=True,
        )

        self.assertEqual(result.status, "risk-controlled")
        self.assertEqual(result.bound_method, "clopper_pearson")
        self.assertEqual(result.selected_threshold, 0.4)
        self.assertEqual(result.selected_count, 4)
        self.assertLessEqual(result.upper_bound, 0.6)
        self.assertEqual(int(result.threshold_table["selected"].sum()), 1)

    def test_severe_fnr_uses_only_severe_targets_as_risk_denominator(self) -> None:
        # One severe target is missed while 99 non-severe targets are correct.
        # Population incidence would be 1%, but conditional severe FNR is 100%.
        scores = np.zeros(100)
        losses = np.r_[1.0, np.zeros(99)]
        severe_targets = np.r_[True, np.zeros(99, dtype=bool)]
        result = calibrate_threshold(
            scores,
            losses,
            threshold_grid=[0.0],
            alpha=0.10,
            delta=0.05,
            loss_is_binary=True,
            risk_denominator_mask=severe_targets,
        )
        self.assertEqual(result.status, "risk-aware")
        self.assertIsNone(result.selected_threshold)

    def test_cluster_aware_bound_uses_group_count_not_row_count(self) -> None:
        result = calibrate_threshold(
            np.zeros(100),
            np.zeros(100),
            threshold_grid=[0.0],
            alpha=0.10,
            delta=0.05,
            loss_is_binary=True,
            risk_group_ids=np.repeat(["a", "b", "c", "d"], 25),
        )
        self.assertEqual(result.status, "risk-aware")
        self.assertEqual(result.threshold_table["risk_group_count"].tolist(), [4])
        self.assertEqual(result.bound_method, "hoeffding")

    def test_grouped_calibration_requires_grouped_locked_test(self) -> None:
        calibration = calibrate_threshold(
            np.zeros(20),
            np.zeros(20),
            threshold_grid=[0.0],
            alpha=0.9,
            delta=0.2,
            loss_is_binary=True,
            risk_group_ids=[f"g{index}" for index in range(20)],
        )
        self.assertTrue(calibration.grouping_required)
        with self.assertRaisesRegex(ValueError, "requires risk_group_ids"):
            evaluate_locked_threshold(
                calibration,
                np.zeros(10),
                np.zeros(10),
                protocol_audit_complete=True,
            )

    def test_final_test_has_no_threshold_search_and_cross_event_is_risk_aware(self) -> None:
        calibration = calibrate_threshold(
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0, 0, 0, 0, 1],
            threshold_grid=[0.2, 0.4, 0.5],
            alpha=0.6,
            delta=0.1,
            loss_is_binary=True,
        )
        held_out = evaluate_locked_threshold(
            calibration,
            risk_scores=[0.05, 0.1, 0.7, 0.8],
            losses=[0, 0, 1, 1],
            protocol_audit_complete=True,
        )
        stress_test = evaluate_locked_threshold(
            calibration,
            risk_scores=[0.05, 0.1, 0.7, 0.8],
            losses=[0, 0, 1, 1],
            protocol_audit_complete=True,
            claim_scope="cross_event_stress_test",
        )

        self.assertEqual(held_out.selected_threshold, calibration.selected_threshold)
        self.assertEqual(held_out.selected_count, 2)
        self.assertEqual(stress_test.status, "risk-aware")
        self.assertIn("cross_event", stress_test.status_reason)

    def test_hoeffding_bound_risk_coverage_and_recall_budgets(self) -> None:
        bound = one_sided_risk_upper_bound(
            [0.0, 0.5, 1.0], delta=0.2, method="hoeffding"
        )
        expected = min(1.0, 0.5 + np.sqrt(np.log(5.0) / 6.0))
        self.assertAlmostEqual(bound, expected)

        curve, aurc = risk_coverage_curve([0.1, 0.2, 0.3], [0, 1, 1])
        np.testing.assert_allclose(curve["empirical_risk"], [0.0, 0.5, 2.0 / 3.0])
        self.assertAlmostEqual(aurc, (0.0 + 0.5 + 2.0 / 3.0) / 3.0)

        recalls = recall_at_review_budgets(
            np.arange(10, dtype=float),
            [False, False, False, False, False, False, False, True, True, True],
            budgets=[5, 10, 20, 30, 50],
        )
        self.assertEqual(recalls["review_count"].tolist(), [1, 1, 2, 3, 5])
        np.testing.assert_allclose(recalls["recall"], [1 / 3, 1 / 3, 2 / 3, 1, 1])

    def test_tied_scores_are_row_order_invariant_and_not_split(self) -> None:
        curve_a, aurc_a = risk_coverage_curve([0.5, 0.5], [0, 1])
        curve_b, aurc_b = risk_coverage_curve([0.5, 0.5], [1, 0])
        self.assertAlmostEqual(aurc_a, aurc_b)
        self.assertEqual(curve_a["accepted_count"].tolist(), [2])
        self.assertEqual(curve_b["accepted_count"].tolist(), [2])

        recall_a = recall_at_review_budgets([0.5, 0.5], [False, True], budgets=[50])
        recall_b = recall_at_review_budgets([0.5, 0.5], [True, False], budgets=[50])
        self.assertEqual(recall_a["review_count"].tolist(), [2])
        self.assertEqual(recall_b["review_count"].tolist(), [2])
        self.assertEqual(recall_a["recall"].tolist(), recall_b["recall"].tolist())

    def test_decision_skeleton_contains_required_operational_columns(self) -> None:
        calibration = calibrate_threshold(
            [0.1, 0.2, 0.9, 1.0],
            [0, 0, 1, 1],
            threshold_grid=[0.2],
            alpha=0.9,
            delta=0.2,
            loss_is_binary=True,
        )
        test_result = evaluate_locked_threshold(
            calibration, [0.1, 0.8], [0, 1], protocol_audit_complete=True
        )
        decisions = build_per_sample_decisions(
            pd.DataFrame({"sample_id": ["a", "b"], "prediction": [0, 1]}),
            [0.1, 0.8],
            test_result,
            losses=[0, 1],
        )

        for column in (
            "decision_source",
            "disposition",
            "acquisition_target",
            "reason",
            "risk_control_status",
        ):
            self.assertIn(column, decisions.columns)
        self.assertEqual(decisions["disposition"].tolist(), ["accept", "defer_human"])

    def test_risk_aware_policy_fails_closed_even_below_threshold(self) -> None:
        calibration = calibrate_threshold(
            [0.1, 0.2, 0.9, 1.0],
            [0, 0, 1, 1],
            threshold_grid=[0.2],
            alpha=0.9,
            delta=0.2,
            loss_is_binary=True,
        )
        stress_result = evaluate_locked_threshold(
            calibration,
            [0.1, 0.8],
            [0, 1],
            protocol_audit_complete=True,
            claim_scope="cross_event_stress_test",
        )
        decisions = build_per_sample_decisions(
            pd.DataFrame({"sample_id": ["a", "b"]}),
            [0.1, 0.8],
            stress_result,
        )
        self.assertEqual(decisions["threshold_eligible"].tolist(), [1, 0])
        self.assertEqual(decisions["policy_authorized"].tolist(), [0, 0])
        self.assertEqual(decisions["accepted"].tolist(), [0, 0])
        self.assertEqual(decisions["disposition"].tolist(), ["defer_human", "defer_human"])
        self.assertTrue(decisions["reason"].str.startswith("policy_not_risk_controlled").all())


class SelectiveTriageCliTests(unittest.TestCase):
    def test_cli_writes_locked_protocol_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gate_path = root / "gate.csv"
            calibration_path = root / "calibration.csv"
            test_path = root / "test.csv"
            output_dir = root / "outputs"
            pd.DataFrame(
                {
                    "sample_id": [f"g{index}" for index in range(5)],
                    "tile_id": [f"g-tile-{index}" for index in range(5)],
                    "dataset": ["ian"] * 5,
                    "gate_artifact_id": ["fixed-gate"] * 5,
                    "latitude": [0.0] * 5,
                    "longitude": [0.0] * 5,
                }
            ).to_csv(
                gate_path, index=False
            )
            pd.DataFrame(
                {
                    "sample_id": [f"c{index}" for index in range(20)],
                    "target": [2] * 20,
                    "prediction": [2] * 20,
                    "risk_score": np.linspace(0.0, 0.9, 20),
                    "tile_id": [f"c-tile-{index}" for index in range(20)],
                    "dataset": ["ian"] * 20,
                    "gate_artifact_id": ["fixed-gate"] * 20,
                    "latitude": [10.0] * 20,
                    "longitude": [0.0] * 20,
                }
            ).to_csv(calibration_path, index=False)
            pd.DataFrame(
                {
                    "sample_id": [f"t{index}" for index in range(10)],
                    "target": [2] * 10,
                    "prediction": [2] * 10,
                    "risk_score": np.linspace(0.0, 0.9, 10),
                    "tile_id": [f"t-tile-{index}" for index in range(10)],
                    "dataset": ["ian"] * 10,
                    "gate_artifact_id": ["fixed-gate"] * 10,
                    "latitude": [20.0] * 10,
                    "longitude": [0.0] * 10,
                }
            ).to_csv(test_path, index=False)

            with redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        "--gate-fit-csv",
                        str(gate_path),
                        "--risk-calibration-csv",
                        str(calibration_path),
                        "--final-test-csv",
                        str(test_path),
                        "--output-dir",
                        str(output_dir),
                        "--severe-labels",
                        "2",
                        "--alpha",
                        "0.5",
                        "--delta",
                        "0.05",
                        "--threshold-grid",
                        "0.5,1.0",
                        "--spatial-separation-m",
                        "25",
                    ]
                )

            self.assertEqual(exit_code, 0)
            for filename in (
                "summary.json",
                "calibration_threshold_grid.csv",
                "risk_coverage.csv",
                "recall_at_review_budgets.csv",
                "per_sample_decision.csv",
                "split_audit.json",
            ):
                self.assertTrue((output_dir / filename).is_file(), filename)
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["selected_threshold"], 1.0)
            self.assertEqual(summary["calibration"]["selected_count"], 20)
            self.assertEqual(summary["final_test"]["selected_count"], 10)
            self.assertEqual(summary["status"], "risk-controlled")
            decisions = pd.read_csv(output_dir / "per_sample_decision.csv")
            self.assertTrue(
                {"decision_source", "disposition", "acquisition_target", "reason"}.issubset(
                    decisions.columns
                )
            )

    def test_cross_event_metadata_forces_stress_scope_and_fail_closed_actions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for role, prefix, dataset, count in (
                ("gate", "g", "ian", 8),
                ("calibration", "c", "ian", 20),
                ("test", "t", "milton", 10),
            ):
                path = root / f"{role}.csv"
                frame = pd.DataFrame(
                    {
                        "sample_id": [f"{prefix}{index}" for index in range(count)],
                        "tile_id": [f"{prefix}-tile-{index}" for index in range(count)],
                        "dataset": [dataset] * count,
                        "gate_artifact_id": ["fixed-gate"] * count,
                        "latitude": [
                            {"g": 0.0, "c": 10.0, "t": 20.0}[prefix]
                        ] * count,
                        "longitude": [0.0] * count,
                    }
                )
                if role != "gate":
                    frame["target"] = 2
                    frame["prediction"] = 2
                    frame["risk_score"] = np.linspace(0.0, 0.9, count)
                frame.to_csv(path, index=False)
                paths[role] = path
            output_dir = root / "output"
            with redirect_stdout(io.StringIO()):
                main(
                    [
                        "--gate-fit-csv", str(paths["gate"]),
                        "--risk-calibration-csv", str(paths["calibration"]),
                        "--final-test-csv", str(paths["test"]),
                        "--output-dir", str(output_dir),
                        "--severe-labels", "2",
                        "--alpha", "0.9",
                        "--delta", "0.2",
                        "--threshold-grid", "1.0",
                        "--spatial-separation-m", "25",
                    ]
                )
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["claim"]["scope"], "cross_event_stress_test")
            self.assertEqual(summary["status"], "risk-aware")
            decisions = pd.read_csv(output_dir / "per_sample_decision.csv")
            self.assertTrue((decisions["disposition"] == "defer_human").all())


if __name__ == "__main__":
    unittest.main()
