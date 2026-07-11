from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from crossview_conflict.decision.routing import (
    build_tile_priority,
    evaluate_route,
    normalize_routing_input,
    route_to_geojson,
    simulate_route,
)
from scripts.build_actionable_report import build_outputs, main


def _decision_frame(*, include_target: bool = True) -> pd.DataFrame:
    data: dict[str, list[object]] = {
        "sample_id": ["a", "b", "c", "d"],
        "lat": [26.000, 26.010, 26.020, 26.021],
        "lon": [-81.000, -81.010, -81.020, -81.021],
        "tile_id": ["tile-a", "tile-b", "tile-c", "tile-c"],
        "severity_probability": [0.1, 0.9, 0.4, 0.8],
        "risk_score": [0.2, 0.1, 0.9, 0.6],
        "disposition": ["accept", "accept", "defer_human", "defer_human"],
        "hard_conflict": [0, 0, 1, 1],
        "js_divergence": [0.01, 0.02, 0.5, 0.4],
    }
    if include_target:
        data["target"] = [0, 2, 1, 2]
    return pd.DataFrame(data)


class RoutingCoreTests(unittest.TestCase):
    def test_accept_without_authorization_metadata_fails_closed(self) -> None:
        frame = _decision_frame(include_target=False)
        frame["disposition"] = [" accept ", "auto_accept", "automatic", "monitor"]
        normalized, info = normalize_routing_input(frame)
        self.assertTrue((normalized["disposition"] == "defer_human").all())
        self.assertTrue(any("lacked both" in warning for warning in info.warnings))

    def test_non_risk_controlled_accept_is_overridden_to_defer(self) -> None:
        frame = _decision_frame(include_target=False)
        frame["risk_control_status"] = "risk-aware"
        normalized, info = normalize_routing_input(frame)
        self.assertTrue((normalized["disposition"] == "defer_human").all())
        self.assertTrue(any("Fail-closed override" in warning for warning in info.warnings))

    def test_fixed_k_and_random_policy_are_deterministic(self) -> None:
        normalized, _ = normalize_routing_input(_decision_frame())
        tiles_a = build_tile_priority(normalized, random_seed=17)
        tiles_b = build_tile_priority(normalized, random_seed=17)
        pd.testing.assert_series_equal(
            tiles_a["priority_random"], tiles_b["priority_random"], check_names=True
        )

        route_a = simulate_route(
            tiles_a,
            policy="random",
            max_stops=2,
            start_latitude=26.0,
            start_longitude=-81.0,
            random_seed=17,
        )
        route_b = simulate_route(
            tiles_b,
            policy="random",
            max_stops=2,
            start_latitude=26.0,
            start_longitude=-81.0,
            random_seed=17,
        )
        self.assertEqual(len(route_a.stops), 2)
        self.assertEqual(route_a.stops["tile_id"].tolist(), route_b.stops["tile_id"].tolist())
        self.assertEqual(
            route_a.stops["cumulative_distance_km"].tolist(),
            route_b.stops["cumulative_distance_km"].tolist(),
        )

    def test_distance_budget_is_never_exceeded(self) -> None:
        frame = pd.DataFrame(
            {
                "sample_id": ["origin", "near", "far"],
                "latitude": [0.0, 0.0, 0.0],
                "longitude": [0.0, 0.01, 0.02],
                "tile_id": ["a", "b", "c"],
                "severity_probability": [0.1, 0.5, 1.0],
                "risk_score": [0.1, 0.5, 1.0],
                "disposition": ["accept", "accept", "defer_human"],
                "hard_conflict": [0, 0, 1],
            }
        )
        normalized, _ = normalize_routing_input(frame)
        tiles = build_tile_priority(normalized)
        budget = 1.2
        route = simulate_route(
            tiles,
            policy="joint",
            distance_budget_km=budget,
            start_latitude=0.0,
            start_longitude=0.0,
        )
        self.assertGreaterEqual(len(route.stops), 1)
        self.assertLessEqual(route.total_distance_km, budget + 1e-9)
        self.assertTrue((route.stops["cumulative_distance_km"] <= budget + 1e-9).all())
        self.assertTrue((route.stops["remaining_distance_budget_km"] >= -1e-9).all())

    def test_geojson_uses_longitude_latitude_coordinate_order(self) -> None:
        normalized, _ = normalize_routing_input(_decision_frame())
        tiles = build_tile_priority(normalized)
        route = simulate_route(
            tiles,
            policy="severity_only",
            max_stops=1,
            start_latitude=26.0,
            start_longitude=-81.0,
        )
        geojson = route_to_geojson(route)
        first_stop = route.stops.iloc[0]
        self.assertEqual(
            geojson["features"][0]["geometry"]["coordinates"],
            [float(first_stop["longitude"]), float(first_stop["latitude"])],
        )
        self.assertEqual(geojson["properties"]["coordinate_order"], "longitude,latitude")
        self.assertFalse(geojson["properties"]["road_network_routing"])

    def test_label_metrics_are_descriptive_and_available_when_target_exists(self) -> None:
        normalized, info = normalize_routing_input(_decision_frame())
        self.assertTrue(info.labels_available)
        self.assertEqual(tuple(map(str, info.severe_labels)), ("2",))
        tiles = build_tile_priority(normalized)
        route = simulate_route(tiles, policy="severity_only", max_stops=2)
        metrics = evaluate_route(route, tiles)
        self.assertTrue(metrics["labels_available"])
        self.assertIsNotNone(metrics["severe_discoveries"])
        self.assertIsNotNone(metrics["severe_recall_at_budget"])
        self.assertIsNotNone(metrics["ndcg_at_budget"])


class ActionableReportTests(unittest.TestCase):
    def test_p02_compatibility_and_no_label_mode_write_all_artifacts(self) -> None:
        evidence = pd.DataFrame(
            {
                "sample_id": ["p0", "p1", "p2"],
                "latitude": [26.1, 26.2, 26.3],
                "longitude": [-81.1, -81.2, -81.3],
                "tile_id": ["x", "y", "z"],
                "gate3_probability_0": [0.8, 0.1, 0.2],
                "gate3_probability_1": [0.1, 0.2, 0.2],
                "gate3_probability_2": [0.1, 0.7, 0.6],
                "hard_conflict": [0, 1, 0],
                "js_divergence": [0.0, 0.4, 0.1],
            }
        )
        normalized, info = normalize_routing_input(evidence)
        self.assertFalse(info.labels_available)
        self.assertTrue((normalized["disposition"] == "evidence_only").all())
        self.assertTrue(info.risk_score_source.startswith("derived_1_minus_max"))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "evidence.csv"
            output_dir = root / "report"
            evidence.to_csv(input_path, index=False)
            summary = build_outputs(
                evidence,
                input_path=input_path,
                output_dir=output_dir,
                max_stops=2,
                random_seed=9,
            )
            self.assertFalse(summary["labels_available"])
            for filename in (
                "tile_priority.csv",
                "inspection_route.geojson",
                "actionable_report.md",
            ):
                self.assertTrue((output_dir / filename).is_file(), filename)
            report = (output_dir / "actionable_report.md").read_text(encoding="utf-8")
            self.assertIn("no target labels supplied", report)
            self.assertIn("not road-network routing", report)
            self.assertIn("n/a", report)
            route = json.loads(
                (output_dir / "inspection_route.geojson").read_text(encoding="utf-8")
            )
            self.assertEqual(route["properties"]["random_seed"], 9)

    def test_cli_accepts_fixed_stop_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "decisions.csv"
            output_dir = root / "outputs"
            _decision_frame(include_target=False).to_csv(input_path, index=False)
            exit_code = main(
                [
                    "--input-csv",
                    str(input_path),
                    "--output-dir",
                    str(output_dir),
                    "--policy",
                    "reliability_aware",
                    "--max-stops",
                    "1",
                ]
            )
            self.assertEqual(exit_code, 0)
            route = json.loads(
                (output_dir / "inspection_route.geojson").read_text(encoding="utf-8")
            )
            point_features = [
                feature for feature in route["features"] if feature["geometry"]["type"] == "Point"
            ]
            self.assertEqual(len(point_features), 1)


if __name__ == "__main__":
    unittest.main()
